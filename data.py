import logging
from collections import OrderedDict, defaultdict
from typing import Union, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import torch
from cogdl.datasets.gtn_data import GTNDataset, ACM_GTNDataset, DBLP_GTNDataset, IMDB_GTNDataset
from cogdl.datasets.han_data import HANDataset, ACM_HANDataset, DBLP_HANDataset, IMDB_HANDataset
from ogb.nodeproppred import PygNodePropPredDataset, DglNodePropPredDataset
from sklearn.cluster import KMeans
from torch import Tensor
from torch.utils import data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.sampler import Adj, EdgeIndex
from torch_geometric.utils.hetero import group_hetero_graph
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor, coalesce, transpose

from conv import is_negative


def load_node_dataset(dataset, method, hparams, train_ratio=None, dir_path="~/datasets/"):
    if dataset == "ACM":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborGenerator(ACM_HANDataset(), [25, 20], node_types=["P"],
                                              metapaths=["PAP", "PSP"] if "LATTE" in method else None,
                                              add_reverse_metapaths=True,
                                              head_node_type="P", inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborGenerator(ACM_GTNDataset(), [25, 20], node_types=["P", "A", "S"],
                                              metapaths=["PA", "AP", "PS", "SP"],
                                              add_reverse_metapaths=False,
                                              head_node_type="P", inductive=hparams.inductive)
        dataset.x_dict["S"] = dataset.x_dict["P"]
        dataset.x_dict["A"] = dataset.x_dict["P"]

    elif dataset == "DBLP":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborGenerator(DBLP_HANDataset(), [25, 20],
                                              node_types=["A"], head_node_type="A", metapaths=None,
                                              add_reverse_metapaths=True,
                                              inductive=hparams.inductive)
        elif "LATTE" in method:
            dataset = HeteroNeighborGenerator(DBLP_HANDataset(), neighbor_sizes=[25, 20],
                                              node_types=["A", "P", "C", "T"], head_node_type="A",
                                              metapaths=["AC", "AP", "AT"],
                                              add_reverse_metapaths=True, inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborGenerator(DBLP_GTNDataset(), [25, 20], node_types=["A"], head_node_type="A",
                                              metapaths=["APA", "ApA", "ACA", "AcA"] if "LATTE" in method else None,
                                              add_reverse_metapaths=False,
                                              inductive=hparams.inductive)
        dataset.x_dict["P"] = dataset.x_dict["A"]
        dataset.x_dict["C"] = dataset.x_dict["A"]
        dataset.x_dict["T"] = dataset.x_dict["A"]

    elif dataset == "IMDB":
        if method == "HAN" or method == "MetaPath2Vec":
            dataset = HeteroNeighborGenerator(IMDB_HANDataset(), [25, 20], node_types=["M"],
                                              metapaths=["MAM", "MDM", "MWM"] if "LATTE" in method else None,
                                              add_reverse_metapaths=True,
                                              head_node_type="M",
                                              inductive=hparams.inductive)
        else:
            dataset = HeteroNeighborGenerator(IMDB_GTNDataset(), neighbor_sizes=[25, 20],
                                              node_types=["M", "A", "D"],
                                              metapaths=["MD", "DM", "MA", "AM"],
                                              add_reverse_metapaths=False,
                                              head_node_type="M", inductive=hparams.inductive)
        dataset.x_dict["A"] = dataset.x_dict["M"]
        dataset.x_dict["D"] = dataset.x_dict["M"]
    else:
        raise Exception(f"dataset {dataset} not found")
    return dataset


class Network:
    def get_networkx(self):
        if not hasattr(self, "G"):
            G = nx.Graph()
            for metapath in self.edge_index_dict:
                edgelist = self.edge_index_dict[metapath].t().numpy().astype(str)
                edgelist = np.core.defchararray.add([metapath[0][0], metapath[-1][0]], edgelist)
                edge_type = "".join([n for i, n in enumerate(metapath) if i % 2 == 1])
                G.add_edges_from(edgelist, edge_type=edge_type)

            self.G = G

        return self.G

    def get_projection_pos(self, embeddings_all, UMAP: classmethod, n_components=2):
        pos = UMAP(n_components=n_components).fit_transform(embeddings_all)
        pos = {embeddings_all.index[i]: pair for i, pair in enumerate(pos)}
        return pos

    def get_embeddings_labels(self, h_dict: dict, global_node_index: dict, cache=True):
        if hasattr(self, "embeddings") and hasattr(self, "ntypes") and hasattr(self, "labels") and cache:
            return self.embeddings, self.ntypes, self.labels

        # Building a dataframe of embeddings, indexed by "{node_type}{node_id}"
        emb_df_list = []
        for ntype in self.node_types:
            nid = global_node_index[ntype].cpu().numpy().astype(str)
            n_type_id = np.core.defchararray.add(ntype[0], nid)

            if isinstance(h_dict[ntype], torch.Tensor):
                df = pd.DataFrame(h_dict[ntype].detach().cpu().numpy(), index=n_type_id)
            else:
                df = pd.DataFrame(h_dict[ntype], index=n_type_id)
            emb_df_list.append(df)

        embeddings = pd.concat(emb_df_list, axis=0)
        ntypes = embeddings.index.to_series().str.slice(0, 1)

        # Build vector of labels for all node types
        if hasattr(self, "y_dict") and len(self.y_dict) > 0:
            labels = pd.Series(
                self.y_dict[self.head_node_type][global_node_index[self.head_node_type]].squeeze(-1).numpy(),
                index=emb_df_list[0].index,
                dtype=str)
        else:
            labels = None

        # Save results
        self.embeddings, self.ntypes, self.labels = embeddings, ntypes, labels

        return embeddings, ntypes, labels

    def predict_cluster(self, n_clusters=8, n_jobs=-2, save_kmeans=False, seed=None):
        kmeans = KMeans(n_clusters, n_jobs=n_jobs, random_state=seed)
        logging.info(f"Kmeans with k={n_clusters}")
        y_pred = kmeans.fit_predict(self.embeddings)
        if save_kmeans:
            self.kmeans = kmeans

        y_pred = pd.Series(y_pred, index=self.embeddings.index, dtype=str)
        return y_pred


class HeteroNetDataset(torch.utils.data.Dataset, Network):
    def __init__(self, dataset, node_types=None, metapaths=None, head_node_type=None, directed=True,
                 resample_train: float = None, add_reverse_metapaths=True, inductive=True):
        """
        This class handles processing of the data & train/test spliting.
        :param dataset:
        :param node_types:
        :param metapaths:
        :param head_node_type:
        :param directed:
        :param resample_train:
        :param add_reverse_metapaths:
        """
        self.dataset = dataset
        self.directed = directed
        self.use_reverse = add_reverse_metapaths
        self.node_types = node_types
        self.head_node_type = head_node_type
        self.inductive = inductive

        # PyTorchGeometric Dataset

        if isinstance(dataset, PygNodePropPredDataset) and not hasattr(dataset[0], "edge_index_dict"):
            print("PygNodePropPredDataset Homogenous (use HeteroNeighborGenerator class)")
            self.process_PygNodeDataset_homo(dataset)
        elif isinstance(dataset, PygNodePropPredDataset) and hasattr(dataset[0], "edge_index_dict"):
            print("PygNodePropPredDataset Hetero (use HeteroNeighborGenerator class)")
            self.process_PygNodeDataset_hetero(dataset)
        elif isinstance(dataset, DglNodePropPredDataset):
            print("DGLNodePropPredDataset Hetero")
            self.process_DglNodeDataset_hetero(dataset)


        elif isinstance(dataset, InMemoryDataset):
            print("InMemoryDataset")
            self.process_inmemorydataset(dataset, train_ratio=0.5)
        elif isinstance(dataset, HANDataset) or isinstance(dataset, GTNDataset):
            print(f"{dataset.__class__.__name__}")
            self.process_COGDLdataset(dataset, metapaths, node_types, resample_train)
        elif "blogcatalog6k" in dataset:
            self.process_BlogCatalog6k(dataset, train_ratio=0.5)
        else:
            raise Exception(f"Unsupported dataset {dataset}")

        if hasattr(self, "y_dict"):
            if self.y_dict[self.head_node_type].dim() > 1 and self.y_dict[self.head_node_type].size(-1) != 1:
                self.multilabel = True
                self.classes = torch.arange(self.y_dict[self.head_node_type].size(1))
                self.class_counts = self.y_dict[self.head_node_type].sum(0)
            else:
                self.multilabel = False

                mask = self.y_dict[self.head_node_type] != -1
                labels = self.y_dict[self.head_node_type][mask]
                self.classes = labels.unique()

                if self.y_dict[self.head_node_type].dim() > 1:
                    labels = labels.squeeze(-1).numpy()
                else:
                    labels = labels.numpy()
                self.class_counts = pd.Series(labels).value_counts(sort=False)

            self.n_classes = self.classes.size(0)
            self.class_weight = torch.true_divide(1, torch.tensor(self.class_counts, dtype=torch.float))

            assert -1 not in self.classes
            assert self.class_weight.numel() == self.n_classes, f"self.class_weight {self.class_weight.numel()}, n_classes {self.n_classes}"
        else:
            self.multilabel = False
            self.n_classes = None
            print("WARNING: Dataset doesn't have node label (y_dict attribute).")

        assert hasattr(self, "num_nodes_dict")

        if not hasattr(self, "x_dict") or len(self.x_dict) == 0:
            self.x_dict = {}

        if resample_train is not None and resample_train > 0:
            self.resample_training_idx(resample_train)
        else:
            print("train_ratio", self.get_train_ratio())
        self.train_ratio = self.get_train_ratio()

    def name(self):
        if not hasattr(self, "_name"):
            return self.dataset.__class__.__name__
        else:
            return self._name

    @property
    def node_attr_shape(self):
        if not hasattr(self, "x_dict") or len(self.x_dict) == 0:
            node_attr_shape = {}
        else:
            node_attr_shape = {k: v.size(1) for k, v in self.x_dict.items()}
        return node_attr_shape

    @property
    def node_attr_size(self):
        node_feat_sizes = np.unique(list(self.node_attr_shape.values()))
        if len(node_feat_sizes) == 1:
            in_features = node_feat_sizes[0]
        else:
            raise Exception(
                f"Must use self.node_attr_shape as node types have different feature sizes. {node_feat_sizes}")

        return in_features

    def split_train_val_test(self, train_ratio, sample_indices=None):
        if sample_indices is not None:
            indices = sample_indices[torch.randperm(sample_indices.size(0))]
        else:
            indices = torch.randperm(self.num_nodes_dict[self.head_node_type])

        num_indices = indices.size(0)
        training_idx = indices[:int(num_indices * train_ratio)]
        validation_idx = indices[int(num_indices * train_ratio):]
        testing_idx = indices[int(num_indices * train_ratio):]
        return training_idx, validation_idx, testing_idx

    def resample_training_idx(self, train_ratio):
        all_idx = torch.cat([self.training_idx, self.validation_idx, self.testing_idx])
        self.training_idx, self.validation_idx, self.testing_idx = \
            self.split_train_val_test(train_ratio=train_ratio, sample_indices=all_idx)
        print(f"Resampled training set at {self.get_train_ratio()}%")

    def get_metapaths(self, khop=False):
        """
        Returns original metapaths including reverse metapaths if use_reverse
        :return:
        """
        metapaths = self.metapaths
        if self.use_reverse:
            metapaths = metapaths + self.get_reverse_metapaths(self.metapaths, self.edge_index_dict)

        return metapaths

    def get_reverse_metapaths(self, metapaths, edge_index_dict) -> list:
        reverse_metapaths = []
        for metapath in metapaths:
            reverse = self.reverse_metapath_name(metapath, edge_index_dict)
            reverse_metapaths.append(reverse)
        return reverse_metapaths

    def get_num_nodes_dict(self, edge_index_dict):
        num_nodes_dict = {}
        for keys, edge_index in edge_index_dict.items():
            key = keys[0]
            N = int(edge_index[0].max() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))

            key = keys[-1]
            N = int(edge_index[1].max() + 1)
            num_nodes_dict[key] = max(N, num_nodes_dict.get(key, N))
        return num_nodes_dict

    def get_node_id_dict(self, edge_index_dict):
        node_ids_dict = {}
        for metapath, edge_index in edge_index_dict.items():
            node_ids_dict.setdefault(metapath[0], []).append(edge_index[0])
            node_ids_dict.setdefault(metapath[-1], []).append(edge_index[1])

        for ntype in node_ids_dict:
            node_ids_dict[ntype] = torch.cat(node_ids_dict[ntype], 0).unique()

        return node_ids_dict

    def add_reverse_edge_index(self, edge_index_dict) -> None:
        reverse_edge_index_dict = {}
        for metapath in edge_index_dict:
            if is_negative(metapath) or edge_index_dict[metapath] == None: continue
            reverse_metapath = self.reverse_metapath_name(metapath, edge_index_dict)

            reverse_edge_index_dict[reverse_metapath] = transpose(index=edge_index_dict[metapath], value=None,
                                                                  m=self.num_nodes_dict[metapath[0]],
                                                                  n=self.num_nodes_dict[metapath[-1]])[0]
        edge_index_dict.update(reverse_edge_index_dict)

    def reverse_metapath_name(self, metapath, edge_index_dict: dict = None):
        if isinstance(metapath, tuple):
            reverse_metapath = tuple(etype + "_by" if i == 1 else etype \
                                     for i, etype in enumerate(reversed(metapath)))
        elif isinstance(metapath, str):
            reverse_metapath = "".join(reversed(metapath))
            if reverse_metapath in self.edge_index_dict:
                logging.info(f"Reversed metapath {reverse_metapath} already exists in {self.edge_index_dict.keys()}")
                # reverse_metapath = reverse_metapath[:1] + "_" + reverse_metapath[1:]

        elif isinstance(metapath, (int, np.int)):
            reverse_metapath = str(metapath) + "_"
        else:
            raise NotImplementedError(f"{metapath} not supported")
        return reverse_metapath


    @staticmethod
    def sps_adj_to_edgeindex(adj):
        adj = adj.tocoo(copy=False)
        return torch.tensor(np.vstack((adj.row, adj.col)).astype("long"))

    def process_COGDLdataset(self, dataset: HANDataset, metapath, node_types, train_ratio):
        data = dataset.data
        assert self.head_node_type is not None
        assert node_types is not None
        print(f"Edge_types: {len(data['adj'])}")
        self.node_types = node_types
        if metapath is not None:
            self.edge_index_dict = {metapath: data["adj"][i][0] for i, metapath in enumerate(metapath)}
        else:
            self.edge_index_dict = {f"{self.head_node_type}{i}{self.head_node_type}": data["adj"][i][0] \
                                    for i in range(len(data["adj"]))}
        self.edge_types = list(range(dataset.num_edge))
        self.metapaths = list(self.edge_index_dict.keys())
        self.x_dict = {self.head_node_type: data["x"]}
        self.in_features = data["x"].size(1)

        self.training_idx, self.training_target = data["train_node"], data["train_target"]
        self.validation_idx, self.validation_target = data["valid_node"], data["valid_target"]
        self.testing_idx, self.testing_target = data["test_node"], data["test_target"]

        self.y_index_dict = {self.head_node_type: torch.cat([self.training_idx, self.validation_idx, self.testing_idx])}
        self.num_nodes_dict = self.get_num_nodes_dict(self.edge_index_dict)

        # Create new labels vector for all nodes, with -1 for nodes without label
        self.y_dict = {
            self.head_node_type: torch.cat([self.training_target, self.validation_target, self.testing_target])}

        new_y_dict = {nodetype: -torch.ones(self.num_nodes_dict[nodetype] + 1).type_as(self.y_dict[nodetype]) \
                      for nodetype in self.y_dict}
        for node_type in self.y_dict:
            new_y_dict[node_type][self.y_index_dict[node_type]] = self.y_dict[node_type]
        self.y_dict = new_y_dict

        if self.inductive:
            other_nodes = torch.arange(self.num_nodes_dict[self.head_node_type])
            idx = ~np.isin(other_nodes, self.training_idx) & \
                  ~np.isin(other_nodes, self.validation_idx) & \
                  ~np.isin(other_nodes, self.testing_idx)
            other_nodes = other_nodes[idx]
            self.training_subgraph_idx = torch.cat(
                [self.training_idx, torch.tensor(other_nodes, dtype=self.training_idx.dtype)],
                dim=0).unique()

        self.data = data


    def process_inmemorydataset(self, dataset: InMemoryDataset, train_ratio):
        data = dataset[0]
        self.edge_index_dict = data.edge_index_dict
        self.num_nodes_dict = data.num_nodes_dict
        if self.node_types is None:
            self.node_types = list(data.num_nodes_dict.keys())
        self.y_dict = data.y_dict
        self.y_index_dict = data.y_index_dict

        new_y_dict = {nodetype: -torch.ones(self.num_nodes_dict[nodetype] + 1).type_as(self.y_dict[nodetype]) for
                      nodetype in self.y_dict}
        for node_type in self.y_dict:
            new_y_dict[node_type][self.y_index_dict[node_type]] = self.y_dict[node_type]
        self.y_dict = new_y_dict

        self.metapaths = list(self.edge_index_dict.keys())
        assert train_ratio is not None
        self.training_idx, self.validation_idx, self.testing_idx = \
            self.split_train_val_test(train_ratio,
                                      sample_indices=self.y_index_dict[self.head_node_type])

    def train_dataloader(self, collate_fn=None, batch_size=128, num_workers=12, **kwargs):
        loader = data.DataLoader(self.training_idx, batch_size=batch_size,
                                 shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) else self.get_collate_fn(collate_fn,
                                                                                                        mode="train",
                                                                                                        **kwargs))
        return loader

    def trainvalidtest_dataloader(self, collate_fn=None, batch_size=None, num_workers=12, **kwargs):
        all_idx = torch.cat([self.training_idx, self.validation_idx, self.testing_idx])
        loader = data.DataLoader(all_idx, batch_size=all_idx.shape[0],
                                 shuffle=True, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) else self.get_collate_fn(collate_fn,
                                                                                                        mode="validation",
                                                                                                        **kwargs))
        return loader

    def valid_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, **kwargs):
        loader = data.DataLoader(self.validation_idx, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) else self.get_collate_fn(collate_fn,
                                                                                                        mode="validation",
                                                                                                        **kwargs))
        return loader

    def test_dataloader(self, collate_fn=None, batch_size=128, num_workers=4, **kwargs):
        loader = data.DataLoader(self.testing_idx, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn if callable(collate_fn) else self.get_collate_fn(collate_fn,
                                                                                                        mode="testing",
                                                                                                        **kwargs))
        return loader

    def get_collate_fn(self, collate_fn: str, mode=None, **kwargs):

        def collate_wrapper(iloc):
            if "HAN_batch" in collate_fn:
                return self.collate_HAN_batch(iloc, mode=mode)
            elif "HAN" in collate_fn:
                return self.collate_HAN(iloc, mode=mode)
            elif "collate_HGT_batch" in collate_fn:
                return self.collate_HGT_batch(iloc, mode=mode)
            else:
                raise Exception(f"Correct collate function {collate_fn} not found.")

        return collate_wrapper

    def filter_edge_index(self, input, allowed_nodes):
        if isinstance(input, tuple):
            edge_index = input[0]
            values = edge_index[1]
        else:
            edge_index = input
            values = None

        mask = np.isin(edge_index[0], allowed_nodes) & np.isin(edge_index[1], allowed_nodes)
        edge_index = edge_index[:, mask]

        if values == None:
            values = torch.ones(edge_index.size(1))
        else:
            values = values[mask]

        return (edge_index, values)

    def collate_HAN(self, iloc, mode=None):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        if "train" in mode:
            filter = True if self.inductive else False
            if self.inductive and hasattr(self, "training_subgraph_idx"):
                allowed_nodes = self.training_subgraph_idx
            else:
                allowed_nodes = self.training_idx
        elif "valid" in mode:
            filter = True if self.inductive else False
            if self.inductive and hasattr(self, "training_subgraph_idx"):
                allowed_nodes = torch.cat([self.validation_idx, self.training_subgraph_idx])
            else:
                allowed_nodes = self.validation_idx
        elif "test" in mode:
            filter = False
            allowed_nodes = self.testing_idx
        else:
            filter = False
            print("WARNING: should pass a value in `mode` in collate_HAN()")

        if isinstance(self.dataset, HANDataset):
            X = {"adj": [(edge_index, values) \
                             if not filter else self.filter_edge_index((edge_index, values), allowed_nodes) \
                         for edge_index, values in self.data["adj"][:len(self.metapaths)]],
                 "x": self.data["x"] if hasattr(self.data, "x") else None,
                 "idx": iloc}
        else:
            X = {
                "adj": [(self.edge_index_dict[i], torch.ones(self.edge_index_dict[i].size(1))) \
                            if not filter else self.filter_edge_index(self.edge_index_dict[i], allowed_nodes) \
                        for i in self.metapaths],
                "x": self.data["x"] if hasattr(self.data, "x") else None,
                "idx": iloc}

        X["adj"] = [edge for edge in X["adj"] if edge[0].size(1) > 0]

        X["global_node_index"] = torch.arange(X["x"].shape[0])

        y = self.y_dict[self.head_node_type][iloc]
        return X, y, None

    def collate_HAN_batch(self, iloc, mode=None):
        if not isinstance(iloc, torch.Tensor):
            iloc = torch.tensor(iloc)

        X_batch, y, weights = self.sample(iloc, mode=mode)  # uses HeteroNetSampler PyG sampler method

        X = {}
        X["adj"] = [(X_batch["edge_index_dict"][metapath], torch.ones(X_batch["edge_index_dict"][metapath].size(1))) \
                    for metapath in self.metapaths if metapath in X_batch["edge_index_dict"]]
        X["x"] = self.data["x"][X_batch["global_node_index"][self.head_node_type]]
        X["idx"] = X_batch["global_node_index"][self.head_node_type]

        X["global_node_index"] = X_batch["global_node_index"]  # Debugging purposes

        return X, y, weights

    def collate_HGT_batch(self, iloc, mode=None):
        X_batch, y, weights = self.sample(iloc, mode=mode)  # uses HeteroNetSampler PyG sampler method

        X = {}
        X["node_inp"] = torch.vstack([X_batch["x_dict"][ntype] for ntype in self.node_types])
        X["node_type"] = torch.hstack([nid * torch.ones((X_batch["x_dict"][ntype].shape[0],), dtype=int) \
                                       for nid, ntype in enumerate(self.node_types)])
        # assert X["node_inp"].shape[0] == X["node_type"].shape[0]

        X["edge_index"] = torch.hstack([X_batch["edge_index_dict"][metapath] \
                                        for metapath in self.metapaths if metapath in X_batch["edge_index_dict"]])
        X["edge_type"] = torch.hstack([eid * torch.ones(X_batch["edge_index_dict"][metapath].shape[1], dtype=int) \
                                       for eid, metapath in enumerate(self.metapaths) if
                                       metapath in X_batch["edge_index_dict"]])
        # assert X["edge_index"].shape[1] == X["edge_type"].shape[0]

        X["global_node_index"] = X_batch["global_node_index"]  # Debugging purposes
        X["edge_time"] = None

        return X, y, weights

    def get_train_ratio(self):
        if self.validation_idx.size() != self.testing_idx.size() or not (self.validation_idx == self.testing_idx).all():
            train_ratio = self.training_idx.numel() / \
                          sum([self.training_idx.numel(), self.validation_idx.numel(), self.testing_idx.numel()])
        else:
            train_ratio = self.training_idx.numel() / sum([self.training_idx.numel(), self.validation_idx.numel()])
        return train_ratio


class HeteroNeighborGenerator(HeteroNetDataset):
    def __init__(self, dataset, neighbor_sizes, node_types=None, metapaths=None, head_node_type=None, edge_dir=True,
                 resample_train=None, add_reverse_metapaths=True, inductive=False):
        self.neighbor_sizes = neighbor_sizes
        super(HeteroNeighborGenerator, self).__init__(dataset, node_types, metapaths, head_node_type, edge_dir,
                                                      resample_train, add_reverse_metapaths, inductive)

        if self.use_reverse:
            self.add_reverse_edge_index(self.edge_index_dict)

        self.graph_sampler = NeighborSampler(neighbor_sizes, self.edge_index_dict, self.num_nodes_dict,
                                             self.node_types, self.head_node_type)

    def get_collate_fn(self, collate_fn: str, mode=None):
        assert mode is not None, "Must pass arg `mode` at get_collate_fn(). {'train', 'valid', 'test'}"

        def default_sampler(iloc):
            return self.sample(iloc, mode=mode)

        def khop_sampler(iloc):
            return self.khop_sampler(iloc, mode=mode)

        if "neighbor_sampler" in collate_fn or collate_fn is None:
            return default_sampler
        elif "khop_sampler" == collate_fn:
            return khop_sampler
        else:
            return super().get_collate_fn(collate_fn, mode=mode)

    def sample(self, n_idx, mode):
        if not isinstance(n_idx, torch.Tensor) and not isinstance(n_idx, dict):
            n_idx = torch.tensor(n_idx)

        # Sample subgraph
        batch_size, n_id, adjs = self.graph_sampler.sample(n_idx)

        # Sample neighbors and return `sampled_local_nodes` as the set of all nodes traversed (in local index)
        sampled_local_nodes = self.graph_sampler.get_nodes_dict(adjs, n_id)

        # Ensure the sampled nodes only either belongs to training, validation, or testing set
        if "train" in mode:
            filter = True if self.inductive else False
            if self.inductive and hasattr(self, "training_subgraph_idx"):
                allowed_nodes = self.training_subgraph_idx
            else:
                allowed_nodes = self.training_idx
        elif "valid" in mode:
            filter = True if self.inductive else False
            if self.inductive and hasattr(self, "training_subgraph_idx"):
                allowed_nodes = torch.cat([self.validation_idx, self.training_subgraph_idx])
            else:
                allowed_nodes = self.validation_idx
        elif "test" in mode:
            filter = False
            allowed_nodes = self.testing_idx
        else:
            raise Exception(f"Must set `mode` to either 'training', 'validation', or 'testing'. mode={mode}")

        if filter:
            node_mask = np.isin(sampled_local_nodes[self.head_node_type], allowed_nodes)
            sampled_local_nodes[self.head_node_type] = sampled_local_nodes[self.head_node_type][node_mask]

        # `global_node_index` here actually refers to the 'local' type-specific index of the original graph
        X = {"edge_index_dict": {},
             "global_node_index": sampled_local_nodes,
             "x_dict": {}}

        X["edge_index_dict"] = self.graph_sampler.get_edge_index_dict(adjs=adjs,
                                                                      n_id=n_id,
                                                                      sampled_local_nodes=sampled_local_nodes,
                                                                      filter_nodes=filter)

        # x_dict attributes
        if hasattr(self, "x_dict") and len(self.x_dict) > 0:
            X["x_dict"] = {node_type: self.x_dict[node_type][X["global_node_index"][node_type]] \
                           for node_type in self.x_dict if node_type in X["global_node_index"]}

        # y_dict
        if hasattr(self, "y_dict") and len(self.y_dict) > 1:
            y = {node_type: y_true[X["global_node_index"][node_type]] \
                 for node_type, y_true in self.y_dict.items()}
        elif hasattr(self, "y_dict"):
            y = self.y_dict[self.head_node_type][X["global_node_index"][self.head_node_type]].squeeze(-1)
        else:
            y = None

        # Weights
        weights = (y != -1) if y.dim() == 1 else (y != -1).all(1)
        weights = weights & np.isin(X["global_node_index"][self.head_node_type], allowed_nodes)
        weights = torch.tensor(weights, dtype=torch.float)

        # Higher weights for sampled focal nodes in `n_idx`
        seed_node_idx = np.isin(X["global_node_index"][self.head_node_type], n_idx, invert=True)
        weights[seed_node_idx] = weights[seed_node_idx] * 0.2 if "train" in mode else 0.0

        return X, y, weights


class HeteroNeighborSampler(torch.utils.data.DataLoader):
    def __init__(self, edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int],
                 node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None,
                 return_e_id: bool = True,
                 **kwargs):

        """
        Args:
            edge_index:
            sizes:
            node_idx:
            num_nodes:
            return_e_id (bool):
            **kwargs:
        """
        self.sizes = sizes
        self.return_e_id = return_e_id
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        edge_index = edge_index.to('cpu')
        if not self.is_sparse_tensor:
            num_nodes = maybe_num_nodes(edge_index, num_nodes)
            value = torch.arange(edge_index.size(1)) if return_e_id else None

            # Sampling source_to_target
            self.adj_t = SparseTensor(row=edge_index[1],
                                      col=edge_index[0],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super(HeteroNeighborSampler, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        """
        Args:
            batch:
        """
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        batch_size: int = len(batch)

        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor:
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([row, col], dim=0)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        return batch_size, n_id, adjs


class NeighborSampler():
    def __init__(self, neighbor_sizes, edge_index_dict, num_nodes_dict, node_types, head_node_type):
        """
        Args:
            neighbor_sizes:
            edge_index_dict:
            num_nodes_dict:
            node_types:
            head_node_type:
        """
        self.head_node_type = head_node_type

        # Ensure head_node_type is first item in num_nodes_dict, since NeighborSampler.sample() function takes in index only the first
        num_nodes_dict = OrderedDict(
            [(node_type, num_nodes_dict[node_type]) for node_type in node_types])

        self.edge_index, self.edge_type, self.node_type, self.local_node_idx, self.local2global, self.key2int = \
            group_hetero_graph(edge_index_dict, num_nodes_dict)

        self.int2node_type = {type_int: node_type for node_type, type_int in self.key2int.items() if
                              node_type in node_types}
        self.int2edge_type = {type_int: edge_type for edge_type, type_int in self.key2int.items() if
                              edge_type in edge_index_dict}

        self.neighbor_sampler = HeteroNeighborSampler(self.edge_index, node_idx=None,
                                                      sizes=neighbor_sizes, batch_size=128,
                                                      shuffle=True)

    def sample(self, node_ids: dict):
        """
        Args:
            node_ids (dict):
        """
        local_node_idx = self.get_global_nidx(node_ids)

        batch_size, n_id, adjs = self.neighbor_sampler.sample(batch=local_node_idx)
        if not isinstance(adjs, list):
            adjs = [adjs]
        return batch_size, n_id, adjs

    def get_global_nidx(self, node_ids):
        """
        Args:
            node_ids:
        """
        if isinstance(node_ids, dict):
            n_idx_to_sample = torch.cat([self.local2global[ntype][nid] for ntype, nid in node_ids.items()], dim=0)
        else:
            n_idx_to_sample = self.local2global[self.head_node_type][node_ids]

        return n_idx_to_sample

    def get_nodes_dict(self, adjs: List[EdgeIndex], n_id):
        """
        Args:
            adjs:
            n_id:
        """
        sampled_nodes = {}
        for adj in adjs:
            for i in [0, 1]:
                node_ids = n_id[adj.edge_index[i]]
                node_types = self.node_type[node_ids]

                for node_type_id in node_types.unique():
                    mask = node_types == node_type_id
                    local_node_ids = self.local_node_idx[node_ids[mask]]
                    sampled_nodes.setdefault(self.int2node_type[node_type_id.item()], []).append(local_node_ids)

        # Concatenate & remove duplicate nodes
        sampled_nodes = {k: torch.cat(v, dim=0).unique() for k, v in sampled_nodes.items()}
        return sampled_nodes

    def get_edge_index_dict(self, adjs: List[EdgeIndex], n_id, sampled_local_nodes: dict, filter_nodes: bool):
        """Conbine all edge_index's across multiple layers and convert local node id to "batch node
        index" that aligns with `x_dict` and `global_node_index`

        Args:
            adjs:
            n_id:
            sampled_local_nodes (dict):
            filter_nodes (bool):
        """
        relabel_nodes = self.get_nid_relabel_dict(sampled_local_nodes)

        edge_index_dict = {}
        for adj in adjs:
            for edge_type_id in self.edge_type[adj.e_id].unique():
                metapath = self.int2edge_type[edge_type_id.item()]
                head, tail = metapath[0], metapath[-1]

                # Filter edges to correct edge_type_id
                edge_mask = self.edge_type[adj.e_id] == edge_type_id
                edge_index = adj.edge_index[:, edge_mask]

                # convert from "sampled_edge_index" to global index
                edge_index[0] = n_id[edge_index[0]]
                edge_index[1] = n_id[edge_index[1]]

                old_local_node_idx = self.local_node_idx.clone()
                # Convert node global index -> local index -> batch index
                if head == tail:
                    edge_index = self.local_node_idx[edge_index].apply_(relabel_nodes[head].get)
                else:
                    edge_index[0] = self.local_node_idx[edge_index[0]].apply_(lambda x: relabel_nodes[head].get(x, -1))
                    edge_index[1] = self.local_node_idx[edge_index[1]].apply_(lambda x: relabel_nodes[tail].get(x, -1))

                assert torch.isclose(old_local_node_idx.sum(), self.local_node_idx.sum())
                # Remove edges labeled as -1, which contain nodes not in sampled_local_nodes
                mask = np.isin(edge_index, [-1], assume_unique=False, invert=True).all(axis=0)
                edge_index = edge_index[:, mask]
                if edge_index.size(1) == 0: continue

                edge_index_dict.setdefault(metapath, []).append(edge_index)

        # Join edges from the adjs (from iterative layer-wise sampling)
        edge_index_dict = {metapath: torch.cat(e_index_list, dim=1) \
                           for metapath, e_index_list in edge_index_dict.items()}

        # Ensure no duplicate edges in each metapath
        edge_index_dict = {metapath: coalesce(index=edge_index,
                                              value=torch.ones_like(edge_index[0], dtype=torch.float),
                                              m=sampled_local_nodes[metapath[0]].size(0),
                                              n=sampled_local_nodes[metapath[-1]].size(0),
                                              op="add")[0] \
                           for metapath, edge_index in edge_index_dict.items()}

        return edge_index_dict

    def get_nid_relabel_dict(self, node_ids_dict):
        """
        Args:
            node_ids_dict:
        """
        relabel_nodes = {node_type: defaultdict(lambda: -1,
                                                dict(zip(node_ids_dict[node_type].numpy(),
                                                         range(node_ids_dict[node_type].size(0))))) \
                         for node_type in node_ids_dict}

        # relabel_nodes = {ntype: pd.Series(data=np.arange(node_ids_dict[ntype].size(0)),
        #                                   index=node_ids_dict[ntype].numpy()) \
        #                  for ntype in node_ids_dict}
        return relabel_nodes
