
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationLoss(nn.Module):
    def __init__(self, n_classes: int, class_weight: torch.Tensor = None, multilabel=True, use_hierar=False,
                 loss_type="SOFTMAX_CROSS_ENTROPY", hierar_penalty=1e-6, hierar_relations=None):
        super(ClassificationLoss, self).__init__()
        self.n_classes = n_classes
        self.loss_type = loss_type
        self.hierar_penalty = hierar_penalty
        self.hierar_relations = hierar_relations
        self.multilabel = multilabel
        self.use_hierar = use_hierar
        print(f"INFO: Using {loss_type}")
        print(f"class_weight for {class_weight.shape} classes") if class_weight is not None else None

        if loss_type == "SOFTMAX_CROSS_ENTROPY":
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        elif loss_type == "NEGATIVE_LOG_LIKELIHOOD":
            self.criterion = torch.nn.NLLLoss(class_weight)
        elif loss_type == "BCE_WITH_LOGITS":
            self.criterion = torch.nn.BCEWithLogitsLoss(weight=class_weight)
        elif loss_type == "BCE":
            self.criterion = torch.nn.BCELoss(weight=class_weight)
        elif loss_type == "MULTI_LABEL_MARGIN":
            self.criterion = torch.nn.MultiLabelMarginLoss(weight=class_weight)
        elif loss_type == "KL_DIVERGENCE":
            self.criterion = torch.nn.KLDivLoss()
        else:
            raise TypeError(f"Unsupported loss type:{loss_type}")

    def forward(self, logits, target, linear_weight: torch.Tensor = None):
        if self.use_hierar:
            assert self.loss_type in ["BCE_WITH_LOGITS",
                                      "SIGMOID_FOCAL_CROSS_ENTROPY"]
            assert linear_weight is not None
            if not self.multilabel:
                target = torch.eye(self.n_classes)[target]

            return self.criterion(logits, target.type_as(logits)) + \
                   self.hierar_penalty * self.recursive_regularize(linear_weight, self.hierar_relations)
        else:
            if self.multilabel:
                assert self.loss_type in ["BCE_WITH_LOGITS", "BCE",
                                          "SIGMOID_FOCAL_CROSS_ENTROPY", "MULTI_LABEL_MARGIN"]
                target = target.type_as(logits)
            else:
                if self.loss_type not in ["SOFTMAX_CROSS_ENTROPY", "NEGATIVE_LOG_LIKELIHOOD",
                                          "SOFTMAX_FOCAL_CROSS_ENTROPY"]:
                    target = torch.eye(self.n_classes, device=logits.device, dtype=torch.long)[target]
            return self.criterion.forward(logits, target)

