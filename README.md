# LATTE: Layer-stacked Attention for Heterogeneous Graph Embedding

This code is provided to reproduce experiments in the paper "Layer-stacked Attention for Heterogeneous Graph Embedding".
Running this code requires installation of Python 3.X packages listed in `requirements.txt`. Install them by running:
> pip install -r requirements.txt

It also requires installation of the COGDL Toolkit (version 0.1.2) from the following GitHub
repository: https://github.com/THUDM/cogdl
To install it, run:
> pip install cogdl==0.1.2

To perform either transductive or inductive experiment runs for a method on a dataset, go to the root repository and run
the command:
> python run.py --method [MetaPath2Vec, HAN, GTN, LATTE-1, LATTE-2, LATTE-2proximity] --dataset [ACM, DBLP, IMDB] --inductive [True, False] --num_gpus [0, 1]

The wandb logger will generate a link for you inspect the performance metrics as it trains. If prompted, choose the
option "(1) Private W&B dashboard, no account required".

The models' hyperparameters have been kept as described in the paper. If you'd like to inspect LATTE's algorithm & code,
open the `conv.py` file and the `LATTENodeClassifier` class in the `methods.py` file.

Happy Reviewing!
--Author