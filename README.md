# CoMPT

Source code for our IJCAI 2021 paper [Learning Attributed Graph Representation with Communicative Message Passing Transformer](#)

The code was built based on [Molecule Attention Transformer](https://github.com/ardigen/MAT) and [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html). Thanks a lot for their code sharing!

## Dependencies

+ cuda >= 9.0
+ cuDNN
+ RDKit == 2020.03.4
+ torch >= 1.4.0 (please upgrade your torch version in order to reduce the training time)
+ numpy == 1.19.1
+ scikit-learn == 0.23.2
+ tqdm == 4.52.0

Tips: Using code `conda install -c conda-forge rdkit` can help you install package RDKit quickly.

## Dataset

| Dataset | Tasks | Type | Molecule | Metric | 
| :---: | :---: | :---: | :---: | :---: |
| bbbp | 1 | Graph Classification | 2,035 | ROC-AUC |
| tox21 | 12 | Graph Classification | 7,821 | ROC-AUC |
| sider | 27 | Graph Classification | 1,379 | ROC-AUC |
| clintox | 2 | Graph Classification | 1,468 | ROC-AUC |
| esol | 1 | Graph Regression | 1,128 | RMSE |
| freesolv | 1 | Graph Regression | 642 | RMSE |
| lipophilicity | 1 | Graph Regression | 4,198 | RMSE |
| 1H-NMR | 1 | Node Regression | 12,800 | MAE |
| 13C-NMR | 1 | Node Regression | 26,859 | MAE |

## Preprocess

For the Graph-level task (Graph classification, Graph Regression), you can download the source dataset from [Molecule-Net](http://moleculenet.ai/datasets-1). 

For the Node-level task (Node Regression), you can download the source dataset from [NMRShiftDB2](https://nmrshiftdb.nmr.uni-koeln.de/portal/js_pane/P-Help), or use a preprocess dataset cleaned by [nmr-mpnn](https://github.com/seokhokang/nmr_mpnn), thanks a lot for their code sharing!

In the folder `./Data`, we have preprocessed every mentioned dataset by the corresponding jupyter notebook. All source datasets can be refered in the `./Data/<dataset>/source/`, and all preprocess files can be refered in the `./Data/<dataset>/preprocess/`.

You can also run the corresponding jupyter notebook in the path `./Data/<dataset>/preprocessing.ipynb` to generate the `<dataset>.pickle` files.

## Training

To train a graph-level task, run:

`python train_graph.py --seed <seed> --gpu <gpu> --fold 5 --dataset <dataset> --split <split>`

where `<seed>` is the seed number, `<gpu>` is the gpu index number, `<dataset>` is the graph-level dataset name (bbbp, tox21, sider, clintox, esol, freesolv, lipophilicity), `<split>` is the split method that mentioned by [Molecule-Net](http://moleculenet.ai/datasets-1) (random, scaffold, cv).

To train a node-level task, run:

`python train_node.py --seed <seed> --gpu <gpu> --dataset nmrshiftdb --element <element>`

where `<seed>` is the seed number, `<gpu>` is the gpu index number, `<element>` is the element name(1H for 1H-NMR, 13C for 13C-NMR).

All hyperparameters can be tuned in the `utils.py`

## Todo

- [ ] Clean the unuse function and write more comments.
- [ ] Replace the unnoticed Chinese comments in English.
- [ ] Generate the split-fold files in `.csv` format, rewrite the code and then make a bash script to train all folds in parallel.
- [ ] Make a suitable padding way to adapt the molecules with more than 100 atoms, which will be used in the protein (long period).
- [ ] Try our best to reduce the training time and the using memory, especially for the large dataset (long period).


## Citation

Please cite the following paper if you use this code in your work.

[#1771 Learning Attributed Graph Representation with Communicative Message Passing Transformer](https://ijcai-21.org/program-main-track/)
