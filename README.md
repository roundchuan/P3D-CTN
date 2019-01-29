# P3D-CTN

A two step manner. First, tube proposals are generated from P3D-module, and box proposals are produced from 2d-module based on the tube proposals.
## Installation
1. Just follow Caffe standard installation instructions.  
2. Run P3D-module setup.py to build fundamental enviroment

## Datasets
Download three benchmark datasets(JHMDB, UCF101, UCFSports).   
Use the scripts on P3D-module/datasets to generate the data format for training

## P3D-module
### Training  
P3D_cls_train.sh and P3D_loc_train.sh are used for training P3D-module  
### Testing  
P3D_cls_eval.py and P3D_loc_eval.py are used for testing P3D-module 

## 2D-module
### Traning  
ll.sh(JHMDB), ll_101.sh(UCF101), ll_sports.sh(UCFSports) are used for training 2D-module  
### Evaluating  
action_tools/jhmdb_eval.py  ucfsports_eval.py  ucf101_eval.py are used for testing 2D-module
