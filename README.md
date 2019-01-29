# P3D-CTN
The P3D-CTN is a framework for spatio-temporal action detection. It integrates the frame-based two-dimensional convolutional module with the video-based P3D convolutional module. 

A two step manner. First, tube proposals are generated from P3D-module, and box proposals are produced from 2d-module based on the tube proposals.
## Installation
1. Just follow Caffe standard installation instructions.  
2. Run P3D-module setup.py to build fundamental enviroment

        python P3D-module/setup.py
        
## Datasets
Download three benchmark datasets(JHMDB, UCF101, UCFSports).   
Use the scripts on P3D-module/datasets to generate the data format for training

## P3D-module
### Training  
P3D_cls_train.sh and P3D_loc_train.sh are used for training P3D-module  

    sh P3D-module/P3D_cls_train.sh
    sh P3D-module/P3D_loc_train.sh
    
### Testing  
P3D_cls_eval.py and P3D_loc_eval.py are used for testing P3D-module  

        python P3D-module/P3D_cls_eval.py
        python P3D-module/P3D_loc_eval.py
        
## 2D-module
### Traning  
ll.sh(JHMDB), ll_101.sh(UCF101), ll_sports.sh(UCFSports) are used for training 2D-module

        sh 2D-module/ll.sh
        sh 2D-module/ll_101.sh
        sh 2D-module/ll_sports.sh

### Evaluating  
action_tools/jhmdb_eval.py  ucfsports_eval.py  ucf101_eval.py are used for evaluating 2D-module(frame-AP, video-AP)

        python 2D-module/action_tools/jhmdb_eval.py --proto 2D-module//models/JHMDB/VGG_16/test_1.prototxt --net 2D-module/output/faster_rcnn_end2end/JHMDB_RGB_1_split_0/RGB_1_VGG_16_iter_70000.caffemodel --imdb JHMDB_RGB_1_split_0 --out 2D-module/action_results/jhmdb.pkl
        python 2D-module/action_tools/ucf101_eval.py --proto 2D-module//models/UCF101/VGG_16/test_1.prototxt --net 2D-module/output/faster_rcnn_end2end/UCF101_RGB_1_split_0/RGB_1_VGG_16_iter_100000.caffemodel --imdb UCF101_RGB_1_split_0 --out 2D-module/action_results/ucf101.pkl
        python 2D-module/action_tools/ucfsports_eval.py --proto 2D-module//models/UCFSports/VGG_16/test_1.prototxt --net 2D-module/output/faster_rcnn_end2end/UCFSports_RGB_1_split_0/RGB_1_VGG_16_iter_70000.caffemodel --imdb UCFSports_RGB_1_split_0 --out 2D-module/action_results/ucfsports.pkl
