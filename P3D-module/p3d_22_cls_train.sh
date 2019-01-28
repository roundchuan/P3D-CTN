#!/usr/bin/env bash
set -e
export PYTHONPATH=/data/wjc/TCNN_STCNN
LOG=/data/wjc/TCNN_STCNN/log/toi/p3d-au-toi-cls22-log-`date +%Y-%m-%d-%H-%M-%S`.log
./caffe/build/tools/caffe train -gpu 8 -solver /data/wjc/TCNN_STCNN/models/jhmdb/toi_rec_p3d_solver_v22.prototxt -weights /data/wjc/tmp_tcnn/model_kinetics/p3d_resnet_kinetics_iter_190000.caffemodel 2>&1  | tee $LOG $@
 #/home/rhou/models/sports_frame_300_400_pre_iter_4000.caffemodel
