#!/usr/bin/env bash
set -e
export PYTHONPATH=/data/wjc/TCNN_STCNN
LOG=/data/wjc/TCNN_STCNN/log/toi/p3d-frame-log-`date +%Y-%m-%d-%H-%M-%S`.log
./caffe/build/tools/caffe train -gpu 7 -solver /data/wjc/TCNN_STCNN/models/jhmdb/p3d_frame_solver.prototxt -weights models/toi_p3d/toi_cls22_300_400_au_122_p3d_nms_iter_6000.caffemodel 2>&1  | tee $LOG $@
 #/home/rhou/models/sports_frame_300_400_pre_iter_4000.caffemodel
