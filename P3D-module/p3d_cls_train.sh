set -e
export PYTHONPATH=/data/wjc/TCNN_STCNN
LOG=/data/wjc/TCNN_STCNN/log/toi/p3d-au-toi-cls22-log-`date +%Y-%m-%d-%H-%M-%S`.log
./caffe/build/tools/caffe train -gpu 8 -solver models/jhmdb/toi_rec_p3d_solver_v22.prototxt -weights model_kinetics/p3d_resnet_kinetics_iter_190000.caffemodel 2>&1  | tee $LOG $@
