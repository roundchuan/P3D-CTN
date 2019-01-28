set -e
#export PYTHONPATH=/data/wjc/TCNN_STCNN
LOG=log-`date +%Y-%m-%d-%H-%M-%S`.log
./action_experiments/scripts/train_action_det.sh 2 VGG_16 UCF-Sports 1 RGB 0 2>&1  | tee $LOG $@
 #/home/rhou/models/sports_frame_300_400_pre_iter_4000.caffemodel
#./action_experiments/scripts/train_action_det.sh 0 VGG_16 UCF-Sports 1 RGB 0
