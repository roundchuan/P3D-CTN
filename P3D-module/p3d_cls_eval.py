from data_layers.toi_rec_data_eval import RecDataLayer
import numpy as np

if __name__ == '__main__':
  net = 'model_kinetics/toi_cls_p3d_resnet_jhmdb_eval.prototxt'
  model = 'models/toi_p3d/toi_cls22_300_400_129_p3d_iter_50000.caffemodel'
  r = RecDataLayer(net, model)
  flag = False
  i = 0
  while not(flag):
    [curr, flag] = r.forward()
    np.save('/data/wjc/TCNN_STCNN/p3d_cls_results/rec_{}_p3d_130.npy'.format(i), curr)
    i += 1
