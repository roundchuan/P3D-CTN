from data_layers.toi_frame_data_eval_p3d import RegDataLayer
import cPickle
import numpy as np

if __name__ == '__main__':
  net = '/data/wjc/tmp_tcnn/model_kinetics/reg_p3d_resnet_jhmdb_eval.prototxt'
  model = '/data/wjc/TCNN_STCNN/models/toi_p3d/p3d_frame_300_400_110_p3d_4k_lr1_iter_50000.caffemodel'
  r = RegDataLayer(net, model)
  i = 0
  flag = False
  b = []
  while not(flag):
    det = np.load('/data/wjc/TCNN_STCNN/toi_p3d/p3d_cls_{}_toi_123.npy'.format(i))
    [flag, a] = r.forward(det)
    b.append(a)
    i += 1
  with open('/data/wjc/TCNN_STCNN/p3d_126.pkl', 'wb') as fid:
    cPickle.dump(b, fid, cPickle.HIGHEST_PROTOCOL)
