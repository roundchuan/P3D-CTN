from data_layers.toi_frame_data_eval_p3d_rgbflow import RegDataLayer
import cPickle
import numpy as np

if __name__ == '__main__':
  net = '/data/wjc/tmp_tcnn/model_kinetics/reg_p3d_resnet_jhmdb_eval.prototxt'
  model = '/data/wjc/TCNN_STCNN/models/toi_p3d/p3d_frame_300_400_110_p3d_4k_iter_45000.caffemodel'#/home/rhou/models/sports_frame_300_400_v4_iter_1000.caffemodel'
  r = RegDataLayer(net, model)
  i = 0
  flag = False
  b = []
  while not(flag):
    det = np.load('/data/wjc/TCNN_STCNN/toi_p3d/p3d_cls_{}_110.npy'.format(i))
    det1 = np.load('/data/wjc/TCNN_STCNN/toi_p3d/p3d_cls_{}_flowi_4.npy'.format(i))
    [flag, a] = r.forward(det,det1)
    b.append(a)
    i += 1
  with open('/data/wjc/TCNN_STCNN/p3d_rgbflow_114.pkl', 'wb') as fid:
    cPickle.dump(b, fid, cPickle.HIGHEST_PROTOCOL)
  '''
  flag = False
  re = []
  while not(flag):
    [curr, flag] = r.forward()
    re.append(curr)
  with open('/home/rhou/cpb_detections.pkl', 'wb') as fid:
    cPickle.dump(re, fid, cPickle.HIGHEST_PROTOCOL)
  '''
