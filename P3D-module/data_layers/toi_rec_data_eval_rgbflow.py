'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/data/wjc/TCNN_STCNN/caffe/python')
#sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
from dataset.jhmdb_flow import jhmdb_flow
from dataset.jhmdb import jhmdb
import numpy as np
from utils.cython_bbox import bbox_overlaps

class RecDataLayer():
  def __init__(self, net, model):
    self._batch_size = 1
    self._depth = 8
    self._height = 300
    self._width = 400
    self.dataset = jhmdb_flow('val', [self._height, self._width], split=1)

    self.dataset_rgb = jhmdb('val', [self._height, self._width], split=1)
    self.anchors, self.valid_idx, self._anchor_dims = self.dataset_rgb.get_anchors()

    caffe.set_mode_gpu()
    self._net = caffe.Net(net, model, caffe.TEST)

  def forward(self):
    self._net.blobs['data'].reshape(self._batch_size, 3,
                                    self._depth, self._height, self._width)
    self._net.blobs['tois'].reshape(self._batch_size * 1752, 5)

    [clip, gt_bboxes, labels, _, is_last] = self.dataset.next_val_video()

    n = int(clip.shape[0] )

    result = np.empty((n-self._depth+1, 1752, 44))
    for i in xrange(n - self._depth + 1):
      batch_clip = clip[i : i + 8].transpose([3, 0, 1, 2])
      batch_clip = np.expand_dims(batch_clip, axis=0)

#      pred = all_pred[i : i + 1* self._depth]
#      pred_anchors = np.reshape(pred, (-1,4)) * 1.25/ 16
 
      #print ((self.anchors).shape)
      batch_tois = np.hstack((np.zeros((1752, 1)), self.anchors))

      self._net.blobs['data'].data[...] = batch_clip.astype(np.float32,
                                                            copy=False)
      self._net.blobs['tois'].data[...] = batch_tois.astype(np.float32,
                                                              copy=False)
      self._net.forward()
      r1 = self._net.blobs['loss'].data[...]
      r2 = self._net.blobs['fc8-1'].data[...]

      result[i] = np.hstack((r1, r2))

    return result, is_last
