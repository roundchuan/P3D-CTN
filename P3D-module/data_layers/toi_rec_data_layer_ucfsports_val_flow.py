'''
The Caffe data layer for training label classifier.
This layer will parse pixel values and actionness labels to the network.
'''
import sys
sys.path.insert(0, '/data/wjc/TCNN_STCNN/caffe/python')
#sys.path.insert(0, '/home/rhou/caffe/python')
import caffe
#from dataset.jhmdb import jhmdb
from dataset.ucfsports_flo import ucfsports_flo
import numpy as np
from utils.cython_bbox import bbox_overlaps

class RecDataLayer(caffe.Layer):
  def setup(self, bottom, top):
    self._batch_size = 1
    self._depth = 8
    self._height = 300
    self._width = 400
    #self.dataset = jhmdb('train', [self._height, self._width],
    #                         '/home/rhou/JHMDB')
    self.dataset = ucfsports_flo('val', [self._height, self._width], split=1)

    self.anchors, self.valid_idx, self._anchor_dims = self.dataset.get_anchors()

  def reshape(self, bottom, top):
    # Clip data.
    top[0].reshape(self._batch_size, 2, self._depth, self._height, self._width)
    # Ground truth labels.
    top[1].reshape(self._batch_size * 64, 1)
    # Ground truth tois.
    top[2].reshape(self._batch_size * 64, 5)

  def forward(self, bottom, top):
    [clips, labels, tmp_bboxes, _, _, _] \
      = self.dataset.next_batch(self._batch_size, self._depth)
    batch_clip = clips.transpose((0, 4, 1, 2, 3))
    batch_tois = np.empty((0, 5))
    batch_labels = np.empty((0,1))

 #   i = 0
    for i in xrange(self._depth):
#      box = np.expand_dims(box, axis=0)
      box = tmp_bboxes[:, :, :]
      gt_bboxes = box[:,i] / 16
#      gt_bboxes = np.array(np.mean(box, axis=1)) / 16
   #   pred = tmp_pred[0,:,:,:]
#      pred_anchors = np.reshape(pred[i], (-1,4)) * 1.25/16

      overlaps = bbox_overlaps(
        np.ascontiguousarray(self.anchors, dtype=np.float),
        np.ascontiguousarray(gt_bboxes, dtype=np.float))
      max_overlaps = overlaps.max(axis=1)
      gt_argmax_overlaps = overlaps.argmax(axis=0)

      curr_labels = np.ones(self.anchors.shape[0]) * (-1)
      curr_labels[max_overlaps < 0.1] = 0
      curr_labels[max_overlaps >= 0.5] = labels[0]

      curr_labels[gt_argmax_overlaps] = labels[0]

      fg_inds = np.where(curr_labels > 0)[0]
      num_fg = len(fg_inds)
      if len(fg_inds) > 2:
        fg_inds = np.random.choice(fg_inds, size=(2))
        num_fg = 2

      bg_inds = np.where(curr_labels == 0)[0]
      if len(bg_inds) > 6:
        bg_inds = np.random.choice(bg_inds, size=(6))
        num_bg = 6
 #     num_bg = num_fg
 #     bg_inds = np.random.choice(bg_inds, size=(num_bg))
      inds = np.hstack((fg_inds, bg_inds))
 #     print inds.shape
      curr_bboxes = np.hstack((np.ones((len(inds), 1)) * i, self.anchors[inds]))
      batch_tois = np.concatenate((batch_tois, curr_bboxes), axis=0)
      curr_l = np.expand_dims(curr_labels[inds], axis=1)
      batch_labels = np.concatenate((batch_labels, curr_l), axis=0)
    #  batch_tois = np.vstack((batch_tois, curr_bboxes))
    #  batch_labels = np.hstack((batch_labels, curr_labels[inds]))
     # i += 1

#    print batch_tois.shape
    top[1].reshape(*batch_labels.shape)
    top[2].reshape(*batch_tois.shape)

    top[0].data[...] = batch_clip.astype(np.float32, copy=False)
    top[1].data[...] = batch_labels.astype(np.float32, copy=False)
    top[2].data[...] = batch_tois.astype(np.float32, copy=False)

  def backward(self, top, propagate_down, bottom):
    """This layer does not propagate gradients."""
    pass
