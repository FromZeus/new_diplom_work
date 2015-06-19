import neuro_tools
import numpy as np
from numpy import copy
from numpy import sum as np_sum
import sys
import pathos.multiprocessing as mp
import dummy
cimport numpy as cnp

cdef class HopfNeuron:

  mem = np.array([], dtype = "float64")
  im_size = 0
  im_size_sq = 0
  img_in_memory = 0

  def __init__(self, int im_size, int img_in_memory = 0, cnp.ndarray ext_mem = cnp.ndarray([])):
    self.im_size = im_size
    self.im_size_sq = self.im_size ** 2
    self.img_in_memory = img_in_memory
    self.mem = copy(ext_mem)
    if not self.mem:
      self.mem = cnp.array([cnp.zeros(self.im_size_sq).astype("float64")] \
        * self.im_size_sq)

  def learn(self, cnp.ndarray image):
    cdef cnp.ndarray mem = self.mem
    cdef int im_size_sq = self.im_size_sq
    cdef cnp.ndarray local_image = image
    cdef int idx1
    for idx1 in xrange(self.im_size_sq):
      mem[idx1] = cnp.add(mem[idx1], local_image[idx1] / float(im_size_sq) * local_image)
      mem[idx1, idx1] = 0.0
    self.img_in_memory += 1

  cdef double recognize(self, cnp.ndarray image):
    cdef cnp.ndarray mem = self.mem
    cdef int converg = 0
    cdef cnp.ndarray result_img = copy(image) #np.array(dummy.shared_array) #copy(image)
    cdef cnp.ndarray pred_img
    cdef int idx
    cdef int idx1
    cdef int col
    cdef int assoc
    for idx in xrange(8):
      pred_img = copy(result_img)
      col = 0
      for idx1 in xrange(self.im_size_sq):
        assoc = np_sum(pred_img * mem[idx1])
        result_img[idx1] = 1 if neuro_tools.sign(assoc) else -1
        if pred_img[idx1] == result_img[idx1]:
          col += 1
        converg += abs(pred_img[idx1] - result_img[idx1])
      if col == self.im_size_sq:
        return converg / (self.img_in_memory ** .5)
    return sys.float_info.max