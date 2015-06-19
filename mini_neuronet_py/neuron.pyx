import neuro_tools
import numpy as np
from numpy import copy
from numpy import sum as np_sum
import sys
import pathos.multiprocessing as mp
import dummy
cimport numpy as cnp
from cython.parallel import parallel, prange
cimport cython

cdef class HopfNeuron:

  cdef public cnp.ndarray mem
  cdef public int im_size
  cdef public int im_size_sq
  cdef public img_in_memory

  def __cinit__(self, int im_size, int img_in_memory = 0, cnp.ndarray[cnp.float64_t, ndim=2] ext_mem = None):
    self.im_size = im_size
    self.im_size_sq = self.im_size ** 2
    self.img_in_memory = img_in_memory
    self.mem = copy(ext_mem)
    if not self.mem:
      self.mem = np.array([np.zeros(self.im_size_sq).astype("float64")] \
        * self.im_size_sq)

  def learn(self, cnp.ndarray[cnp.int64_t, ndim=1] image):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mem = self.mem
    cdef int im_size_sq = self.im_size_sq
    cdef cnp.ndarray local_image = image
    cdef int idx1
    for idx1 in xrange(self.im_size_sq):
      mem[idx1] = np.add(mem[idx1], local_image[idx1] / float(im_size_sq) * local_image)
      mem[idx1, idx1] = 0.0
    self.img_in_memory += 1

  @cython.boundscheck(False)
  @cython.wraparound(False)
  def recognize(self, cnp.ndarray[cnp.int64_t, ndim=1] image):
    return self.recognize_c(image)

  cdef recognize_c(self, cnp.ndarray[cnp.int64_t, ndim=1] image):
    cdef cnp.ndarray[cnp.float64_t, ndim=2] mem = self.mem
    cdef int converg = 0, idx, idx1, col, assoc, im_size_sq = self.im_size_sq
    cdef int img_in_memory = self.img_in_memory
    cdef cnp.ndarray[cnp.int64_t, ndim=1] result_img = copy(image), pred_img #np.array(dummy.shared_array) #copy(image)
    for idx in xrange(8):
      pred_img = copy(result_img)
      col = 0
      for idx1 in xrange(im_size_sq):
        assoc = np_sum(pred_img * mem[idx1])
        result_img[idx1] = 1 if neuro_tools.sign(assoc) else -1
        if pred_img[idx1] == result_img[idx1]:
          col += 1
        converg += abs(pred_img[idx1] - result_img[idx1])
      if col == im_size_sq:
        return converg / (img_in_memory ** .5)
    return sys.float_info.max