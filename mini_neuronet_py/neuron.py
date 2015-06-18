import neuro_tools
from neuro_tools import sign
import numpy as np
from numpy import copy
from numpy import sum as np_sum
import sys
import pathos.multiprocessing as mp
import dummy

class HopfNeuron:

  mem = np.array([])
  im_size = 0
  im_size_sq = 0
  img_in_memory = 0

  def __init__(self, im_size, img_in_memory = 0, ext_mem = []):
    self.im_size = im_size
    self.im_size_sq = self.im_size ** 2
    self.img_in_memory = img_in_memory
    self.mem = copy(ext_mem)
    if not self.mem:
      self.mem = np.array([np.zeros(self.im_size_sq).astype("float64")] \
        * self.im_size_sq)

  def learn(self, image):
    mem = self.mem
    local_image = image
    for idx1 in xrange(self.im_size_sq):
      mem[idx1] = np.add(mem[idx1], local_image[idx1] / float(self.im_size_sq) * local_image)
      mem[idx1, idx1] = 0.0
    self.img_in_memory += 1

  def recognize(self, image, char, qu):
    mem = self.mem
    converg = 0
    result_img = copy(image) #np.array(dummy.shared_array) #copy(image)
    for idx in range(8):
      pred_img = copy(result_img)
      col = 0
      for idx1 in range(self.im_size_sq):
        assoc = np_sum(pred_img * mem[idx1])
        result_img[idx1] = 1 if sign(assoc) else -1
        if pred_img[idx1] == result_img[idx1]:
          col += 1
        converg += abs(pred_img[idx1] - result_img[idx1])
      if col == self.im_size_sq:
        qu.put((char, (converg / (self.img_in_memory ** .5))))
        #return converg / (self.img_in_memory ** .5)
    qu.put((char, sys.float_info.max))
    #return sys.float_info.max