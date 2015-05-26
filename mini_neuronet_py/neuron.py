import neuro_tools
import numpy as np
import sys

class HopfNeuron:

  mem = []
  im_size = 0
  img_in_memory = 0

  def __init__(self, im_size, img_in_memory = 0, ext_mem = []):
    self.im_size = im_size
    self.img_in_memory = img_in_memory
    self.mem = list(ext_mem)
    if not self.mem:
      mem = [[0] * self.im_size for el in xrange(self.im_size)]

  def learn(self, img):
    for idx1 in xrange(self.im_size):
      assoc = 0.0
      for idx2 in xrange(idx1):
        assoc += img[idx1] * img[idx2]
        self.mem[idx2][idx1] = self.mem[idx1][idx2] = assoc / im_size
    self.img_in_memory += 1

  def recognize(self, img):
    converg = 0
    result_img = np.copy(img)
    for idx in xrange(10):
      pred_img = result_img
      col = 0
      for idx1 in xrange(self.im_size):
        assoc = 0
        for idx2 in xrange(self.im_size):
          assoc += result_img[idx2] * self.mem[idx1][idx2]
        result_img[idx1] = 1 if neuro_tools.sign(assoc) >= 0 else -1
        if pred_img[dix1] == result_img[dix1]:
          col += 1
        converg += abs(pred_img[idx1] - result_img[idx1])
      if col == im_size:
        return converg / (self.img_in_memory ** .5)
    return sys.maxfloat