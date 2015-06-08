import neuro_tools
import numpy as np
import sys

class HopfNeuron:

  mem = np.array([])
  im_size = 0
  im_size_sq = 0
  img_in_memory = 0

  def __init__(self, im_size, img_in_memory = 0, ext_mem = []):
    self.im_size = im_size
    self.im_size_sq = self.im_size ** 2
    self.img_in_memory = img_in_memory
    self.mem = np.copy(ext_mem)
    if not self.mem:
      self.mem = np.array([[0.0] * self.im_size_sq for el in xrange(self.im_size_sq)])

  def learn(self, image):
    for idx1 in xrange(self.im_size_sq):
      self.mem[idx1] = np.add(self.mem[idx1], image[idx1] / float(self.im_size_sq) * image)
      self.mem[idx1, idx1] = 0.0
    self.img_in_memory += 1

  def recognize(self, image):
    converg = 0
    result_img = np.copy(image)
    for idx in xrange(10):
      pred_img = np.copy(result_img)
      col = 0
      for idx1 in xrange(self.im_size_sq):
        assoc = np.sum(pred_img * self.mem[idx1])
        result_img[idx1] = 1 if neuro_tools.sign(assoc) else -1
        if pred_img[idx1] == result_img[idx1]:
          col += 1
        converg += abs(pred_img[idx1] - result_img[idx1])
      if col == self.im_size_sq:
        return converg / (self.img_in_memory ** .5)
    return sys.float_info.max