import neuro_tools
import numpy as np
import sys

class KernelHopfNeuron:

	mult_tr = np.array([])
	mem_images = np.array([])
	im_size = 0
  	im_size_sq = 0
  	future_im_size_sqq = 0