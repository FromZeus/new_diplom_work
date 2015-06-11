#!/usr/bin/env python
# -*- coding: utf-8 -*-

import neuronet
import neuro_tools
import Image
import argparse
import pdb
import yaml
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.transform import rotate
from skimage import data
import numpy as np
import time
import math
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config', help='Configuration YAML')
args = parser.parse_args()

IM_SIZE = 32

def main():
  #pdb.set_trace()
  pool = Pool()
  conf = open(args.config, 'r')
  tempConf = yaml.load_all(conf)
  images = dict()

  for line in tempConf:
    mem_directory = line["MemDirectory"]

  neuro_tools.load_images(mem_directory, "", images)

  bin_images = dict((sect, neuro_tools.get_bin_symb_otsu(images[sect], (IM_SIZE, IM_SIZE)))
    for sect in images.keys())

  net = neuronet.HopfNet(zip(images.keys(), [IM_SIZE] * len(images)))
  for sect, images in bin_images.iteritems():
    for image in images:
      net.learn(neuro_tools.transform_to_neuro_form(image), sect)

  #print net.recognize(neuro_tools.transform_to_neuro_form(bin_images[u"я"][0]))
  #pad_test_image = Image.open("/home/s-quark/Desktop/new_diplom_work/" \
  #  "mini_neuronet_py/test_images/test1.jpg").convert("L")

  #pad_test_image = np.array(pad_test_image)

  #from scipy.misc import lena
  #img = lena()

  #A = pad_test_image.shape[0] / 40.0
  #w = 60.0 / pad_test_image.shape[1]
  #shift1 = lambda x: A * np.sin(x * w)

  #div1 = 50.0
  #shift2 = lambda x: 10.0 * (x / div1 - math.floor(x / div1))

  #shift3 = lambda x: A * np.cos(x * w)

  #for idx in xrange(pad_test_image.shape[0]):
  #  pad_test_image[idx,:] = np.roll(pad_test_image[idx,:], int(shift3(idx)))
  #  pad_test_image[idx,:] = np.roll(pad_test_image[idx,:], int(shift2(idx)))
  #pad_res = np.pad(pad_test_image, (2, 2), mode='constant')
  #Image.fromarray(pad_test_image).show()
  test_im = Image.open("/home/s-quark/Desktop/new_diplom_work/" \
    "mini_neuronet_py/test_images/test3.jpg")
  test_im_formated = neuro_tools.transform_to_neuro_form(
    neuro_tools.get_bin_symb_otsu([test_im],(IM_SIZE, IM_SIZE)))
  start = time.time()
  print net.recognize(test_im_formated)
  end = time.time()
  print end - start

if __name__ == '__main__':
  main()