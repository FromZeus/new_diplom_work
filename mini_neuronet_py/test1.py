#!/usr/bin/env python
# -*- coding: utf-8 -*-

import neuronet
import neuro_tools
import Image
import argparse
import pdb
import yaml
from skimage.filters import threshold_otsu, threshold_adaptive
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config', help='Configuration YAML')
args = parser.parse_args()

IM_SIZE = 48

def main():
  pdb.set_trace()
  conf = open(args.config, 'r')
  tempConf = yaml.load_all(conf)
  images = dict()

  for line in tempConf:
    mem_directory = line["MemDirectory"]

  neuro_tools.load_images(mem_directory, "", images)
  bin_images = dict((sect, neuro_tools.get_bin_image_otsu(images[sect], (IM_SIZE, IM_SIZE)))
    for sect in images.keys())

  net = neuronet.HopfNet(zip(images.keys(), [IM_SIZE] * len(images)))
  for sect, images in bin_images.iteritems():
    for image in images:
      net.learn(image, sect)

  neuro_tools.load_images(mem_directory, "", images)
  bw_image = np.array(images["а".decode("utf-8")][0].convert("L"))
  global_thresh = threshold_otsu(bw_image)
  binary_global = bw_image > global_thresh
  test = neuro_tools.crop(binary_global, images["а".decode("utf-8")][0].size)
  im = Image.fromarray(test.astype("uint8") * 255)
  new_image = im.resize((48, 48))
  new_image.show()
  new_image.save("test.jpg")

  #net1 = neuronet.HopfNet(zip(images.keys(), [el.size[0] for el in images.values()]))

if __name__ == '__main__':
  main()