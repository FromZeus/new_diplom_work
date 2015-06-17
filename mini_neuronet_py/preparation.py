#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Image
import neuro_tools
from skimage.transform import rotate
from skimage.transform import rescale
from skimage import data
from collections import deque
import numpy as np
import argparse
import yaml
import os
import math
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config', help='Configuration YAML')
args = parser.parse_args()

def main():
  #pdb.set_trace()

  conf = open(args.config, 'r')
  tempConf = yaml.load_all(conf)
  images = dict()

  for line in tempConf:
    img_directory = line["ImgDirectory"]
    mem_directory = line["MemDirectory"]

  neuro_tools.load_images(img_directory, "", images)

  def rescaled_form(image, scale, norm_size):
    rescaled_image = rescale(np.array(el.convert("L")), scale)
    rescaled_image = rescaled_image * 255
    mx = max(rescaled_image.shape[0], rescaled_image.shape[1])
    buf_size = (mx, mx)
    image_with_edges = neuro_tools.add_edges(
      rescaled_image, buf_size)
    return Image.fromarray(image_with_edges.astype("uint8")).resize(norm_size)

  for name, sect in images.iteritems():
    for el in list(sect):
      #images[name].append(get_distorted(el, [("sin", 25.0, 20.0), ("triang", 35.0, 50.0)]))
      images[name].append(neuro_tools.get_distorted(el,
        [("cos", 25.0, 20.0)], "vert"))
      images[name].append(neuro_tools.get_distorted(el,
        [("dilation", 2), ("sin", 25.0, 20.0)]))
      images[name].append(neuro_tools.get_distorted(el,
        [("erosion", 1), ("sin", 25.0, 20.0)]))
      images[name].append(neuro_tools.get_distorted(el,
        [("dilation", 2)]))
      images[name].append(neuro_tools.get_distorted(el,
        [("erosion", 1)]))
      rescaled1 = rescaled_form(el, (2.5, 1), (32, 32))
      rescaled2 = rescaled_form(el, (1, 2.5), (32, 32))
      images[name].append(rescaled1)
      images[name].append(rescaled2)

  for name, sect in images.iteritems():
    for idx1, el in enumerate(sect):
      for idx2, rotated_image in enumerate(neuro_tools.get_rotated(el.resize((32, 32)), 0, 30, 30)):
        filled_rotated_image = neuro_tools.fill_edges(rotated_image, 0, 255)
        if filled_rotated_image.mode != "RGBA":
          filled_rotated_image = filled_rotated_image.convert("RGBA")
        if not os.path.exists("{0}/{1}".format(
          mem_directory, name.encode("utf-8"))):
          os.makedirs("{0}/{1}".format(
            mem_directory, name.encode("utf-8")))
        filled_rotated_image.save("{0}/{4}/{1}_{2}_{3}.png".format(
          mem_directory, name.encode("utf-8"),
          idx1, idx2, name.encode("utf-8")))

  #test_img = Image.open("test/1_з.png".decode("utf-8"))
  #test_img = fill_edges(test_img.convert("L"), 0, 255)
  #test_img.show()

if __name__ == '__main__':
  main()