#!/usr/bin/env python
# -*- coding: utf-8 -*-

import Image
import neuro_tools
from skimage.transform import rotate
from skimage.morphology import dilation
from skimage.morphology import erosion
from skimage.morphology import square
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

def get_rotated(image, begin_ang, end_ang, step):
  rotated_images = []
  for ang in xrange(begin_ang, end_ang, step):
    rotated_images.append(image.convert("L").rotate(ang, Image.BICUBIC))
  return rotated_images

def get_distorted(image, params, orient = "horizont"):
  shifts = []
  np_image = np.array(image.convert("L"))
  for el in params:
    if el[0] == "sin":
      shifts.append(lambda x: np_image.shape[0] / el[1] * \
        np.sin(x * el[2] / np_image.shape[1]))
    if el[0] == "cos":
      shifts.append(lambda x: np_image.shape[0] / el[1] * \
        np.cos(x * el[2] / np_image.shape[1]))
    if el[0] == "triang":
      lambda x: np_image.shape[0] / el[1] * \
        (x / el[2] / np_image.shape[1] - math.floor(x / (el[2] / np_image.shape[1])))
    if el[0] == "erosion":
      np_image = erosion(np_image, square(el[1]))
    if el[0] == "dilation":
      np_image = dilation(np_image, square(el[1]))

  if orient == "horizont":
    for idx in xrange(np_image.shape[0]):
      for shift in shifts:
        np_image[idx,:] = np.roll(np_image[idx,:], int(shift(idx)))
  if orient == "vert":
    for idx in xrange(np_image.shape[1]):
      for shift in shifts:
        np_image[:, idx] = np.roll(np_image[:, idx], int(shift(idx)))

  return Image.fromarray(np_image)

def fill_with_bfs(pixels, old_bri, new_bri, size_y, size_x, y, x):
  used = [[False] * size_x for i in xrange(size_y)]
  qu = deque([(y, x)])

  while qu:
    el = qu.popleft()
    for _y in xrange(-1, 2):
      for _x in xrange(-1, 2):
        new_y = el[0] + _y
        new_x = el[1] + _x
        if new_y < size_y and new_y >= 0 and new_x < size_x and new_x >= 0:
          if not used[new_y][new_x] and pixels[new_y, new_x] == old_bri:
            used[new_y][new_x] = True
            pixels[new_y, new_x] = new_bri
            qu.append((new_y, new_x))

def fill_edges(image, old_bri, new_bri):
  pixels = image.load()
  fill_with_bfs(pixels, 0, 255,
    image.size[0], image.size[1], 0, 0)
  fill_with_bfs(pixels, 0, 255,
    image.size[0], image.size[1], 0, image.size[1] - 1)
  fill_with_bfs(pixels, 0, 255,
    image.size[0], image.size[1], image.size[0] - 1, 0)
  fill_with_bfs(pixels, 0, 255,
    image.size[0], image.size[1], image.size[0] - 1, image.size[1] - 1)
  return image

def main():
  #pdb.set_trace()

  conf = open(args.config, 'r')
  tempConf = yaml.load_all(conf)
  images = dict()

  for line in tempConf:
    img_directory = line["ImgDirectory"]

  neuro_tools.load_images(img_directory, "", images)

  for name, sect in images.iteritems():
    for el in list(sect):
      #images[name].append(get_distorted(el, [("sin", 25.0, 20.0), ("triang", 35.0, 50.0)]))
      images[name].append(get_distorted(el, [("cos", 25.0, 20.0)], "vert"))
      images[name].append(get_distorted(el, [("dilation", 2), ("sin", 25.0, 20.0)]))
      images[name].append(get_distorted(el, [("erosion", 1), ("sin", 25.0, 20.0)]))

  for name, sect in images.iteritems():
    for idx1, el in enumerate(sect):
      for idx2, rotated_image in enumerate(get_rotated(el, -60, 60, 30)):
        filled_rotated_image = fill_edges(rotated_image, 0, 255)
        if filled_rotated_image.mode != "RGBA":
          filled_rotated_image = filled_rotated_image.convert("RGBA")
        if not os.path.exists("mem_0.3/{0}".format(name.encode("utf-8"))):
          os.makedirs("mem_0.3/{0}".format(name.encode("utf-8")))
        filled_rotated_image.save("mem_0.3/{3}/{0}_{1}_{2}.png".format(
          name.encode("utf-8"), idx1, idx2, name.encode("utf-8")))

  #test_img = Image.open("test/1_з.png".decode("utf-8"))
  #test_img = fill_edges(test_img.convert("L"), 0, 255)
  #test_img.show()

if __name__ == '__main__':
  main()