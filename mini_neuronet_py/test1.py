#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import neuronet
import neuro_tools
import Image
from PIL import ImageFont
from PIL import ImageDraw
import argparse
import pdb
import yaml
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.transform import rotate
from skimage import data
import numpy as np
import time
import math
#import multiprocessing as mp
import pathos.multiprocessing as mp

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Rectangle
from kivy.uix.textinput import TextInput
from kivy.graphics.texture import Texture
from kivy.config import Config
from kivy.core.window import Window

import subprocess
from threading import Thread

from pymongo import MongoClient
from bson.binary import Binary
import dill
from bson.objectid import ObjectId
from bson.errors import InvalidId

import hashlib
import os
os.system("taskset -p 0xff %d" % os.getpid())

IM_SIZE = 32

class NeuroLayout(FloatLayout):

  def __init__(self, **kwargs):
    super(NeuroLayout, self).__init__(**kwargs)

  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', dest='config', help='Configuration YAML')
  args = parser.parse_args()

  conf = open(args.config, 'r')
  tempConf = yaml.load_all(conf)

  images = bin_images = dict()
  net = None

  for line in tempConf:
    mem_directory = line["MemDirectory"]
    recognize_image_path = line["RecognizeImage"]
    instance_hash = line["InstanceHash"]
    load_mem = line["LoadMemory"]

  #pdb.set_trace()

  def learn(self, path = mem_directory):
    print "Re-education in progress..."

    t1 = time.time()
    neuro_tools.load_images(path, "", self.images)

    formated_bin_images = dict((sect,
      neuro_tools.format_bin_image(neuro_tools.get_bin_symb_otsu(self.images[sect]),
        (IM_SIZE, IM_SIZE)))
    for sect in self.images.keys())

    self.net = neuronet.HopfNet(zip(self.images.keys(), [IM_SIZE] * len(self.images)))
    for sect, images in formated_bin_images.iteritems():
      for image in images:
        self.net.learn(neuro_tools.transform_to_neuro_form(image), sect)

    print time.time() - t1
    print "Re-education has been finished successfully!"

  def recognize(self, path = recognize_image_path):
    pool = mp.Pool()
    test_im = Image.open(path)
    draw = ImageDraw.Draw(test_im)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", test_im.size[0] / 25)
    #test_im_formated = neuro_tools.transform_to_neuro_form(
    #  neuro_tools.format_bin_image(neuro_tools.get_bin_symb_otsu([test_im]),
    #    (IM_SIZE, IM_SIZE)))
    def calc():
      min_el = min(self.net.recognize(
        neuro_tools.transform_to_neuro_form(
          neuro_tools.format_bin_image([symb[0]], (IM_SIZE, IM_SIZE)))),
            key = lambda x: x[1])
      return min_el
     
    t1 = time.time()
    min_el = ("a", 1)
    for symb in neuro_tools.get_symbols(test_im):
      #img = Image.fromarray(im[0].astype("uint8") * 255)
      #img.show()
      min_el = calc()
      #pool.apply_async(calc).get()
      a, b = symb[1]
      draw.text((b, a), u"{0}".format(min_el[0]), (255, 0, 0), font = font)
    pool.close()
    pool.join()
    #sorted_res = sorted(self.net.recognize(test_im_formated), key = lambda x: x[0])
    #for name, res in sorted_res:
    #  print u"{0}: {1}".format(name, res)
    #min_el = min(sorted_res, key = lambda x: x[1])
    #print u"\n{0}: {1}".format(min_el[0], min_el[1])
    test_im.show()
    t2 = time.time()
    print t2 - t1

  def save_to_db(self,
    coll_name  = "test_collection",
    db_name    = "neuronet",
    ip_address = "localhost",
    port       = 27017):

    print "Saving..."

    client = MongoClient(ip_address, port)
    db = client[db_name]
    db_collection = db[coll_name]
    date_field = 'bin-data from {0}'.format(time.strftime("%d-%m-%Y %H:%M:%S"))
    thebytes = dill.dumps(neuro_tools.pack_net_instance(self.net))
    hash_object = hashlib.md5(thebytes + date_field)
    self.instance_hash = hash_object.hexdigest()
    chunks = neuro_tools.split_into_chunks(thebytes)

    for idx, chunk in enumerate(chunks):
      db_collection.insert(
        {'bin-data': Binary(chunk), 'date': date_field,
         'chunk': idx, 'hash': self.instance_hash})

    print "Successfully saved!\nDate: {0}, Hash: {1}" \
      .format(date_field, self.instance_hash)

  def load_from_db(self,
    instance_hash = instance_hash,
    coll_name     = "test_collection",
    db_name       = "neuronet",
    ip_address    = "localhost",
    port          = 27017):

    client = MongoClient(ip_address, port)
    db = client[db_name]
    db_collection = db[coll_name]

    if not instance_hash:
      print "Can't load, you have to specify Hash of saved instance"
      return None

    print "Loading..."

    thebytes = ""
    found = db_collection.find({"hash": instance_hash})
    found = sorted(found, key = lambda x: x["chunk"])
    for el in found:
      thebytes += el["bin-data"]

    self.net = neuro_tools.unpack_net_instance(dill.loads(thebytes))

    print "Successfully loaded!\nDate: {0}, Hash: {1}"\
      .format(found[0]['date'], found[0]['hash'])

  def clear_collection(self,
    coll_name     = "test_collection",
    db_name       = "neuronet",
    ip_address    = "localhost",
    port          = 27017):

    client = MongoClient(ip_address, port)
    db = client[db_name]
    db_collection = db[coll_name]

    print db_collection.remove({})

    print "Collection has been cleared successfully!"

  def hulk_smash(self):
    print Window.size

class NeuronetApp(App):
  icon = 'attracto.png'
  title = 'Attracto'

  def build(self):
    return NeuroLayout()


def main():
  #pdb.set_trace()
  time_0 = time.time()
  pool = mp.Pool(4)
  conf = open(args.config, 'r')
  tempConf = yaml.load_all(conf)
  images = dict()

  for line in tempConf:
    mem_directory = line["MemDirectory"]
    load_mem = line["LoadMemory"]

  neuro_tools.load_images(mem_directory, "", images)

  bin_images = dict((sect, neuro_tools.get_bin_symb_otsu(images[sect], (IM_SIZE, IM_SIZE)))
    for sect in images.keys())

  net = neuronet.HopfNet(zip(images.keys(), [IM_SIZE] * len(images)))
  for sect, images in bin_images.iteritems():
    for image in images:
      net.learn(neuro_tools.transform_to_neuro_form(image), sect)
      #net.learn(pool.apply_async(
      #  neuro_tools.transform_to_neuro_form, (image,)).get(), sect)

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
  time_1 = time.time()
  print time_1 - time_0
  print net.recognize(test_im_formated)
  time_2 = time.time()
  print time_2 - time_1

if __name__ == '__main__':
  threads = []
  threads.append(Thread(target=NeuronetApp().run()).start())
  for thread in threads:
    thread.join()