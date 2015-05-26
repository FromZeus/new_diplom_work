#!/usr/bin/env python
# -*- coding: utf-8 -*-

import neuronet
import neuro_tools
import Image
import argparse
import pdb
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', dest='config', help='Configuration YAML')
args = parser.parse_args()

def main():
  pdb.set_trace()
  conf = open(args.config, 'r')
  tempConf = yaml.load_all(conf)
  images = dict()

  for line in tempConf:
    mem_directory = line["MemDirectory"]

  neuro_tools.load_images(mem_directory, "", images)
  new_image = neuro_tools.move_symbol_bin(images.values()[0], 0, 0)
  new_image.show()
  #print images["а".decode("utf-8")]
  #print images.values()

  net1 = neuronet.HopfNet(zip(images.keys(), [el.size[0] for el in images.values()]))

if __name__ == '__main__':
  main()