#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import listdir
from os.path import isfile, join
import Image
import pdb

def main():
  #pdb.set_trace()
  for el in listdir(sys.argv[1]):
    im = Image.open(join(sys.argv[1], el))
    im.thumbnail((32, 32))
    im.save(el + "fixed", "JPEG")

if __name__ == '__main__':
  main()