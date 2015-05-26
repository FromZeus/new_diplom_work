import numpy as np
import re
from os import listdir
from os.path import isfile, join
import Image

extention_filter = re.compile(".*(.png|.jpg)")

def sign(x):
  return 1 if x > 0 else 0

def move_symbol_bin(image, x, y):
  min_y, min_x = image.size
  for idx_y, y_el in enumerate(image):
    for idx_x, x_el in enumerate(y_el):
      if x_el:
        min_y = min(min_y, idx_y)
        min_x = min(min_x, idx_x)
  new_image = image[min_y:, min_x:]
  #new_image = Image.new('RGB', (48,48), "white")
  #new_image.paste(image.crop((min_x, min_y, image.size[0], image.size[1])), x, y)
  return new_image

def load_images(path, sect, images):
  for file_name in listdir(path):
    path_to_file = join(path, file_name)
    if isfile(path_to_file):
      if extention_filter.search(path_to_file):
        images[sect] = Image.open(path_to_file)
    else:
      load_images(path_to_file, file_name.decode("utf-8"), images)

def binarisation_otsu():
  pass

def binarisation_niblack():
  pass