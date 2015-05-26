import numpy as np
import re
from os import listdir
from os.path import isfile, join
from skimage.filters import threshold_otsu, threshold_adaptive
import Image

extention_filter = re.compile(".*(.png|.jpg)")

def sign(x):
  return 1 if x > 0 else 0

def transform_to_neuro_form(image):
  return np.array([-1 if el else 1 for el in np.nditer(image)])

def get_bin_image_otsu(images, new_size):
  bin_images = []
  for image in images:
    black_white_im = np.array(image.convert("L"))
    otsu_thresh = threshold_otsu(black_white_im)
    otsu_im = black_white_im > otsu_thresh
    croped_im = crop(otsu_im, image.size)
    for_resize = Image.fromarray(croped_im.astype("uint8") * 255)
    resized_im = for_resize.resize(new_size)
    bin_images.append(np.array([False if not el else True for el in resized_im.getdata()]))
  return bin_images

def crop(image, size):
  min_x, min_y = size
  max_x, max_y = (0, 0)
  for idx_y, y_el in enumerate(image):
    for idx_x, x_el in enumerate(y_el):
      if not x_el:
        min_y = min(min_y, idx_y)
        min_x = min(min_x, idx_x)
        max_y = max(max_y, idx_y)
        max_x = max(max_x, idx_x)
  new_image = image[min_y:max_y, min_x:max_x]
  return new_image

def load_images(path, sect, images):
  for file_name in listdir(path):
    path_to_file = join(path, file_name)
    if isfile(path_to_file):
      if extention_filter.search(path_to_file):
        if not images.has_key(sect):
          images[sect] = []
        images[sect].append(Image.open(path_to_file))
    else:
      load_images(path_to_file, file_name.decode("utf-8"), images)

def binarisation_otsu():
  pass

def binarisation_niblack():
  pass