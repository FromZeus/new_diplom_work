import numpy as np
from numpy import array
import re
from os import listdir
from os.path import isfile, join
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.transform import rotate
from skimage.transform import rescale
from skimage.morphology import dilation
from skimage.morphology import erosion
from skimage.morphology import square
from skimage.measure import label
from collections import deque
from scipy.ndimage import find_objects
import Image
import neuronet
import dill
import math

extention_filter = re.compile(".*(.png|.jpg)")

def sign(x):
  return 1 if x >= 0 else 0

def transform_to_neuro_form(image):
  return array([-1 if el else 1 for el in np.nditer(image)])

def get_rotated(image, begin_ang, end_ang, step):
  rotated_images = []
  for ang in xrange(begin_ang, end_ang, step):
    rotated_images.append(image.convert("L").rotate(ang, Image.BICUBIC))
  return rotated_images

def get_distorted(image, params, orient = "horizont"):
  shifts = []
  np_image = array(image.convert("L"))
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

def dilatation_cross_numb(image, numb):
  block_size = 50
  binary_adaptive = dilation(threshold_adaptive(array(image.convert("L")),
    block_size, offset=10), square(numb))
  return get_cross_numb(binary_adaptive)

def get_cross_numb(bin_image):
  col = np.zeros(bin_image.shape[0]).astype("uint32")
  for idx_y, y_el in enumerate(bin_image):
    pred = y_el[0]
    for x_el in y_el:
      if x_el != pred:
        pred = x_el
        col[idx_y] += 1
  return np.average(col)

def bin_search(func, f_args, range_val, val, inc_dec):
  new_range_val = range_val
  for i in xrange(4):
    midle = (new_range_val[1] - new_range_val[0]) / 2
    new_f_args = f_args + [midle]
    res = func(*new_f_args)
    if inc_dec == "inc":
      if res > val:
        new_range_val = (new_range_val[0], midle)
      elif res < val:
        new_range_val = (midle, new_range_val[1])
    elif inc_dec == "dec":
      if res > val:
        new_range_val = (midle, new_range_val[1])
      elif res < val:
        new_range_val = (new_range_val[0], midle)

  if (new_range_val[1] - new_range_val[0]) % 2 == 0:
    midle -= 1

  return midle

def get_symbols(image):
  dil_eros = bin_search(dilatation_cross_numb, [image], (1, 16), 1.0, "dec")
  block_size = 50
  binary_adaptive_image = erosion(dilation(threshold_adaptive(
    array(image.convert("L")), block_size, offset=10),
      square(dil_eros)), square(dil_eros))

  all_labels = label(binary_adaptive_image, background = True)
  objects = find_objects(all_labels)

  av_width = av_height = 0
  symbols = []

  for obj in objects:
    symb = (binary_adaptive_image[obj], (obj[0].start, obj[1].start))
    symbols.append(symb)
    av_height += symb[0].shape[0]
    av_width += symb[0].shape[1]

  av_width /= float(len(objects))
  av_height /= float(len(objects))

  symbols = [symb for symb in symbols
    if symb[0].shape[0] >= av_height and symb[0].shape[1] >= av_width]

  return symbols

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

def add_edges(image, new_size):
  if new_size[0] < image.shape[0] or new_size[1] < image.shape[1]:
    print "Error: New size have to be bigger past"
    return image

  y_indent = (new_size[0] - image.shape[0]) / 2
  x_indent = (new_size[1] - image.shape[1]) / 2

  np_new_image = array([np.arange(new_size[0])] * new_size[1])
  np_new_image.fill(255)
  np_new_image[y_indent:y_indent + image.shape[0],
    x_indent:x_indent + image.shape[1]] = image

  return np_new_image

def format_bin_image(images, new_size):
  fromarray = Image.fromarray

  def ft_translation(x):
    return True if x else False
  v_ft_translation = np.vectorize(ft_translation)

  resized_images = []
  for image in images:
    for_resize = fromarray(image.astype("uint8") * 255)
    resized_im = for_resize.resize(new_size)
    resized_images.append(v_ft_translation(resized_im.getdata()).reshape(new_size))
  return resized_images

def get_bin_symb_otsu(images):

  bin_images = []
  for image in images:
    black_white_im = array(image.convert("L"))
    otsu_thresh = threshold_otsu(black_white_im)
    otsu_im = black_white_im > otsu_thresh
    croped_im = bin_crop_alpha(otsu_im, image.size)
    bin_images.append(croped_im)

  return bin_images

def bin_crop_alpha(bin_image, size):
  min_x, min_y = size
  max_x, max_y = (0, 0)
  for idx_y, y_el in enumerate(bin_image):
    for idx_x, x_el in enumerate(y_el):
      if not x_el:
        min_y = min(min_y, idx_y)
        min_x = min(min_x, idx_x)
        max_y = max(max_y, idx_y)
        max_x = max(max_x, idx_x)
  new_image = bin_image[min_y:max_y, min_x:max_x]
  return new_image

def load_images(path, sect, images):
  for file_name in listdir(path):
    path_to_file = join(path, file_name)
    if isfile(path_to_file):
      if extention_filter.search(path_to_file):
        if not images.has_key(sect):
          images[sect] = []
        tmp_image = Image.open(path_to_file)
        images[sect].append(tmp_image.copy())
        tmp_image.close()
    else:
      load_images(path_to_file, file_name.decode("utf-8"), images)

def pack_net_instance(instance):
  bytes_dict = {}
  bytes_dict["rec_objs"] = dill.dumps(instance.rec_objs)
  for alpha, net in instance.neurons.iteritems():
    bytes_dict[alpha] = dill.dumps(net)
  return bytes_dict

def unpack_net_instance(bytes_dict):
  net = neuronet.HopfNet(dill.loads(bytes_dict["rec_objs"]), 
    dict((alpha, dill.loads(bytes_dict[alpha]))
      for alpha in bytes_dict.keys() if alpha != "rec_objs"))
  return net

def split_into_chunks(data, chunk_size = 16777000):
  chunks = []
  for idx in xrange(0, len(data) + 1, chunk_size):
    chunks.append(data[idx : idx + chunk_size])
  return chunks

def merge_chunks(chunks):
  data = ""
  for chunk in chunks:
    data += chunk
  return data

def binarisation_otsu():
  pass

def binarisation_niblack():
  pass