import numpy as np
import re
from os import listdir
from os.path import isfile, join
from skimage.filters import threshold_otsu, threshold_adaptive
import Image
import neuronet
import dill

extention_filter = re.compile(".*(.png|.jpg)")

def sign(x):
  return 1 if x >= 0 else 0

def transform_to_neuro_form(image):
  return np.array([-1 if el else 1 for el in np.nditer(image)])

def get_bin_symb_otsu(images, new_size):
  bin_images = []
  for image in images:
    black_white_im = np.array(image.convert("L"))
    otsu_thresh = threshold_otsu(black_white_im)
    otsu_im = black_white_im > otsu_thresh
    croped_im = crop_alpha(otsu_im, image.size)
    for_resize = Image.fromarray(croped_im.astype("uint8") * 255)
    resized_im = for_resize.resize(new_size)
    bin_images.append(np.array([False if not el else True for el in resized_im.getdata()]). \
      reshape(new_size))
  return bin_images

def crop_alpha(bin_image, size):
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

def stretch_image(bin_image, shift, center = None):
  if not center:
    center = (bin_image.size[0] / 2, bin_image[1] / 2)
  relative_shift = (shift[0] * bin_image.size[0] / 10,
    shift[1] * bin_image.size[1] / 10)

  def shift_pix_lr(lr, part):
    black = np.arange(relative_shift)
    black.fill(False)
    white = np.arange(relative_shift)
    white.fill(True)

    l = 1 if lr == "l" else 0
    r = 1 if lr == "r" else 0

    for idx_y, y_el in enumerate(part):
      for idx_x, x_el in enumerate(y_el):
        if not x_el:
          if idx_x > 0 and idx_x < len(y_el) - 1:
            if y_el[idx_x - 1]:
              part[idx_y, max(0, idx_x - relative_shift * l): \
                min(len(idx_y) - 1, idx_x + relative_shift * r)] = \
                  black if lrud == "l" else white
            if y_el[idx_x + 1]:
              part[idx_y, max(0, idx_x - relative_shift * l + 1): \
                min(len(idx_y) - 1, idx_x + relative_shift * r + 1)] = \
                  white if lrud == "l" else black

  stretched_image = np.array(bin_image)

  if relative_shift[1]:
    left_part = stretched_image[:,:center[1]]
    right_part = stretched_image[:,center[1]:]

    left_part = shift_pix_lr("l", left_part)
    right_part = shift_pix_lr("r", right_part)

  stretched_image = np.append(left_part.T, right_part.T, axis = 0).T

  if relative_shift[0]:
    up_part = bin_image[:center[0],:]
    down_part = bin_image[center[0]:,:]

  stretched_image = np.append(up_part, down_part, axis = 0)

  return stretched_image

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