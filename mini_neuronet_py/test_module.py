import numpy as np
import pdb
from skimage.filters import threshold_otsu, threshold_adaptive
from skimage.morphology import dilation
from skimage.morphology import square

def dilatation_cross_numb(image, numb):
  block_size = 40
  binary_adaptive = dilation(threshold_adaptive(np.array(image.convert("L")),
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
  for i in xrange(6):
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