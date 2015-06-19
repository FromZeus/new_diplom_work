import neuron
import pathos.multiprocessing as mp
import multiprocessing as def_mp
import ctypes
import dummy
import cython
from cython.parallel import parallel, prange  

def test_init(arr):
  dummy.shared_array = arr

class HopfNet:

  neurons = dict()
  rec_objs = []

  def __init__(self, rec_objs, neurons = None):
    self.rec_objs = rec_objs
    for el in self.rec_objs:
      self.neurons[el[0]] = neuron.HopfNeuron(el[1])
    if neurons:
      self.neurons = neurons

  def learn(self, image, name):
    self.neurons[name].learn(image)

  @cython.boundscheck(False)
  @cython.wraparound(False)
  def recognize(self, image):
    results = []
    #mp_image = def_mp.Array(ctypes.c_int, image)
    #pool = mp.Pool(initializer = test_init, initargs = (mp_image, ))
    #with nogil, parallel():
    #  for i in prange(5, schedule='guided'):
    #    j = 0
    for name, neuron in self.neurons.iteritems():
      #results.append((name, pool.apply_async(neuron.recognize).get()))
      results.append((name, neuron.recognize(image)))
    return results