import neuron
import pathos.multiprocessing as mp
import multiprocessing as def_mp
import ctypes
import dummy
import cython
from cython.parallel import parallel, prange
cimport numpy as cnp  

def test_init(arr):
  dummy.shared_array = arr

cdef class HopfNet:

  cdef public dict neurons
  cdef public rec_objs

  def __cinit__(self, rec_objs, neurons = None):
    self.neurons = dict()
    self.rec_objs = rec_objs
    for el in self.rec_objs:
      self.neurons[el[0]] = neuron.HopfNeuron(el[1])
    if neurons:
      self.neurons = neurons

  def learn(self, cnp.ndarray[cnp.int64_t, ndim=1] image, Py_UCS4 name):
    self.neurons[name].learn(image)

  @cython.boundscheck(False)
  @cython.wraparound(False)
  def recognize(self, cnp.ndarray[cnp.int64_t, ndim=1] image):
    results = []
    #mp_image = def_mp.Array(ctypes.c_int, image)
    #pool = mp.Pool(initializer = test_init, initargs = (mp_image, ))
    pool = mp.Pool()
    #with nogil, parallel():
    #  for i in prange(5, schedule='guided'):
    #    j = 0
    #with nogil, parallel():
    for name, neuron in self.neurons.iteritems():
      #results.append((name, pool.apply_async(neuron.recognize, args = (image, )).get()))
      results.append((name, neuron.recognize(image)))
    return results