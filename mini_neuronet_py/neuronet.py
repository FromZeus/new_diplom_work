import neuron
import pathos.multiprocessing as mp
from multiprocessing import Process, Queue
import multiprocessing as def_mp
import ctypes
import dummy

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

  def recognize(self, image):
    results = []
    qu = Queue()
    mp_image = def_mp.Array(ctypes.c_int, image)
    pool = mp.Pool(initializer = test_init, initargs = (mp_image, ))
    for name, neuron in self.neurons.iteritems():
      p = Process(target = neuron.recognize, args = (image, name, qu))
      p.start()
      p.join()
    while not qu.empty():
      results.append(qu.get())
      #results.append((name, pool.apply_async(neuron.recognize).get()))
      #results.append((name, neuron.recognize(image)))
    return results
    #for name, res in results:
      #print u"{0}: {1}".format(name, res) 
      #res = pool.apply_async(neuron.recognize, (image, ))
      #print u"{0}: {1}".format(name, res.get()) 
      #print u"{0}: {1}".format(name, neuron.recognize(image))