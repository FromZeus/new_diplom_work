import neuron
from multiprocessing import Pool

class HopfNet:

  neurons = dict()
  rec_objs = []

  def __init__(self, rec_objs):
    self.rec_objs = rec_objs
    for el in self.rec_objs:
      self.neurons[el[0]] = neuron.HopfNeuron(el[1])

  def learn(self, image, name):
    self.neurons[name].learn(image)

  def recognize(self, image):
    pool = Pool()
    for name, neuron in self.neurons.iteritems():
      res = pool.apply_async(neuron.recognize, [image])
      print u"{0}: {1}".format(name, res.get(timeout=10)) 
      #print u"{0}: {1}".format(name, neuron.recognize(image))