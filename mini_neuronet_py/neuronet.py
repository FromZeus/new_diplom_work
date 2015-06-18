import neuron
import pathos.multiprocessing as mp

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
    pool = mp.Pool()
    results = []
    for name, neuron in self.neurons.iteritems():
      results.append((name, pool.apply_async(neuron.recognize, (image, )).get()))
      #results.append((name, neuron.recognize(image)))
    return results
    #for name, res in results:
      #print u"{0}: {1}".format(name, res) 
      #res = pool.apply_async(neuron.recognize, (image, ))
      #print u"{0}: {1}".format(name, res.get()) 
      #print u"{0}: {1}".format(name, neuron.recognize(image))