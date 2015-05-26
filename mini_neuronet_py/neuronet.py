import neuron

class HopfNet:

  neurons = dict()
  rec_objs = []

  def __init__(self, rec_objs):
    self.rec_objs = rec_objs
    for el in self.rec_objs:
      self.neurons[el[0]] = neuron.HopfNeuron(el[1])

  def learn(self, img, name):
    self.neurons[name].learn(img)

  def recognize(self, img):
    for name, neuron in self.neurons.iteritems():
      print "{0}: {1}".format(name, neuron.recognize(img))