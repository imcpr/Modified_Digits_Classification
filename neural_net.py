# coding=utf-8
# Authors:
#   Kian Kenyon-Dean wrote the architecture of the network and all of the objects.
#
# Coding began October 31st, 2015

import numpy as np
from nn_math import sigmoid as sig
from operator import attrgetter

sigmoid = sig

""" The main neural network classifier object and its methods are here. """
class NeuralNetwork(object):
    """ Our neural network. """

    def __init__(self, num_inputs, hidden_layers_nodes, num_outputs, error=lambda y,hat: 0.5*(y-hat)**2, sigmoid_constant=1, dummy=True):
        """ hidden_layers_nodes: a python list whose length is the number of hidden layers and values are number of nodes per layer. """
        global sigmoid
        self.dummy = 1 if dummy else 0
        self.network = [[] for _ in range(len(hidden_layers_nodes)+2)] # We have h hidden layers plus 1 input layer and 1 output layer.
        self.__make_input_layer(num_inputs)
        self.__make_hidden_layers(hidden_layers_nodes)
        self.__make_output_layer(num_outputs)
        self.error = error
        sigmoid = sig(0, c=sigmoid_constant)

    # ------------ Initialization functions. ---------------- #
    def __make_input_layer(self, num_inputs):
        """ We add +1 so as to have an x0 initial value. """
        for _ in range(num_inputs + self.dummy):
            self.network[0].append(Neuron())

    def __make_output_layer(self, num_outputs):
        for i in range(num_outputs):
            self.network[-1].append(Neuron(classification=i)) #i is the classification associated with the neuron.

        # Here we are connecting each node in the previous layer to each node in the output layer.
        for node in self.network[-2]:
            for output_node in self.network[-1]:
                self.__connect(node, output_node)

    def __make_hidden_layers(self, hidden_layers_nodes):
        # Iterate through hidden_layers_nodes list - each element is the number of nodes we make for the hidden layer.
        for i in range(len(hidden_layers_nodes)):
            num_nodes_to_make = hidden_layers_nodes[i]
            h = i+1 # We are making the h^th layer of the network.
            for _ in range(num_nodes_to_make):
                n = Neuron()
                self.network[h].append(n)
                for node in self.network[h-1]:
                    self.__connect(node,n) # Connect each node in the previous layer to the current node we are making.

    @staticmethod
    def __connect(from_node, to_node):
        s = Synapse(from_node, to_node)
        from_node.add_out_synapse(s)
        to_node.add_in_synapse(s)

    def initialize_weights(self, weights_layers):
        assert len(weights_layers) == len(self.network)-1
        for k in range(len(self.network)-1):
            j = 0
            for node in self.network[k]:
                for synapse in node.out_synapses:
                    synapse.weight = weights_layers[k][j]
                    j += 1

    # ------------ Training functions for taking in samples. ------------- #
    def fit(self, X, y, training_horizon=5, grad_descent='stochastic', verbose=False):
        """ training_horizon designates how many times we iterate through the training set. """
        assert len(X) == len(y)
        if grad_descent == 'stochastic':
            for _ in range(training_horizon):
                for i in range(len(X)):
                    if verbose:
                        if i % 100 == 0:
                            print '%d... '%i,
                    self.feedforward(X[i])
                    self.backpropagate(y[i])

    def input(self, input_array):
        """ Initialize the values of all of the input nodes to equal each value in the input_array. I assume the input_array has shape (m,)"""
        assert len(input_array) + self.dummy == len(self.network[0])
        if self.dummy:
            self.network[0][0].set_value(1.0) # Set the dummy variable.
        for i in range(len(input_array)):
            self.network[0][i + self.dummy].set_value(input_array[i])

    def feedforward(self, sample):
        """ Returns the network's prediction. """
        self.input(sample)
        for k in range(len(self.network)): # For each layer, k...
            for n in self.network[k]:
                n.output() # This is all we need because output() sets the nodes value as well.

        #The prediction will be the node with the highest value (closest to 1.0).
        if len(self.network[-1]) > 1:
            prediction_node = max(self.network[-1], key=attrgetter('value'))
            return prediction_node.classification
        else:
            return self.network[-1][0].value

    def backpropagate(self, target_value):
        """ Trains the network. """
        for i in range(len(self.network[-1])):
            if type(target_value) == int:
                target = 1.0 if i==target_value else 0.0 # We want to return a 1 for the correct classification and 0 for the other nodes.
            else:
                target = target_value

            self.network[-1][i].error(target)
            self.network[-1][i].update_weights()

        for k in reversed(range(1,len(self.network)-1)):
            for node in self.network[k]:
                node.hidden_error()
                node.update_weights()

    def predict(self, X):
        return np.array([self.feedforward(sample) for sample in X])

    # ------------ Debugging functions ------------- #
    def count(self):
        synapses = 0
        for k in range(len(self.network)):
            for node in self.network[k]:
                synapses += len(node.in_synapses)
            print 'Number of nodes in layer %d: %d'%(k, len(self.network[k]))
        print 'Total number of synapses: {:,}'.format(synapses)

    def __repr__(self):
        s = ''
        for k in range(len(self.network)):
            s += 'Layer %d:\n'%k
            for node in self.network[k]:
                s += '\tNode value: %0.4f\n'%node.value
                for syn in node.out_synapses:
                    s += '\t  synapse weight: %0.5f\n'%syn.weight
        return s
#----------------- Neuron class -----------------#
class Neuron(object):
    """ The super class for all Nodes. """
    def __init__(self, classification=None, learning_rate=1.0):
        self.value = None
        self.delta = 0.0
        self.in_synapses = []
        self.out_synapses = []
        self.learning_rate = learning_rate
        self.classification = classification

    def add_in_synapse(self, s):
        self.in_synapses.append(s)

    def add_out_synapse(self, s):
        self.out_synapses.append(s)

    def set_value(self, v):
        self.value = v

    def get_value(self):
        return self.value

    def get_in_synapse_weights(self):
        return np.array([s.weight for s in self.in_synapses])

    def get_node_input_values(self):
        return np.array([s.in_node.value for s in self.in_synapses])

    def output(self):
        global sigmoid
        if self.in_synapses != []:
            self.value = sigmoid( np.dot(self.get_in_synapse_weights(),self.get_node_input_values()) )
        return self.value

    def error(self, target):
        self.delta = self.value * (1.0 - self.value) * (target - self.value)
        return self.delta

    def hidden_error(self):
        self.delta = self.value * (1.0 - self.value) * sum([synapse.out_node.delta * synapse.weight for synapse in self.out_synapses])
        return self.delta

    def update_weights(self):
        for synapse in self.in_synapses:
            synapse.weight += self.learning_rate * self.delta * synapse.in_node.value

#----------------- Synapse class -----------------#
class Synapse:
    """ Connects nodes. One node on each end per synapse. """
    def __init__(self, in_node, out_node, weight=1.0):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = weight

