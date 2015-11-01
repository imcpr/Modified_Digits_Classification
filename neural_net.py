# coding=utf-8
# Authors:
#   Kian Kenyon-Dean wrote the architecture of the network and all of the objects.
#
# Coding began October 31st, 2015

import numpy as np
from nn_math import sigmoid as sig
from operator import attrgetter

sigmoid = sig
deriv_sigmoid = None

""" The main neural network classifier object and its methods are here. """
class NeuralNetwork(object):
    """ Our neural network. """

    def __init__(self, num_inputs, hidden_layers_nodes, num_outputs, error=lambda y,hat: 0.5*(y-hat)**2, sigmoid_constant=1):
        """ hidden_layers_nodes: a python list whose length is the number of hidden layers and values are number of nodes per layer. """
        global sigmoid,deriv_sigmoid

        self.network = [[] for _ in range(len(hidden_layers_nodes)+2)] # We have h hidden layers plus 1 input layer and 1 output layer.
        self.__make_input_layer(num_inputs)
        self.__make_hidden_layers(hidden_layers_nodes)
        self.__make_output_layer(num_outputs)
        self.error = error
        sigmoid = sig(0, c=sigmoid_constant)
        deriv_sigmoid = lambda x: sigmoid(x)*(1 - sigmoid(x))

    # ------------ Private initialization functions. ---------------- #
    def __make_input_layer(self, num_inputs):
        """ We add +1 so as to have an x0 initial value. """
        for _ in range(num_inputs+1):
            self.network[0].append(InputNode())

    def __make_output_layer(self, num_outputs):
        for i in range(num_outputs):
            self.network[-1].append(OutputNode(i+1)) # Here I am setting the class values to be 1+ the actual value (so that we don't have 0 as a class).

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

                # Connect each node in the previous layer to the current node we are making.
                for node in self.network[h-1]:
                    self.__connect(node,n)

    @staticmethod
    def __connect(from_node, to_node):
        s = Synapse(from_node, to_node)
        from_node.add_out_synapse(s)
        to_node.add_in_synapse(s)

    # ------------ Training functions for taking in samples. ------------- #
    def fit(self, X, y, training_horizon=5, grad_descent='stochastic'):
        """ :training_horizon designates how many times we iterate through the training set. """
        assert len(X) == len(y)

        if grad_descent == 'stochastic':
            for _ in range(training_horizon):
                for i in range(len(X)):
                    self.initialize_input_sample(X[i])
                    error = self.error(y[i],self.feed_forward())

                    # TODO: backpropagate

    def initialize_input_sample(self, input_array):
        """ Initialize the values of all of the input nodes to equal each value in the input_array. I assume the input_array has shape (m,)"""
        assert len(input_array)+1 == len(self.network[0])
        self.network[0][0].set_value(1.0) # Set the dummy variable.
        for i in range(len(input_array)):
            self.network[0][i+1].set_value(input_array[i])

    def feed_forward(self):
        """ Returns the network's prediction. """
        for k in range(len(self.network)): # For each layer, k...
            for n in self.network[k]:
                n.set_value(n.output())

        #TODO: prediction analysis
        """
         I do not know what the prediction of a network will be. Here I assume that the prediction will simply be
         the output node that has the highest value after feeding forward, but this may not be the case!!!
        """
        return max(self.network[-1], key=attrgetter('value'))

    # ------------ Debugging functions ------------- #
    def count(self):
        synapses = 0
        for k in range(len(self.network)):
            for node in self.network[k]:
                synapses += len(node.in_synapses)
            print 'Number of nodes in layer %d: %d'%(k, len(self.network[k]))
        print 'Total number of synapses: {:,}'.format(synapses)

#----------------- Node super class -----------------#
class Node(object):
    """ The super class for all Nodes. """
    def __init__(self):
        self.value = None
        self.in_synapses = []
        self.out_synapses = []

    def add_in_synapse(self, s):
        self.in_synapses.append(s)

    def add_out_synapse(self, s):
        self.out_synapses.append(s)

    def set_value(self, v):
        self.value = v

    def get_value(self):
        return self.value

    def get_synapse_weights(self):
        return np.array([s.weight for s in self.in_synapses])

    def get_node_input_values(self):
        return np.array([s.in_node.get_value() for s in self.in_synapses])

    def output(self):
        global sigmoid
        return sigmoid(np.dot(self.get_synapse_weights(),self.get_node_input_values()))

#----------------- Input node class -----------------#
class InputNode(Node):
    """ All training features will each have a single input node. """
    def __init__(self):
        Node.__init__(self)

    def output(self):
        return self.value

#----------------- Output node class -----------------#
class OutputNode(Node):
    """ This node represents an output classifcation. """
    def __init__(self, classification):
        Node.__init__(self)
        self.classification = classification

#----------------- Neuron class -----------------#
class Neuron(Node):
    """ A neuron (a special type of node). """
    def __init__(self):
        Node.__init__(self)
        self.learning_rate = 1.0

#----------------- Synapse class -----------------#
class Synapse:
    """ Connects nodes. One node on each end per synapse. """
    def __init__(self, in_node, out_node):
        self.in_node = in_node
        self.out_node = out_node
        self.weight = 1.0

