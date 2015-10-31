# coding=utf-8
# Authors:
#   Kian Kenyon-Dean wrote the architecture of the network and all of the objects.
#
# Coding began October 31st, 2015

""" The main neural network classifier object and its methods are here. """

class NeuralNetwork:
    """ Our neural network. """

    def __init__(self, hidden_layers_nodes, num_inputs, num_outputs):
        """ hidden_layers_nodes: a python list whose length is the number of hidden layers and values are number of nodes per layer. """

        self.network = [[] for _ in range(len(hidden_layers_nodes)+2)] # We have h hidden layers plus 1 input layer and 1 output layer.
        self.make_input_layer(num_inputs)
        self.make_hidden_layers(hidden_layers_nodes)
        self.make_output_layer(num_outputs)

    def make_input_layer(self, num_inputs):
        for _ in range(num_inputs):
            self.network[0].append(InputNode())

    def make_output_layer(self, num_outputs):
        for _ in range(num_outputs):
            self.network[-1].append(OutputNode())

        # Here we are connecting each node in the previous layer to each node in the output layer.
        for node in self.network[-2]:
            for output_node in self.network[-1]:
                self.connect(node, output_node)

    def make_hidden_layers(self, hidden_layers_nodes):
        # Iterate through hidden_layers_nodes list - each element is the number of nodes we make for the hidden layer.
        for i in range(len(hidden_layers_nodes)):
            num_nodes_to_make = hidden_layers_nodes[i]
            h = i+1 # We are making the h^th layer of the network.

            for _ in range(num_nodes_to_make):
                n = Neuron()
                self.network[h].append(n)

                # Connect each node in the previous layer to the current node we are making.
                for node in self.network[h-1]:
                    self.connect(node,n)

    @staticmethod
    def connect(from_node, to_node, weight=1.0):
        s = Synapse(weight=weight)
        from_node.add_out_synapse(s)
        to_node.add_in_synapse(s)


#----------------- Node super class -----------------#
class Node:
    """ The super class for all nodes. """
    def __init__(self):
        self.value = None
        self.in_synapses = []
        self.out_synapses = []

    def add_in_synapse(self, s):
        self.in_synapses.append(s)

    def add_out_synapse(self, s):
        self.out_synapses.append(s)


#----------------- Input node class -----------------#
class InputNode(Node):
    """ All training features will each have a single input node. """
    def __init__(self):
        Node.__init__(self)




#----------------- Output node class -----------------#
class OutputNode(Node):
    """ This object represents an output classifcations node for the network. """
    def __init__(self):
        Node.__init__(self)


#----------------- Neuron class -----------------#
class Neuron(Node):
    """ A neuron (a special type of node). """

    def __init__(self, learning_rate=1.0):
        Node.__init__(self)
        self.learning_rate = learning_rate

#----------------- Synapse class -----------------#
class Synapse:
    """ Connects nodes. One node on each end per synapse. """
    def __init__(self, weight=1.0):
        self.weight = weight
