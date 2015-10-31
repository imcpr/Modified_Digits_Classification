# coding=utf-8
# @author Kian Kenyon-Dean

""" The main neural network classifier object and its methods are here. """

class NeuralNetworkClassifier:
    """ Our neural network that perform classification. """
    def __init__(self, num_hidden_layers, hidden_nodes_per_layer):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_nodes_per_layer = hidden_nodes_per_layer


    def fit(self, X, y):
        return self.num_hidden_layers

class InputNode:
    """ All training features will each have a single input node. """
    def __init__(self, value):
        self.value = value

class Neuron:
    """ A neuron. """
    def __init__(self):
        self.x = 0

class Synapse:
    def __init__(self):
        self.x = 0

