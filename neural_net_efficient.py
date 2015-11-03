# coding=utf-8
# Authors:
#   Kian Kenyon-Dean wrote the architecture of the network and all of the objects.
#
# Coding began October 31st, 2015

import numpy as np
import pyprind
import copy
from scipy.special import expit


""" The main neural network classifier object and its methods are here. """
class NeuralNetwork(object):
    """ Our neural network. """

    def __init__(self, num_inputs, hidden_layers_nodes, num_outputs, dummy=True, learning_rate=1.0, weight_range=(-1.0,1.0), seed=1917):
        """ hidden_layers_nodes: a python list whose length is the number of hidden layers and values are number of nodes per layer. """
        np.random.seed(seed)

        self.dummy = 1 if dummy else 0
        self.lr = learning_rate

        # quantities of nodes
        self.ni = num_inputs + dummy
        self.nhl = len(hidden_layers_nodes) # num hidden layers
        self.nhn = list(np.array(hidden_layers_nodes) + dummy) # num hidden nodes per layer
        self.no = num_outputs

        # initialize node matrix
        nodes = [np.ones(self.ni)]
        for i in range(self.nhl):
            nodes.append(np.ones(self.nhn[i]))
        nodes.append(np.ones(self.no))
        self.nodes = np.array(nodes)

        # initialize error matrix
        self.errors = copy.deepcopy(self.nodes)

        # initialize weight matrix such that weights[k][(b,c)] is the weight from node b on layer k to node c on layer k+1
        weights = []
        for k in range(len(self.nodes)-1):
            weights.append(np.random.uniform(low=weight_range[0],high=weight_range[1], size=(len(self.nodes[k]),len(self.nodes[k+1]))))
        self.weights = np.array(weights)

        # initialize momentum matrix
        momentums = []
        for k in range(len(self.nodes)-1):
            momentums.append(np.random.uniform(low=weight_range[0],high=weight_range[1], size=(len(self.nodes[k]),len(self.nodes[k+1]))))
        self.momentums = np.array(momentums)

        del nodes,weights,momentums

        # set error function that we use
        self.error = lambda target,output: output * (1.0 - output) * (target - output)

    # ------------ Training functions for taking in samples. ------------- #
    def fit(self, X, y, training_horizon=5, grad_descent='stochastic', verbose=False):
        """ training_horizon designates how many times we iterate through the training set. """
        assert len(X) == len(y)
        assert len(X[0])+self.dummy == len(self.nodes[0])
        if verbose:
            bar = pyprind.ProgBar(training_horizon*len(X))
        if grad_descent == 'stochastic':
            for _ in range(training_horizon):
                for i in range(len(X)):
                    if verbose:
                        bar.update()
                    self.feedforward(X[i])
                    self.backpropagate(y[i])

    def input(self, input_array):
        """ Initialize the values of all of the input nodes to equal each value in the input_array. I assume the input_array has shape (m,)"""
        assert len(input_array) + self.dummy == len(self.nodes[0])
        self.nodes[0][self.dummy:] = input_array

    def feedforward(self, sample):
        """ Returns the network's prediction. """
        self.input(sample)
        for k in range(1,len(self.nodes)):
            self.nodes[k] = expit(self.weights[k-1][:,:].T.dot(self.nodes[k-1][:]))
        return np.argmax(self.nodes[-1])

    def backpropagate(self, target_value):
        """ Trains the network. """
        targets = np.array([1.0 if i==target_value else 0.0 for i in range(len(self.nodes[-1]))])
        # print 'targets',targets
        self.errors[-1] = self.nodes[-1][:] * (1.0 - self.nodes[-1][:]) * (targets - self.nodes[-1][:])
        # print 'errors',self.errors[-1]

        self.update_weights(len(self.weights)-1)
        # for i in range(len(self.nodes[-2])):
        #     self.weights[-1][i,:] += self.lr * self.errors[-1][:] * self.nodes[-2][i]
        # print self.weights
            # for j in range((self.no)):
            #     print self.weights[-1][i,j],self.lr, self.errors[-1][j], self.nodes[-2][i]
            #     self.weights[-1][i,j] += self.lr * self.errors[-1][j] * self.nodes[-2][i]
            #     print self.weights[-1][i,j]
            # self.weights[-1][i,:] += self.lr * self.errors[-1][:] * self.nodes[-1][i]

        for k in reversed(range(1,len(self.nodes)-1)):
            # print k
            self.errors[k] = self.nodes[k][:] * (1.0 - self.nodes[k][:]) * np.dot(self.errors[k+1][:], self.weights[k][:][:].T)
            self.update_weights(k-1)
            # for i in range(len(self.nodes[k])):
            #     self.weights[k][i,:] += self.lr * self.errors[k][:] * self.nodes[k][i]
        return

    def update_weights(self, layer):
        # print self.errors[layer+1]

        for i in range(len(self.nodes[layer])):
            self.weights[layer][i,:] += self.lr * self.errors[layer+1][:] * self.nodes[layer][i]
        # print self.weights

    def predict(self, X, verbose=False):
        p = []
        if verbose:
            bar = pyprind.ProgBar(len(X))
        for i in range(len(X)):
            p.append(self.feedforward(X[i]))
            if verbose:
                bar.update()
        return np.array(p)