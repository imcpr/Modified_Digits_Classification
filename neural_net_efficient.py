# coding=utf-8
# Authors:
#   Kian Kenyon-Dean wrote the architecture of the network and all of the objects.
#   Yann Long changed the weights initialization
#
# Coding began October 31st, 2015

import numpy as np
import pyprind
import copy
from scipy.special import expit
from math import sqrt


""" The main neural network classifier object and its methods are here. """
class NeuralNetwork(object):
    """ Our neural network. """

    def __init__(self, num_inputs, hidden_layers_nodes, num_outputs, dummy=True, learning_rate=1.0, dropout=None, maxnorm=None,schedule=None,update=None, seed=1717):
        """ hidden_layers_nodes: a python list whose length is the number of hidden layers and values are number of nodes per layer. """
        np.random.seed(seed)

        self.dummy = num_outputs if dummy else 0
        self.lr = learning_rate
        self.p = dropout
        self.c = maxnorm
        self.sched = schedule
        self.up = update

        # quantities of nodes
        self.ni = num_inputs + self.dummy
        self.nhl = len(hidden_layers_nodes) # num hidden layers
        self.nhn = list(np.array(hidden_layers_nodes) + self.dummy) # num hidden nodes per layer
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
            weights.append(np.random.randn(len(self.nodes[k]),len(self.nodes[k+1]))*sqrt(2.0/len(self.nodes[k])))
        self.weights = np.array(weights)
        
        # initialize Dropout matrix
        if self.p:
            assert ((self.p < 1.0) and (self.p > 0.0)),"Dropout must be between 0.0 and 1.0"
            dom = []
            for k in range(self.nhl):
                dom.append(np.ones(self.nhn[k]))
            self.dom = np.array(dom)
        
        # initialize momentum matrix
        momentums = []
        for k in range(len(self.nodes)-1):
            momentums.append(np.random.uniform(0,1, size=(len(self.nodes[k]),len(self.nodes[k+1]))))
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
            print 'Fitting the Neural Network...'
            bar = pyprind.ProgBar(training_horizon*len(X))
        if grad_descent == 'stochastic':
            for j in range(training_horizon):
                for i in range(len(X)):
                    if verbose:
                        bar.update()    
                    self.feedforward(X[i])
                    self.backpropagate(y[i])
                            
                if self.c:
                    self.maxnorm()
                    
                if self.sched:
                    if j % self.sched == 0:
                        self.lr *= self.up

    def input(self, input_array):
        """ Initialize the values of all of the input nodes to equal each value in the input_array. I assume the input_array has shape (m,)"""
        assert len(input_array) + self.dummy == len(self.nodes[0])
        self.nodes[0][self.dummy:] = input_array

    def feedforward(self, sample):
        """ Returns the network's prediction. """              
        self.input(sample)
        if self.p:
            self.random_dropout()
            for k in range (len(self.nodes)-2):
                self.nodes[k+1]=(expit(self.nodes[k].dot(self.weights[k])))*self.dom[k]
            self.nodes[-1]=expit(self.nodes[-2].dot(self.weights[-1]))
            return np.argmax(self.nodes[-1])
        else:
            for k in range(len(self.nodes)-1):
                self.nodes[k+1] = expit(self.nodes[k].dot(self.weights[k]))
            return np.argmax(self.nodes[-1])

    def backpropagate(self, target_value):
        """ Trains the network. """
        targets = np.array([1.0 if i==target_value else 0.0 for i in range(len(self.nodes[-1]))])
        self.errors[-1] = self.nodes[-1] * (1.0 - self.nodes[-1]) * (targets - self.nodes[-1])
        self.update_weights(len(self.weights)-1)

        for k in reversed(range(1,len(self.nodes)-1)):
            self.errors[k] = self.nodes[k] * (1.0 - self.nodes[k]) * np.dot(self.weights[k], self.errors[k+1])
            self.update_weights(k-1)

    def update_weights(self, layer):
        # for i in range(len(self.nodes[layer])):
        #     self.weights[layer][i,:] += self.lr * self.errors[layer+1][:] * self.nodes[layer][i]
        self.weights[layer] += self.lr * (self.errors[layer+1][:,np.newaxis] * self.nodes[layer][:][np.newaxis]).T

    def predict(self, X, verbose=False):
        pred = []
        if verbose:
            bar = pyprind.ProgBar(len(X))
        if self.p:
            for i in range(len(X)):
                self.input(X[i])
                for k in range (len(self.nodes)-2):
                    self.nodes[k+1]=(expit(self.nodes[k].dot(self.weights[k])))/self.p
                    # self.nodes[k+1]=(expit(self.nodes[k].dot(self.weights[k])))*self.p
                pred.append(np.argmax(expit(self.nodes[-2].dot(self.weights[-1]))))
            if verbose:
                bar.update()
        else:
            for i in range(len(X)):
                self.input(X[i])
                for k in range(len(self.nodes)-1):
                    self.nodes[k+1] = expit(self.nodes[k].dot(self.weights[k]))
                pred.append(np.argmax(self.nodes[-1]))
            if verbose:
                bar.update()
        
        return np.array(pred)
        
    def random_dropout(self):
        for k in range(self.nhl):
            self.dom[k]=np.r_[1, np.random.binomial(1, self.p, size=(self.nhn[k]-1))]
    
    def maxnorm(self):
        for k in range(len(self.nodes)-1):
            for row in self.weights[k]:
                norm = np.linalg.norm(row)
                if norm > self.c:
                    row *= self.c/norm


