"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import csv
import sys
import timeit
import numpy
import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from features import get_train, get_test


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


class ConvNet:
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """


    def shared_dataset(self, data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    def set_data(self, data_x, data_y):
        octet = len(data_x)/8

        self.train_set_x, self.train_set_y = self.shared_dataset((data_x[:octet*6], data_y[:octet*6]))
        self.valid_set_x, self.valid_set_y = self.shared_dataset((data_x[octet*6:octet*7], data_y[octet*6:octet*7]))
        self.test_set_x, self.test_set_y = self.shared_dataset((data_x[octet*7:], data_y[octet*7: ]))

    def __init__(self, data_x, data_y, learning_rate=0.1, n_epochs=100,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
        self.rng = numpy.random.RandomState(23455)
        self.nkerns = nkerns
         # pX = PCA(n_components=28*28).fit_transform(X)

        self.set_data(data_x, data_y)


        # kX = theano.shared(numpy.asarray(kaggle_input,
        #                                 dtype=theano.config.floatX), borrow=True)
        # n_kaggle_batches = kX.get_value(borrow=True).shape[0]
        # n_kaggle_batches /= batch_size

        self.batch_size = batch_size
        # compute number of minibatches for training, validation and testing
        self.n_train_batches = self.train_set_x.get_value(borrow=True).shape[0]
        self.n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0]
        self.n_test_batches = self.test_set_x.get_value(borrow=True).shape[0]
        self.n_train_batches /= self.batch_size
        self.n_valid_batches /= self.batch_size
        self.n_test_batches /= self.batch_size



        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
        kX = T.matrix('kX')
        # start-snippet-1
        self.x = T.matrix('x')   # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
                            # [int] labels

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print '... building the model'

        w, h = 48, 48
        w1, h1 = (w-5+1)/2, (h-5+1)/2
        w2, h2 = (w1-7+1)/2, (h1-7+1)/2

        # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
        # to a 4D tensor, compatible with our LeNetConvPoolLayer
        # (28, 28) is the size of MNIST images.
        self.layer0_input = self.x.reshape((self.batch_size, 1, w, h))

        # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
        self.layer0 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer0_input,
            image_shape=(self.batch_size, 1, w, h),
            filter_shape=(self.nkerns[0], 1, 5, 5),
            poolsize=(2, 2)
        )

        # Construct the second convolutional pooling layer
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
        self.layer1 = LeNetConvPoolLayer(
            self.rng,
            input=self.layer0.output,
            image_shape=(self.batch_size, self.nkerns[0], w1, h1),
            filter_shape=(self.nkerns[1], self.nkerns[0], 7, 7),
            poolsize=(2, 2)
        )

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
        # or (500, 50 * 4 * 4) = (500, 800) with the default values.
        self.layer2_input = self.layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        self.layer2 = HiddenLayer(
            self.rng,
            input=self.layer2_input,
            n_in=self.nkerns[1] * w2 * h2,
            n_out=500,
            activation=T.tanh
        )

        # classify the values of the fully-connected sigmoidal layer
        self.layer3 = LogisticRegression(input=self.layer2.output, n_in=500, n_out=10)

        # the cost we minimize during training is the NLL of the model
        self.cost = self.layer3.negative_log_likelihood(self.y)

        # create a function to compute the mistakes that are made by the model
        self.test_model = theano.function(
            [index],
            self.layer3.errors(self.y),
            givens={
                self.x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: self.test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        self.validate_model = theano.function(
            [index],
            self.layer3.errors(self.y),
            givens={
                self.x: self.valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: self.valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )

        self.compute_kaggle = theano.function(
            [kX, index],
            self.layer3.y_pred,
            givens={
                self.x: kX[index * batch_size: (index + 1) * batch_size],
            }
        )

        # create a list of all model parameters to be fit by gradient descent
        self.params = self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params

        # create a list of gradients for all model parameters
        self.grads = T.grad(self.cost, self.params)

        # train_model is a function that updates the model parameters by
        # SGD Since this model has many parameters, it would be tedious to
        # manually create an update rule for each model parameter. We thus
        # create the updates list by automatically looping over all
        # (params[i], grads[i]) pairs.
        updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(self.params, self.grads)
        ]

        self.train_model = theano.function(
            [index],
            self.cost,
            updates=updates,
            givens={
                self.x: self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
                self.y: self.train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )
        # end-snippet-1

    def predict(self, kaggle_input):
        n_batches = len(kaggle_input)/self.batch_size
        preds = np.zeros((1,1))
        for idx in range(n_batches):
            preds = np.vstack((preds,self.compute_kaggle(kaggle_input, idx).reshape(self.batch_size,1)))
        return preds[1:]


    def train(self, n_epochs=100):

        ###############
        # TRAIN MODEL #
        ###############
        print '... training'
        # early-stopping parameters
        patience = 10000  # look as this many examples regardless
        patience_increase = 2  # wait this much longer when a new best is
                               # found
        improvement_threshold = 0.995  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(self.n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatche before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.
        start_time = timeit.default_timer()

        epoch = 0
        done_looping = False

        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(self.n_train_batches):

                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if iter % 100 == 0:
                    print 'training @ iter = ', iter
                cost_ij = self.train_model(minibatch_index)

                if (iter + 1) % validation_frequency == 0:

                    # compute zero-one loss on validation set
                    validation_losses = [self.validate_model(i) for i
                                         in xrange(self.n_valid_batches)]
                    this_validation_loss = numpy.mean(validation_losses)
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                          (epoch, minibatch_index + 1, self.n_train_batches,
                           this_validation_loss * 100.))

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                           improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = [
                            self.test_model(i)
                            for i in xrange(self.n_test_batches)
                        ]
                        test_score = numpy.mean(test_losses)
                        # preds = compute_kaggle(0)
                        # print preds
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, self.n_train_batches,
                               test_score * 100.))

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print('Optimization complete.')
        print('Best validation score of %f %% obtained at iteration %i, '
              'with test performance %f %%' %
              (best_validation_loss * 100., best_iter + 1, test_score * 100.))


if __name__ == '__main__':
    X, Y = get_train(30000)
    cnn = ConvNet(X,Y)
    cnn.train(10)
    # kaggle_input = get_test(20000)
    # cnn.predict(kaggle_input)


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)

def predict(kaggle_input):
    n_batches = len(kaggle_input)/cnn.batch_size
    preds = np.zeros((1,1))
    for idx in range(n_batches):
        # print cnn.compute_kaggle(kaggle_input, idx)
        preds = np.vstack((preds,cnn.compute_kaggle(kaggle_input, idx).reshape(cnn.batch_size,1)))
    return preds[1:]

def save_preds(preds, filename):
    with open (filename, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Id", "Prediction"])
        for i in range(len(preds)):
            writer.writerow([i+1] + [int(preds[i].sum())])