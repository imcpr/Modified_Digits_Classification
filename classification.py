# coding=utf-8
# Authors:
#   Kian Kenyon-Dean.
#
# Coding began October 31st, 2015

from import_data import import_csv
from neural_net import NeuralNetwork
from sklearn.feature_selection import SelectKBest,chi2
import time
import numpy as np

data_files_path = '/Users/kian/documents/COMP598/Assignment3/data_and_scripts/'

TRAIN_INPUTS_PATH = data_files_path+'train_inputs_subset.csv'
TRAIN_OUTPUTS_PATH = data_files_path+'train_outputs_subset.csv'
TEST_INPUTS_PATH = data_files_path+'test_inputs.csv'

def accuracy(predictions, actual):
    assert len(predictions) == len(actual)
    right = 0
    for i in range(len(predictions)):
        right += 1 if predictions[i] == actual[i] else 0
    print '%d correct out of %d. %0.3f accuracy.'%(right, len(predictions), float(right)/len(predictions))

if __name__ == '__main__':
    # Below provides a good test to show that it works succesfully based on the example given
    # on this paper (pg. 20): https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

    # nn = NeuralNetwork(2, [2], 1, dummy=False)
    # nn.initialize_weights([[0.1, 0.4, 0.8, 0.6],[0.3, 0.9]])
    # nn.fit(np.array([[0.35, 0.9]]), np.array([0.5]), training_horizon=1)


    starttime = time.clock()
    train_outputs = import_csv(TRAIN_OUTPUTS_PATH).astype(int)
    train_inputs = import_csv(TRAIN_INPUTS_PATH)
    print 'Time to import: %0.1f'%(time.clock() - starttime)

    num_features = len(train_inputs[0])
    num_classes = 10

    starttime = time.clock()
    nn = NeuralNetwork(num_features, [100], num_classes, dummy=True)
    train_x = train_inputs[0:500]
    train_y = train_outputs[0:500]
    print 'Training network...'
    nn.fit(train_x, train_y, training_horizon=1, verbose=True)
    print 'Time to train: %0.1f'%(time.clock() - starttime)

    test_x = train_inputs[4000:4500]
    test_y = train_outputs[4000:4500]
    p = nn.predict(test_x)
    accuracy(p, test_y)


