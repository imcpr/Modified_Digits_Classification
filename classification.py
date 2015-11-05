# coding=utf-8
# Authors:
#   Kian Kenyon-Dean.
#   Yann Long
#
# Coding began October 31st, 2015

import numpy as np
from import_data import import_csv
from neural_net_efficient import NeuralNetwork
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
import os


data_files_path = 'data_and_scripts/'

TRAIN_INPUTS_PATH = data_files_path+'transformed_train_inputs.csv'
TRAIN_OUTPUTS_PATH = data_files_path+'train_outputs.csv'
TEST_INPUTS_PATH = data_files_path+'transformed_test_inputs.csv'

TRAIN_INPUTS_SUBSET_PATH = data_files_path+'train_inputs_subset.csv'
TRAIN_OUTPUTS_SUBSET_PATH = data_files_path+'train_outputs_subset.csv'

def feature_reduce(dataset, desired):
    print 'Reducing feature set size from %d to %d...'%(len(dataset[0]),desired)
    return PCA(n_components=desired).fit_transform(dataset)

def accuracy(predictions, actual):
    assert len(predictions) == len(actual)
    right = 0
    for i in range(len(predictions)):
        right += 1 if predictions[i] == actual[i] else 0
    print '%d correct out of %d. %0.3f accuracy.'%(right, len(predictions), float(right)/len(predictions))
    
def heatmap(predictions, actual, filename):

    cm = confusion_matrix(actual, predictions)
    plt.matshow(cm)
    plt.title('Confusion matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    i = 0
    while os.path.exists('{}{:d}.png'.format(filename, i)):
        i += 1
    plt.savefig('{}{:d}.png'.format(filename, i))
    
if __name__ == '__main__':
    # Below provides a good test to show that it works succesfully based on the example given
    # on this paper (pg. 20): https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

    # nn = NeuralNetwork(2, [2], 1, dummy=False)
    # nn.initialize_weights([[0.1, 0.4, 0.8, 0.6],[0.3, 0.9]])
    # nn.fit(np.array([[0.35, 0.9]]), np.array([0.5]), training_horizon=1)

    starttime = time.clock()
    train_outputs = import_csv(TRAIN_OUTPUTS_SUBSET_PATH).astype(int)
    train_inputs = import_csv(TRAIN_INPUTS_SUBSET_PATH)
    print 'Time to import: %0.1f'%(time.clock() - starttime)
    
    print np.shape(train_outputs)
    print np.shape(train_inputs)   

    #randomly split the data into a train set and a validation set
    train_x, test_x, train_y, test_y = train_test_split(train_inputs, train_outputs, test_size=0.2, random_state=17)

    starttime = time.clock()
    print 'Building network...'
    num_classes = 10
    nn = NeuralNetwork(len(train_inputs[0]), [50], num_classes, dummy=True, learning_rate=0.05, dropout=0.5)

    print 'Training network...'
    nn.fit(train_x, train_y, training_horizon=200, verbose=True)
    print 'Time to train: %0.1f'%(time.clock() - starttime)

    starttime = time.clock()
    print '\nTest set results:'
    p_test = nn.predict(test_x, verbose=True)
    print classification_report(test_y, p_test)
    print confusion_matrix(test_y, p_test)
    heatmap(p_test, test_y, 'testheatmap')
    accuracy(p_test, test_y)
    print 'Time to predict: %0.1f'%(time.clock() - starttime)

    starttime = time.clock()
    print '\nTrain set results:'
    p_train = nn.predict(train_x, verbose=True) 
    print classification_report(train_y, p_train)
    print confusion_matrix(train_y, p_train)
    heatmap(p_train, train_y, 'trainheatmap')
    accuracy(p_train, train_y)
    print 'Time to predict: %0.1f'%(time.clock() - starttime)




    # test_inputs = import_csv(TEST_INPUTS_PATH)
    # p = nn.predict(test_inputs)
    # with open(data_files_path+'predictions_1layer_250nodes.csv','w') as f:
    #     f.write('Id,Prediction\n')
    #     for i in range(len(p)):
    #         f.write('%d,%d\n'%(i+1,p[i]))

