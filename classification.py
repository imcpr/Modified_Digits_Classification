# coding=utf-8
# Authors:
#   Kian Kenyon-Dean.
#   Yann Long
#
# Coding began October 31st, 2015

import numpy as np
from import_data import import_csv
from neural_net_efficient import NeuralNetwork
from sklearn.decomposition import PCA
from features import transform_features
from sklearn.svm import SVC,LinearSVC
from sklearn.cross_validation import KFold
from graphic import heatmap
from sys import argv
import numpy as np
import time
import random
import pyprind
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
    print '\t%d correct out of %d. %0.3f accuracy.'%(right, len(predictions), float(right)/len(predictions))
    return float(right)/len(predictions)

def cross_validate(X, Y, n, k=5, epochs=50, lr=1.0, lam=1.0, nhn=50, dropout=None, v=False):
    kf = KFold(n, n_folds=k, shuffle=True, random_state=1917)
    avgs_train,avgs_test = [],[]
    num_outputs = 10

    for train_idx,test_idx in kf:
        x_train,x_test = X[train_idx],X[test_idx]
        y_train,y_test = Y[train_idx],Y[test_idx]

        cv_nn = NeuralNetwork(len(x_train[0]), [nhn], num_outputs, learning_rate=lr, dropout=dropout, maxnorm=None)
        cv_nn.fit(x_train, y_train, training_horizon=epochs, verbose=v)

        avgs_train.append(accuracy(cv_nn.predict(x_train), y_train))
        avgs_test.append(accuracy(cv_nn.predict(x_test), y_test))

    return np.mean(avgs_test), np.mean(avgs_train)

def optimize_hyper_params(X, Y, features=0, hidden_nodes=0, dropout=0.0, lr=0.0):
    if not features:
        with open('number_of_features_optimization_incrementing_epochs.csv','w') as f:
            my_features = np.linspace(100,2300, num=45).astype(int)
            accs = {num : (0.0,0.0) for num in my_features}
            f.write("Results from cross validating for finding the optimal number of features:\n")
            f.write("features,epochs,test_accuracy,train_accuracy\n")
            
            for key in my_features:
                epochs = 10 + (key/50 - 1)
                print "Testing %d features, %d epochs..."%(key,epochs),
                new_x = feature_reduce(X, key)
                accs[key] = cross_validate(new_x, Y, 5000, epochs=epochs, nhn=50)
                f.write('%d,%d,%s,%s\n'%(key,epochs,accs[key][0],accs[key][1]))
        
            features = max(accs, key=accs.get)

    #Next we will optimize the number of hidden nodes.    
    if not hidden_nodes:
        with open('number_of_hidden_nodes_optimization_incrementing_epochs.csv','w') as f:
            reduced_x = feature_reduce(X, features)

            num_nodes = [1] + [i*10 for i in range(1,101)]
            num_nodes.remove(300)
            num_nodes.append(299) 
            num_nodes.sort()

            accs = {num_nodes[i] : (0.0,0.0) for i in range(len(num_nodes))}
            f.write("Results from cross validating for finding the optimal number of hidden nodes:\n")
            f.write("num_nodes,test_accuracy,train_accuracy\n")
            for key in num_nodes:
                accs[key] = cross_validate(reduced_x, Y, 5000, nhn=key, epochs=10 + (key/10), lr=1.0, dropout=None)
                f.write('%d,%s,%s\n'%(key,accs[key][0],accs[key][1]))

            hidden_nodes = max(accs, key=accs.get)

    #Next we optimize the dropout value.
    if not dropout:
        with open('dropout_optimization_100epochs.csv','w') as f:
            reduced_x = feature_reduce(X, features)
            dropouts = np.linspace(0.05, 0.95, 19)
            accs = {val : 0.0 for val in dropouts}
            
            f.write("Results from cross validating for finding the optimal value of the dropout constant:\n")
            f.write("dropout_value,test_accuracy,train_accuracy\n")
            for key in dropouts:
                print 'Testing dropout = %s...'%key
                accs[key] = cross_validate(reduced_x, Y, 5000, nhn=hidden_nodes, epochs=100, lr=1.0, dropout=key)
                f.write('%s,%s,%s\n'%(key,accs[key][0],accs[key][1]))

            dropout = max(accs, key=accs.get)

    if not lr:
        with open('learning_rate_validation_results.csv','w') as f:
            reduced_x = feature_reduce(X, features)
            # learning_rates = [10**i for i in range(-3,4)]
            learning_rates = np.linspace(0.1, 2.0, 39)

            accs_log = {val : 0.0 for val in learning_rates}
            f.write("Results from cross validating for finding the optimal value of the learning rate:\n")
            f.write("learning_rate,test_accuracy,train_accuracy\n")
            for key in learning_rates:
                print '\nTesting learning_rate = %s...'%key
                accs_log[key] = cross_validate(reduced_x, Y, 5000, nhn=hidden_nodes, epochs=50, lr=key, dropout=dropout)
                f.write('%s,%s,%s\n'%(key,accs_log[key][0],accs_log[key][1]))

    return np.mean(avgs_test)#, np.mean(avgs_train)

if __name__ == '__main__':
    num_classes = 10

    if len(argv) == 1:
        print 'Please call this program as follows:\n$ python classification.py options'
        print 'Where the following options are available:'
        print '\t-validate: runs the default neural network over the validation set and prints the results'
        print '\t-test: runs the default neural network over the kaggle test set and writes the results file'
        print '\t-hn X1 X2 X3 Xk: sets the number of hidden nodes for each hidden layer, i.e. two values after -hn \
                    means two hidden layers. Default is empty'
        print '\t-d X: sets the dropout value, must be between 0 and 1'
        print '\t-f X: sets number of features to reduce to, default is 300'
        print '\t-lr X: sets the learning rate, default is 1.0'
        print '\t-t: transform features using Caspers feature transformation.'

    starttime = time.clock()
    train_outputs = import_csv(TRAIN_OUTPUTS_PATH).astype(int)
    train_inputs = import_csv(TRAIN_INPUTS_PATH)
    # kaggle_test_inputs = import_csv(TEST_INPUTS_PATH)

    random.seed(1917)
    validation_size = 40000

    train_inputs = feature_reduce(train_inputs, 300)

    rand_idxs = random.sample(range(len(train_inputs)), 50000)

    train_inputs = train_inputs[rand_idxs[0 : validation_size]]
    train_y = train_outputs[rand_idxs[0 : validation_size]]
    test_inputs = train_inputs[rand_idxs[validation_size :]]
    test_y = train_outputs[rand_idxs[validation_size :]]

    # We have to reduce the features all at the same time because it is unsupervised learning and
    # we want the same features to be picked by PCA for both of the train and test sets.
    alll = feature_reduce(np.array(list(train_inputs)+list(test_inputs)), 500)
    train = alll[: len(train_inputs)]
    test = alll[len(train_inputs) :]

    starttime = time.clock()
    print 'Building network...'
    num_classes = 10
    nn = NeuralNetwork(len(train_inputs[0]), [50], num_classes, dummy=True, learning_rate=0.05, dropout=0.5)

    print 'Training network...'
    nn.fit(train_x, train_y, training_horizon=200, verbose=True)
    print 'Time to train: %0.1f'%(time.clock() - starttime)

    #Validation set results
    nn = NeuralNetwork(len(train_x[0]), [50], num_classes, dummy=True, learning_rate=1.0, dropout=0.1)
    nn.fit(train_x, train_y, training_horizon=100, verbose=True)
    p = nn.predict(test_x)
    print accuracy(p, test_y)
    # heatmap(p, test_y, 'ff_nn_results_50hiddennodes_lr1_dropout1')

    nn = NeuralNetwork(len(train_x[0]), [50, 15], num_classes, dummy=True, learning_rate=1.0, dropout=0.1)
    nn.fit(train_x, train_y, training_horizon=100, verbose=True)
    p = nn.predict(test_x)
    print accuracy(p, test_y)
    # heatmap(p, test_y, 'ff_nn_results_50hiddennodes_15hiddennodes2_lr1_dropout01')


    #For testing on the main kaggle test file.
    p = nn.predict(test)
    with open(data_files_path+'predictions_500f_umodified_1layer_25nodes_th100.csv','w') as f:
        f.write('Id,Prediction\n')
        for i in range(len(p)):
            f.write('%d,%d\n'%(i+1,p[i]))

    """
    print '------------'
    print 'SVM:'
    s1 = LinearSVC()
    print 'Training linear svm...'
    s1.fit(train_x, train_y)
    print 'Test results...'
    accuracy(s1.predict(test_x),test_y)

    print 'Train results...'
    accuracy(s1.predict(train_x),train_y)

    s1 = SVC()
    print '\n\nTraining svm...'
    s1.fit(train_x, train_y)
    print 'Test results...'
    accuracy(s1.predict(test_x),test_y)

    print 'Train results...'
    accuracy(s1.predict(train_x),train_y)
    """
