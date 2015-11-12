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
import subprocess

data_files_path = 'data_and_scripts/'

TRAIN_INPUTS_PATH = data_files_path+'train_inputs.csv'
TRAIN_OUTPUTS_PATH = data_files_path+'train_outputs.csv'
TEST_INPUTS_PATH = data_files_path+'test_inputs.csv'

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

def command_line_run(args):
    args_dict = {}
    for i in range(1,len(args)):
        if '-' in args[i]:
            args_dict[args[i]] = []
            args_dict[-1] = args_dict[args[i]]
        else:
            args_dict[-1].append(float(args[i]))
    del args_dict[-1]

    num_classes = 10
    random.seed(1917)

    if '-debug' in args_dict:
        train_outputs = import_csv(TRAIN_OUTPUTS_SUBSET_PATH).astype(int)
        train_inputs = import_csv(TRAIN_INPUTS_SUBSET_PATH)
    else:
        train_outputs = import_csv(TRAIN_OUTPUTS_PATH).astype(int)
        train_inputs = import_csv(TRAIN_INPUTS_PATH)
    
    if '-t' in args_dict:
        print len(train_inputs)
        train_inputs = np.array(transform_features(train_inputs))
        print len(train_inputs)

    # Default values.
    hnh = []
    num_features = 300
    dropout = None
    lr = 1.0
    epochs = 50

    if '-f' in args_dict:
        num_features = map(int, args_dict['-f'])[0]

    if '-test' in args_dict:
        test_inputs = import_csv(TEST_INPUTS_PATH)

        if '-t' in args_dict:
            test_inputs = transform_features(test_inputs)
 
        if not num_features == len(train_inputs[0]):
            alll = feature_reduce(np.array(list(train_inputs)+list(test_inputs)), num_features)
            train_inputs = alll[: len(train_inputs)]
            test_inputs = alll[len(train_inputs) :]

    if '-validate' in args_dict:
        validation_size = (4 * len(train_inputs)) / 5

        # Randomize the train and validation set.
        rand_idxs = random.sample(range(0, len(train_inputs)), len(train_inputs))

        test_inputs = train_inputs[rand_idxs[validation_size : ]]
        test_outputs = train_outputs[rand_idxs[validation_size : ]]
        train_inputs = train_inputs[rand_idxs[0 : validation_size]]
        train_outputs = train_outputs[rand_idxs[0 : validation_size]]

        # We have to reduce the features all at the same time because it is unsupervised learning and
        # we want the same features to be picked by PCA for both of the train and test sets.
        if not num_features == len(train_inputs[0]):
            alll = feature_reduce(np.array(list(train_inputs)+list(test_inputs)), num_features)
            train_inputs = alll[: len(train_inputs)]
            test_inputs = alll[len(train_inputs) :]

    if '-hn' in args_dict:
        hnh = map(int, args_dict['-hn'])

    if '-d' in args_dict:
        if not (0.0 <= args_dict['-d'][0] <= 1.0):
            print 'Please input a dropout rate between 0 and 1!'
            exit(0)
        dropout = args_dict['-d'][0]

    if '-lr' in args_dict:
        lr = args_dict['-lr'][0]

    if '-e' in args_dict:
        epochs = int(args_dict['-e'][0])

    nn = NeuralNetwork(len(train_inputs[0]), hnh, num_classes, learning_rate=lr, dropout=dropout)
    nn.fit(train_inputs, train_outputs, training_horizon=epochs, verbose=True)
    p = nn.predict(test_inputs)

    fname = data_files_path+'predictions_with_%depochs_%dfeatures_%0.2flf'%(epochs,num_features,lr)
    if '-test' in args_dict:
        with open(fname+'.csv','w') as f:
            f.write('Id,Prediction\n')
            for i in range(len(p)):
                f.write('%d,%d\n'%(i+1,p[i]))
    else:
        print accuracy(p, test_outputs)
        if '-record' in args_dict:
            heatmap(p, test_outputs, fname)


if __name__ == '__main__':
    subprocess.call('clear')

    if len(argv) == 1:
        print '\n\nPlease call this program as follows:\n$ python classification.py options\n'
        time.sleep(1)
        print 'Where the following options are available:\n'
        print '\t-validate: runs the default neural network over the \n\t\tvalidation set and prints the results. \n\t\t40000 trainings samples, 10000 validation\n'
        print '\t-test: runs the default neural network over the kaggle \n\t\ttest set and writes the results file\n'
        print '\t-hn X1 X2 X3 Xk: sets the number of hidden nodes for \n\t\teach hidden layer, i.e. two values after -hn means two \n\t\thidden layers. Default is empty\n'
        print '\t-d X: sets the dropout value, must be between 0 and 1\n'
        print '\t-f X: sets number of features to reduce to, default is 300\n'
        print '\t-lr X: sets the learning rate, default is 1.0\n'
        print '\t-e X: number of epochs. Default is 50\n'
        print '\t-t: transform features using Caspers feature transformation\n'
        print '\t-record: record the results in a heatmap (if -validation)\n'
        exit(0)

    else:
        command_line_run(argv)
        exit(0)

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
