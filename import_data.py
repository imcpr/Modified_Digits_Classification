 # coding=utf-8
# Authors:
#   Kian Kenyon-Dean.
#
# Coding began October 31st, 2015

""" Simple data importing from the CSV files. """

import numpy as np
import csv
from matplotlib import pyplot as pl

def import_csv(file_path):
    """ Here we are taking in a csv file and returning it as a numpy array, but we build it with python lists. """
    print 'Loading file: %s'%file_path

    dataset = []
    with open(file_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # Skip the header row.
        for sample in reader:
            row_values = []
            for value in sample[1:]: # Skip the first index - the ID column.
                row_values.append(float(value))
            if len(row_values) == 1:
                dataset.append(row_values[0])
            else:
                dataset.append(np.array(row_values))

    return np.array(dataset)

def import_results_csv(file_path):
    dataset = []
    with open(file_path, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None) # Skip the header row.
        for sample in reader:
            row_values = []
            for value in sample:
                try:
                    row_values.append(float(value))
                except ValueError:
                    row_values.append(value)
            if len(row_values) == 1:
                dataset.append(row_values[0])
            else:
                dataset.append(np.array(row_values))
    
    return np.array(dataset)

def get_best_k(datax, datay, k=5):
    best = []
    datax = list(datax)
    datay = list(datay)
    for _ in range(k):
        tmp = (0,0)
        idx = -1
        for i in range(len(datay)):
            if datay[i] > tmp[0]:
                tmp = (datay[i],datax[i])
                idx = i
        best.append(tmp)
        datax = datax[0:idx] + datax[idx+1:]
        datay = datay[0:idx] + datay[idx+1:]
    return best

def analysis(hidden_nodes = True): # for dropouts and hidden nodes.
    if hidden_nodes:
        name = 'number_of_hidden_nodes_optimization_incrementing_epochs.csv'
    else:
        name = 'dropout_optimization.csv'

    data = import_results_csv(name)
    # print data
    var_names = data[0]
    data = data[1:]
    x = data[:,0]
    test_acc = data[:,1]
    train_acc = data[:,2]

    pl.plot(x, test_acc, 'ro-', label='Test accuracy')
    pl.plot(x, train_acc, 'bo-', label='Train accuracy')    

    if hidden_nodes:    
        pl.axis([-10, 510, 0.05, 0.60])    
        pl.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    else:
        pl.axis([0.0, 1.0, 0.05, 0.60])
        pl.xticks(np.linspace(0.05, 0.95, 10))
        print get_best_k(x, test_acc, k=10)


    # print np.array([x[i] for i in range(len(x)) if i%2==0])
    pl.yticks(np.linspace(0.1,0.6,11))

    pl.legend(bbox_to_anchor=(1,0.6))

    pl.show()

    exit(0)

def learning_rate_analysis(log=True):
    if log:
        data = import_results_csv('learning_rate_optimization_logarithmic.csv')
    else:
        data = import_results_csv('learning_rate_optimization.csv')

    var_names = data[0]
    data = data[1:]
    x = data[:,0]
    test_acc = data[:,1]
    train_acc = data[:,2]

    if log:
        pl.semilogx(x, test_acc, 'ro-', label='Test accuracy')
        pl.semilogx(x, train_acc, 'bo-', label='Train accuracy')
    else:
        pl.plot(x, test_acc, 'ro-', label='Test accuracy')
        pl.plot(x, train_acc, 'bo-', label='Train accuracy')

    if log:
        pl.axis([-100, 1010, 0, 0.5])
    else:
        pl.axis([0.0, 2.1, 0.0, 0.5])
    pl.yticks(np.linspace(0.05,0.5,10))

    print get_best_k(x, test_acc, k=10)

    if log:
        pl.legend(bbox_to_anchor=(1,0.6))
    else:
        pl.legend(bbox_to_anchor=(1,0.2))

    pl.show()

    exit(0)

if __name__ == '__main__':
    # analysis(hidden_nodes=False)
    learning_rate_analysis(log=False)

    # get = import_results_csv('number_of_features_optimization.csv')
    # first_test = get[1][1]
    # first_train = get[1][2]
    # del get

    name = 'number_of_features_optimization_incrementing_epochs(v1).csv'
    epochs = 1 if 'epochs' in name else 0

    features_graph = import_results_csv(name)
    var_names = features_graph[0]
    features_graph = features_graph[1:]

    x = features_graph[:,0]
    test_acc = features_graph[:,1+epochs]
    train_acc = features_graph[:,2+epochs]

    pl.plot(x, test_acc, 'ro-', label='Test accuracy')
    pl.plot(x, train_acc, 'bo-', label='Train accuracy')

    best_line_test = get_best_k(x, test_acc, k=1)[0][0]
    best_line_train = get_best_k(x, train_acc, k=1)[0][0]
    # pl.plot(x, [best_line_test]*len(x), 'r')
    # pl.plot(x, [best_line_train]*len(x), 'b')

    # print train_acc
    # print test_acc
    # ymax = get_best_k(x, train_acc, k=1)[0][0]

    pl.axis([50,2350,0.05,0.6]) # change first val to -50 for prior graph
    pl.yticks(np.linspace(0.1,0.7,13)) # 0.1,0.6,11 for no incrementing epochs.

    pl.xticks(np.linspace(0,2200,12))
    pl.legend(bbox_to_anchor=(1,0.3))
    # pl.xlabel('Number of features after PCA')
    # pl.ylabel('Test/train accuracy')
    # pl.title('Feature choice accuracies resulting from 5-fold cross validation')
    pl.show()

    # test = get_best_k(x,test_acc, k=10)
    # trains = get_best_k(x,train_acc)
    # for t in test:
    #     print t
    # for t in trains:
    #     print t

    pass











