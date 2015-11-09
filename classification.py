# coding=utf-8
# Authors:
#   Kian Kenyon-Dean.
#
# Coding began October 31st, 2015

from import_data import import_csv
from neural_net_efficient import NeuralNetwork
from sklearn.decomposition import PCA
# from features import transform_features
from sklearn.svm import SVC,LinearSVC
from sklearn.cross_validation import KFold
import numpy as np
import time
import random
import pyprind

data_files_path = 'data_and_scripts/'

TRAIN_INPUTS_PATH = data_files_path+'train_inputs.csv'
TRAIN_OUTPUTS_PATH = data_files_path+'train_outputs.csv'
TEST_INPUTS_PATH = data_files_path+'test_inputs.csv'

TRAIN_INPUT_SUBSET_PATH = data_files_path+'train_inputs_subset.csv'
TRAIN_OUTPUT_SUBSET_PATH = data_files_path+'train_outputs_subset.csv'

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

def optimize_hyper_params(X, Y):
    f_name = 'number_of_features_optimization_incrementing_epochs.csv'
    # with open(f_name,'w') as f:

        #First we find optimal number of features.
        # my_features = np.linspace(100,2300, num=45).astype(int)
        # accs = {num : (0.0,0.0) for num in my_features}
        # f.write("Results from cross validating for finding the optimal number of features:\n")
        # f.write("features,epochs,test_accuracy,train_accuracy\n")
        
        # for key in my_features:
        #     epochs = 10 + (key/50 - 1)

        #     print "Testing %d features, %d epochs..."%(key,epochs),
        #     new_x = feature_reduce(X, key)
        #     accs[key] = cross_validate(new_x, Y, 5000, epochs=epochs, nhn=50)
        #     f.write('%d,%d,%s,%s\n'%(key,epochs,accs[key][0],accs[key][1]))
        
    #Next we will optimize the number of hidden nodes.    
    # with open('number_of_hidden_nodes_optimization_incrementing_epochs.csv','w') as f:
        
    #     reduced_x = feature_reduce(X, 300)
    #     num_nodes = [1] + [i*10 for i in range(1,101)]
    #     num_nodes.remove(300)
    #     num_nodes.append(299)
    #     num_nodes.sort()
    #     accs = {num_nodes[i] : (0.0,0.0) for i in range(len(num_nodes))}
    #     f.write("Results from cross validating for finding the optimal number of hidden nodes:\n")
    #     f.write("num_nodes,test_accuracy,train_accuracy\n")
    #     for key in num_nodes:
    #         accs[key] = cross_validate(reduced_x, Y, 5000, nhn=key, epochs=10 + (key/10), lr=1.0, dropout=None)
    #         f.write('%d,%s,%s\n'%(key,accs[key][0],accs[key][1]))

    # exit(0)

    #Next we optimize the dropout value.
    with open('dropout_optimization_100epochs.csv','w') as f:
        reduced_x = feature_reduce(X, 300)
        dropouts = np.linspace(0.05, 0.95, 19)
        accs = {val : 0.0 for val in dropouts}
        f.write("Results from cross validating for finding the optimal value of the dropout constant:\n")
        f.write("dropout_value,test_accuracy,train_accuracy\n")
        for key in dropouts:
            print 'Testing dropout = %s...'%key
            accs[key] = cross_validate(reduced_x, Y, 5000, nhn=50, epochs=100, lr=1.0, dropout=key)
            f.write('%s,%s,%s\n'%(key,accs[key][0],accs[key][1]))

    exit(0)

    with open('learning_rate_optimization.csv','w') as f:
        reduced_x = feature_reduce(X, 300)
        # dropouts = np.linspace(0.05, 0.95, 19)
        # learning_rates = [10**i for i in range(-3,4)]
        learning_rates = np.linspace(0.1, 2.0, 39)

        accs_log = {val : 0.0 for val in learning_rates}
        f.write("Results from cross validating for finding the optimal value of the learning rate:\n")
        f.write("learning_rate,test_accuracy,train_accuracy\n")
        for key in learning_rates:
            print '\nTesting learning_rate = %s...'%key
            accs_log[key] = cross_validate(reduced_x, Y, 5000, nhn=50, epochs=50, lr=key, dropout=0.1)
            f.write('%s,%s,%s\n'%(key,accs_log[key][0],accs_log[key][1]))

    exit(0)

        #The code below was for selecting the best weight initialization but it is unnecessary..
        # logs = [i for i in range(-3,4)]
        # accuracies_logistic = {logs[v]: 0.0 for v in logs}
        # f.write('Results from logistic weight range cross validation:\n')
        # for key in accuracies_logistic:
        #     accuracies_logistic[key] = cross_validate(X, Y, 5000, weights=(-1.0*(10.0**key),10.0**key))
        #     f.write('\tweights range = (%s,%s): %s\n'%((10.0**key)*-1.0,10.0**key,accuracies_logistic[key]))
        #
        # best = max(accuracies_logistic, accuracies_logistic.get)
        # iterative_test = np.linspace(10.0**(best-0.5), 10.0**(best+0.5), num=20)
        # accuracies_iterative = {iterative_test[i] : 0.0 for i in range(len(iterative_test))}
        # for key in accuracies_iterative:
        #     accuracies_iterative[key] = cross_validate(X, Y, 5000, weights=(-1.0*key, key))
        #     f.write('\tweights range = (%s,%s): %s\n'%((-1.0*key), key, accuracies_iterative[key]))
        # best_weights = max(accuracies_iterative, accuracies_iterative.get)

if __name__ == '__main__':
    # Below provides a good test to show that it works succesfully based on the example given
    # on this paper (pg. 20): https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
    #
    """
    nn = NeuralNetwork(2, [3], 2, dummy=False, weight_range=(-3,3))
    nn.fit(np.array([[0.35, 0.9]]), np.array([1]), training_horizon=1)

    x = []
    b = [1.0,0.0]
    for ii in b:
        for jj in b:
            for kk in b:
                x.append([ii,jj,kk])
    y = []
    for l in x:
        y.append(1.0 if (l[0] or l[1]) and l[2] else 0.0)
    # for i in x: print i
    # print y

    nn = NeuralNetwork(3,[5],2,weight_range=(0,1),learning_rate=1.0,dummy=True)
    # print nn.weights
    # # nn.feedforward(x[0])
    # # nn.backpropagate(target_value=1.0)
    # print nn.weights
    nn.fit(x,y, training_horizon=1000, verbose=True)
    p = nn.predict(x)
    print '-------------------------------'
    print p,y
    accuracy(p,y)

    exit(0)
    """

    starttime = time.clock()
    train_outputs = import_csv(TRAIN_OUTPUTS_PATH).astype(int)
    train_inputs = (import_csv(TRAIN_INPUTS_PATH))
    # test_inputs = (import_csv(TEST_INPUTS_PATH))
    # train_inputs = import_csv(TRAIN_INPUT_SUBSET_PATH)
    print 'Time to import: %0.1f'%(time.clock() - starttime)

    random.seed(1917)
    rand_idxs = random.sample(range(len(train_inputs)), 5000)
    train_inputs = train_inputs[rand_idxs]
    train_outputs = train_outputs[rand_idxs]

    reduced_x = feature_reduce(train_inputs, 300)
    cross_validate(reduced_x, train_outputs, 5000, nhn=50, epochs=1000, lr=0.5, dropout=0.1, v=True)
    exit(0)
    # optimize_hyper_params(train_inputs, train_outputs)


    # alll = feature_reduce(np.array(list(train_inputs)+list(test_inputs)), 500)
    # train = alll[:len(train_inputs)]
    # test = alll[len(train_inputs):]
    # train_x = train_inputs[0:4500]
    # train_y = train_outputs[0:4500]
    # test_x = train_inputs[4501:]
    # test_y = train_outputs[4501:]

    starttime = time.clock()
    print 'Building network...'
    num_classes = 10
    nn = NeuralNetwork(len(train[0]), [25], num_classes, dummy=True, learning_rate=1.0)

    print 'Training network...'
    nn.fit(train, train_outputs, training_horizon=100, verbose=True)
    print 'Time to train: %0.1f'%(time.clock() - starttime)

    p = nn.predict(test)
    with open(data_files_path+'predictions_500f_umodified_1layer_25nodes_th100.csv','w') as f:
        f.write('Id,Prediction\n')
        for i in range(len(p)):
            f.write('%d,%d\n'%(i+1,p[i]))

    # print '\nTest set results:'
    # p_test = nn.predict(test_x, verbose=True)
    # print p_test
    # accuracy(p_test, test_y)
    #
    # print '\nTrain set results:'
    # p_train = nn.predict(train_x, verbose=True)
    # print p_train
    # accuracy(p_train, train_y)
    # nn.log()

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
