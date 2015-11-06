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
    print '%d correct out of %d. %0.3f accuracy.'%(right, len(predictions), float(right)/len(predictions))

def cross_validate(X, Y, n, k=5, epochs=50, lr=1.0, lam=1.0, nhn=50, weights=(-1.0,1.0)):
    kf = KFold(n, n_folds=k, shuffle=True, random_state=1917)
    avgs_train,avgs_test = [],[]
    for train_idx,test_idx in kf:
        x_train,x_test = X[train_idx],X[test_idx]
        y_train,y_test = Y[train_idx],Y[test_idx]

        cv_nn = NeuralNetwork(len(x_train[0]), [nhn], 10, learning_rate=lr, regularization=lam, weight_range=weights)
        cv_nn.fit(x_train, y_train, training_horizon=epochs)

        avgs_train.append(accuracy(cv_nn.predict(x_train), y_train))
        avgs_test.append(accuracy(cv_nn.predict(x_test), y_test))

    return np.mean(avgs_test)#, np.mean(avgs_train)

def optimize_hyper_params(X, Y, features=500):
    f_name = 'hyper_parameters_results.txt'

    with open(f_name,'w') as fr:
        X = feature_reduce(X, features)
        logs = [i for i in range(-3,4)]
        accuracies_logistic = {logs[v]: 0.0 for v in logs}

        fr.write('Results from logistic weight range cross validation:\n')
        for key in accuracies_logistic:
            accuracies_logistic[key] = cross_validate(X, Y, 5000, weights=(-1.0*(10.0**key),10.0**key))
            fr.write('\tweights range = (%s,%s): %s\n'%((10.0**key)*-1.0,10.0**key,accuracies_logistic[key]))

        best = max(accuracies_logistic,accuracies_logistic.get)
        iterative_test = np.linspace(10.0**(best-0.5), 10.0**(best+0.5), num=20)
        accuracies_iterative = {iterative_test[i] : 0.0 for i in range(len(iterative_test))}
        for key in accuracies_iterative:
            accuracies_iterative[key] = cross_validate(X, Y, 5000, weights=(-1.0*key, key))
            fr.write('\tweights range = (%s,%s): %s\n'%((-1.0*key), key, accuracies_iterative[key]))

        best_weights = max()




if __name__ == '__main__':
    # Below provides a good test to show that it works succesfully based on the example given
    # on this paper (pg. 20): https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf
    #
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

    starttime = time.clock()
    train_outputs = import_csv(TRAIN_OUTPUTS_PATH).astype(int)
    train_inputs = (import_csv(TRAIN_INPUTS_PATH))
    test_inputs = (import_csv(TEST_INPUTS_PATH))

    # train_inputs = import_csv(TRAIN_INPUT_SUBSET_PATH)
    print 'Time to import: %0.1f'%(time.clock() - starttime)

    alll = feature_reduce(np.array(list(train_inputs)+list(test_inputs)), 500)
    train = alll[:len(train_inputs)]
    test = alll[len(train_inputs):]
    # train_x = train_inputs[0:4500]
    # train_y = train_outputs[0:4500]
    # test_x = train_inputs[4501:]
    # test_y = train_outputs[4501:]
    
    starttime = time.clock()
    print 'Building network...'
    num_classes = 10
    nn = NeuralNetwork(len(train[0]), [25], num_classes, dummy=True, learning_rate=1.0, weight_range=(0,1))

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
