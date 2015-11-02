# coding=utf-8
# Authors:
#   Kian Kenyon-Dean.
#
# Coding began October 31st, 2015

from import_data import import_csv
from neural_net import NeuralNetwork
from sklearn.decomposition import PCA
import time

data_files_path = '/Users/kian/documents/COMP598/Assignment3/data_and_scripts/'

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

if __name__ == '__main__':
    # Below provides a good test to show that it works succesfully based on the example given
    # on this paper (pg. 20): https://www4.rgu.ac.uk/files/chapter3%20-%20bp.pdf

    # nn = NeuralNetwork(2, [2], 1, dummy=False)
    # nn.initialize_weights([[0.1, 0.4, 0.8, 0.6],[0.3, 0.9]])
    # nn.fit(np.array([[0.35, 0.9]]), np.array([0.5]), training_horizon=1)

    starttime = time.clock()
    train_outputs = import_csv(TRAIN_OUTPUT_SUBSET_PATH).astype(int)
    train_inputs = import_csv(TRAIN_INPUT_SUBSET_PATH)
    print 'Time to import: %0.1f'%(time.clock() - starttime)

    train_inputs = feature_reduce(train_inputs, 500)

    starttime = time.clock()
    print 'Building network...'
    num_classes = 10
    nn = NeuralNetwork(len(train_inputs[0]), [], num_classes, dummy=True)
    nn.randomly_initalize_weights((-1.0,1.0))

    train_x = train_inputs[0:1000]
    train_y = train_outputs[0:1000]
    test_x = train_inputs[4501:4601]
    test_y = train_outputs[4501:4601]

    print 'Training network...'
    nn.fit(train_x, train_y, training_horizon=15, verbose=True, print_mod=500)
    print 'Time to train: %0.1f'%(time.clock() - starttime)

    starttime = time.clock()
    print '\nTest set results:'
    p_test = nn.predict(test_x, verbose=True, print_mod=500)
    print p_test
    accuracy(p_test, test_y)
    print 'Time to predict: %0.1f'%(time.clock() - starttime)

    starttime = time.clock()
    print '\nTrain set results:'
    p_train = nn.predict(train_x, verbose=True, print_mod=500)
    print p_train
    accuracy(p_train, train_y)
    print 'Time to predict: %0.1f'%(time.clock() - starttime)




    # test_inputs = import_csv(TEST_INPUTS_PATH)
    # p = nn.predict(test_inputs)
    # with open(data_files_path+'predictions_1layer_250nodes.csv','w') as f:
    #     f.write('Id,Prediction\n')
    #     for i in range(len(p)):
    #         f.write('%d,%d\n'%(i+1,p[i]))

