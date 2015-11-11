# -*- coding: utf-8 -*-
"""
Created on Mon Nov 09 09:05:32 2015

@author: YannLong
"""

from sklearn.svm import LinearSVC 
from import_data import import_csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from graphic import heatmap
from classification import accuracy  

if __name__ == '__main__':
    
    data_files_path = 'data_and_scripts/'

    TRAIN_INPUTS_PATH = data_files_path+'train_inputs324.csv'
    TRAIN_OUTPUTS_PATH = data_files_path+'train_outputs.csv'
    TEST_INPUTS_PATH = data_files_path+'test_inputs324.csv'

    TRAIN_INPUTS_SUBSET_PATH = data_files_path+'train_inputs_subset.csv'
    TRAIN_OUTPUTS_SUBSET_PATH = data_files_path+'train_outputs_subset.csv'

    train_outputs = import_csv(TRAIN_OUTPUTS_PATH).astype(int)
    train_inputs = import_csv(TRAIN_INPUTS_PATH)

    print np.shape(train_outputs)
    print np.shape(train_inputs)   

    
    #randomly split the data into a train set and a validation set
    train_x, test_x, train_y, test_y = train_test_split(train_inputs, train_outputs, test_size=0.2, random_state=17)

    #use the training set to find best learning rate c
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
    clf = GridSearchCV(LinearSVC(penalty='l2'), param_grid, n_jobs=-1)
    clf.fit(train_x,train_y)
    #best parameter
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
      
    #grid score
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
    print()
   
    #validation
    print("Performance of optimal learner on validation set")
    print()
    expected, predicted = test_y, clf.predict(test_x)     
    print(classification_report(expected, predicted))
    print(confusion_matrix(expected, predicted))
    print()
    accuracy(predicted,expected)
    heatmap(predicted,expected,'LinSVM/testAccuracy')

    #training accuracy
    print("Performance of optimal learner on training set")
    print()
    expected, predicted = train_y, clf.predict(train_x)     
    print(classification_report(expected, predicted))
    print(confusion_matrix(expected, predicted))
    print()
    accuracy(predicted,expected)
    heatmap(predicted,expected,'LinSVM/trainAccuracy')
 
    #training on the whole dataset
    print("fitting the best estimator on the complete training set")
    learner=clf.best_estimator_
    learner.fit(train_inputs,train_outputs)
    print()    
    
    #read in test data
    print("reading in the test data")
    test_inputs = import_csv(TEST_INPUTS_PATH)
    print()

    #make a prediction
    print("making a prediction")
    prediction = learner.predict(test_inputs)
    print()

    #printing the prediction
    print("saving prediction")    
    N = len(prediction)
    ID = np.arange(N).reshape(N,1)
    prediction = np.concatenate((ID,prediction.reshape(N,1)), axis=1)
    print(prediction)
    np.savetxt('LinSVM/SVM_predictions.csv',prediction,fmt='%i', delimiter=',', newline='\n', header='Id,Prediction',comments='')
    print("Done")

