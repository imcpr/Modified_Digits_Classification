# coding=utf-8
# Authors:
#   Casper Liu
#   Yann Long
#
# Coding began Novembre 2nd, 2015

import csv
import math
import numpy as np
import pyprind
from scipy.spatial import distance
from scipy.ndimage import interpolation
from math import atan, pi, ceil
from scipy import misc
from skimage import transform
from skimage.morphology import disk
from skimage import exposure
from matplotlib import pyplot as plt
from matplotlib.image import imsave
from sklearn import svm, metrics
from skimage.morphology import reconstruction
from skimage.transform import resize

# calculates the moment of the image
def M(image, p, q):
    w, h = image.shape
    acc = 0.0
    for x in range(0, w):
        for y in range(0, h):
            acc += (x**p)*(y**q)*image[x,y]
    return acc

# using image moments, tries to align the image by its principal axis
def deskew(image):
    w, h = image.shape
    x = M(image, 1, 0) / M(image, 0, 0)
    y = M(image, 0, 1) / M(image, 0, 0)
    mu02 = M(image, 0, 2) - y * M(image, 0, 1)
    mu20 = M(image, 2, 0) - x * M(image, 1, 0)
    mu11 = M(image, 1, 1) - x * M(image, 0, 1)
    lambda1 = 0.5*( mu20 + mu02 ) + 0.5*( mu20**2 + mu02**2 - 2*mu20*mu02 + 4*mu11**2 )**0.5
    lambda2 = 0.5*( mu20 + mu02 ) - 0.5*( mu20**2 + mu02**2 - 2*mu20*mu02 + 4*mu11**2 )**0.5 
    lambda_m = max(lambda1, lambda2)
    # Convert from radians to degrees
    angle =  ceil(atan((lambda_m - mu20)/mu11)*18000/pi)/100
    #print angle
    center = tuple(map(int, (x, y)))
    #print center
    tf_rotate = transform.SimilarityTransform(rotation=np.deg2rad(angle))
    tf_shift = transform.SimilarityTransform(translation=[-center[0], -center[1]])
    tf_shift_inv = transform.SimilarityTransform(translation=[center[0], center[1]])
    image_rotated = transform.warp(image, (tf_shift + (tf_rotate + tf_shift_inv)).inverse)
    return image_rotated

# converts a flattened image array to a 2d matrix
def a2m(arr):
    m = np.array(arr)
    dim = int(math.sqrt(len(arr)))
    m.resize(dim, dim)
    return m

def show_image(image):
    plt.imshow(image, cmap="Greys_r")
    plt.show()
    
# def save_image(image):
    

# helper to load training set
def get_train(index=50000, start=0, transform=True):
    train_inputs = []
    train_outputs = []
    if transform:
        filename = 'data_and_scripts/transformed_train_inputs.csv'
    else:
        filename = 'data_and_scripts/train_inputs.csv'
    with open(filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        if not transform:
            next(reader, None)  # skip the header
        i = 0
        bar = pyprind.ProgBar(index, title="Loading training data from csv")
        for train_input in reader: 
            if i >= index:
                break
            i += 1
            train_input_no_id = []
            # print len(train_input)
            if i >= start:
                for pixel in train_input[1:2305]: # Start at index 1 to skip the Id
                    train_input_no_id.append(float(pixel))
                train_inputs.append(train_input_no_id) 
            bar.update()
    with open('data_and_scripts/train_outputs.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the header
        i = 0
        for train_output in reader:  
            if i >= index:
                break
            i += 1
            if i >= start:
                train_output_no_id =  int(train_output[1])
                train_outputs.append(train_output_no_id)
    return train_inputs, train_outputs


def get_test(index=20000):
    train_inputs = []
    train_outputs = []
    with open('data_and_scripts/test_inputs.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the header
        i = 0
        bar = pyprind.ProgBar(index, title="Loading training data from csv")
        for train_input in reader: 
            if i >= index:
                break
            i += 1
            train_input_no_id = []
            for pixel in train_input[1:]: # Start at index 1 to skip the Id
                train_input_no_id.append(float(pixel))
            train_inputs.append(train_input_no_id) 
            bar.update()
    print i
    return train_inputs

# use dilations
def get_dilated(image):
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    dilated = reconstruction(seed, mask, method='dilation')
    return dilated

# creates a filter matrix for size col, rows with gradually dimming corners
def get_circle_filter(cols, rows):
    f = np.zeros(rows*cols).reshape(rows,cols)
    cx = rows/2
    cy = cols/2
    max_d = math.sqrt(cx**2+cy**2)+10
    for x in range(cols):
        for y in range(rows):
            d = 1-math.sqrt(abs(x-cx)**2+abs(y-cy)**2)/max_d
            f[x,y] = d*d
    return f

def apply_linear_filter(image, f):
    f = f.flatten()
    new = []
    for i in range(len(image)):
        new.append(min(1,image[i]*f[i]))
    return new
# apply a filter matrix to image
def apply_filter(image, f):
    rows, cols = image.shape
    new = np.zeros(rows*cols).reshape(rows,cols)
    for x in range(cols):
        for y in range(rows):
            new[x,y] = min(1, image[x,y]*f[x,y])
    return new

# takes in a dataset X, and transforms each row, outputs a new dataset
# might wanna make it in place if dealing with large dataset to save memory
def transform_features(data):
    with open('transformed_test_inputs.csv', 'wb') as csvfile:
        d = get_circle_filter(48,48)
        # writer = csv.writer(csvfile)
        # writer.writerow(["Id", "Prediction"])
        out_data = []
        i = 1
        bar = pyprind.ProgBar(len(data), title="Transforming raw features")
        for row in data:
            # from (2304,) to (48,48) because i wrote the functions for 2d , can optimize later
            m = a2m(row)
            dm = deskew(apply_filter(exposure.adjust_gamma(m,0.4), d))
            dm = dm-get_dilated(dm)*0.5
            # out_data.append(dm.flatten())
            # print len(dm.flattenedten())
            r = "%s," % str(i)
            for j in dm.flatten():
                r += "%s," % str(j)
            csvfile.write(r[:-1] + "\n")
            i += 1
            bar.update()
        return out_data

def read_preds(filename):
    pred = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader: 
            if row[1] != "Prediction":
                pred.append(int(row[1]))
    return pred

def get_ensemble(preds):
    best = []
    for i in range(len(preds[0])):
        tmp = []
        for p in preds:
            tmp.append(p[i])
        bin = np.bincount(tmp)
        idx = np.argmax(bin)
        if bin[idx] != 3:
            print "Disagree at index %d" % i
        best.append(np.argmax(np.bincount(tmp)))
    return best


def makeHeader(i):
    header = 'Id'
    for _ in range(i):
        dim = ',dim_%i'%_
        header += dim
    return header
    
def main():
    from import_data import import_csv
    from sklearn.decomposition import PCA
    import time
    data_files_path = 'data_and_scripts/'
    
    TRAIN_INPUTS_PATH = data_files_path+'train_inputs.csv'
    TEST_INPUTS_PATH = data_files_path+'test_inputs.csv'

    TRAIN_INPUTS_SUBSET_PATH = data_files_path+'train_inputs_subset.csv'
    
    #get the original inputs
    starttime = time.clock()
    train_inputs = import_csv(TRAIN_INPUTS_PATH)
    test_inputs = import_csv(TEST_INPUTS_PATH)
    print 'Time to import: %0.1f'%(time.clock() - starttime)    
    
    N,K = np.shape(train_inputs)
    T,Ki = np.shape(test_inputs)
    print N , K
    print T , Ki
    #concatenate train and test image
    concat = np.concatenate((train_inputs, test_inputs), axis=0)    
    print 'concatenated'
    print np.shape(concat)
    
    #apply transformation
    starttime = time.clock()
    transformed_concat = transform_features(concat, 500)
    print 'Time to transform: %0.1f'%(time.clock() - starttime)
    
    #apply PCA
    # starttime = time.clock()
    # desired=500
    # print 'Reducing feature set size from %d to %d...'%(K,desired)
    # features = PCA(n_components=desired).fit_transform(transformed_concat)
    # print 'Time to transform: %0.1f'%(time.clock() - starttime)
    features = transformed_concat
    
    #split
    transform_train_inputs, transform_test_inputs = features[:N,], features[N:,]
    
    #save to csv file    
    starttime = time.clock()
    print 'saving to csv file'
    header = makeHeader(desired)
    #Id column
    transform_train_inputs = np.concatenate((np.arange(N).reshape(N,1),transform_train_inputs), axis=1)
    transform_test_inputs = np.concatenate((np.arange(T).reshape(T,1),transform_test_inputs),axis=1)
    np.savetxt(data_files_path+'transformed_train_inputs.csv',transform_train_inputs,fmt='%f', delimiter=',', newline='\n', header=header,comments='')
    np.savetxt(data_files_path+ 'transformed_test_inputs.csv',transform_test_inputs,fmt='%f', delimiter=',', newline='\n', header=header,comments='')
    print 'Time to save: %0.1f'%(time.clock() - starttime)
    

   
# if __name__ == '__main__': main()
