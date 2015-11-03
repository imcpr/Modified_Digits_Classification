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
# from matplotlib import pyplot as plt
from matplotlib.image import imsave
from sklearn import svm, metrics
from skimage.morphology import reconstruction

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
    # print angle
    center = tuple(map(int, (x, y)))
    # print center
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

# helper to load training set
def get_train(index=50000):
    train_inputs = []
    train_outputs = []
    with open('data_and_scripts/train_inputs.csv', 'rb') as csvfile:
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
    with open('data_and_scripts/train_outputs.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader, None)  # skip the header
        i = 0
        for train_output in reader:  
            if i >= index:
                break
            i += 1
            train_output_no_id =  int(train_output[1])
            train_outputs.append(train_output_no_id)
    return train_inputs, train_outputs

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
    d = get_circle_filter(48,48)
    out_data = []
    bar = pyprind.ProgBar(len(data))
    for row in data:
        # from (2304,) to (48,48) because i wrote the functions for 2d , can optimize later
        m = a2m(row)
        dm = deskew(apply_filter(exposure.adjust_gamma(m,0.4), d))
        dm = dm-get_dilated(dm)*0.5
        out_data.append(dm.flatten())
        bar.update()
    return out_data

# d = get_circle_filter(48,48)
# X, Y = get_train(100)
# dX = []
# for i in range(0, len(X)):
#     m = a2m(X[i])
#     dm = deskew(apply_filter(exposure.adjust_gamma(m,0.4), d))
#     dm = dm-get_dilated(dm)*0.5
#     # imsave("data_and_scripts/original/%d.png" % (i+1), m, cmap="Greys_r")
#     imsave("data_and_scripts/deskew/%d.png" % (i+1), dm, cmap="Greys_r")

# cl = svm.SVC(gamma=0.001)
# cl = cl.fit(dX[:8000], Y[:8000])
# pred = cl.predict(dX[8000:])
# print metrics.zero_one_loss(Y[8000:], pred, normalize=False)

# MNIST test
# for i in range(0, 70000):
#     if (i %700 == 0):
#         m = a2m(mnist.data[i])
#         dm = deskew(m)
#         imsave("original/%d.png" % (i+1), m)
#         imsave("deskew/%d.png" % (i+1), dm)
