# coding=utf-8
# Authors:
#   Kian Kenyon-Dean.
#
# Coding began October 31st, 2015

# import numpy as np
import math

def sigmoid(x):
    return 1.0/(1.0 + math.e**(-x))