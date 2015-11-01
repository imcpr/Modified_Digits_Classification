# coding=utf-8
# Authors:
#   Kian Kenyon-Dean.
#
# Coding began October 31st, 2015

# import numpy as np
import math

def sigmoid(x, c=None):
    # Returns a function if it is passed a constant. Else return basic sigmoid
    if c:
        return lambda v: 1.0/(1.0 + math.e**(c*-v))
    return 1.0/(1.0 + math.e**(-x))