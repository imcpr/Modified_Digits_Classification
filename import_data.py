# coding=utf-8
# @author Kian Kenyon-Dean

""" Simple data importing from the CSV files. """

import numpy as np
import csv

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
            dataset.append(row_values)

    return np.array(dataset)


