# coding=utf-8
# Authors:
#   Kian Kenyon-Dean.
#
# Coding began October 31st, 2015

from import_data import import_csv


data_files_path = '/Users/kian/documents/COMP598/Assignment3/data_and_scripts/'

TRAIN_INPUTS_PATH = data_files_path+'train_inputs.csv'
TRAIN_OUTPUTS_PATH = data_files_path+'train_outputs.csv'
TEST_INPUTS_PATH = data_files_path+'test_inputs.csv'

if __name__ == '__main__':
    train_outputs = import_csv(TRAIN_OUTPUTS_PATH)
