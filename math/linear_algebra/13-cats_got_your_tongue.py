#!/usr/bin/env python3
'''
    Script Documentation
'''


import numpy as np


def np_cat(mat1, mat2, axis=0):
    '''
        Function Documentation
    '''
    new_mat = np.concatenate((mat1, mat2), axis)
    return new_mat
