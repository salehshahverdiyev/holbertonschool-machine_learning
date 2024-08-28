#!/usr/bin/env python3
'''
    Script Documentation
'''

import numpy as np


def one_hot_encode(Y, classes):
    '''
        Converts a numeric label vector into a one-hot matrix:
        Y is a numpy.ndarray with shape (m,) containing numeric class labels
            m is the number of examples
        classes is the maximum number of classes found in Y
        Returns: a one-hot encoding of Y
            with shape (classes, m), or None on failure
    '''
    if Y is None\
       or type(Y) is not np.ndarray\
       or type(classes) is not int:
        return None
    try:
        matrix = np.zeros((len(Y), classes))
        matrix[np.arange(len(Y)), Y] = 1
        return matrix.T
    except Exception:
        return None
