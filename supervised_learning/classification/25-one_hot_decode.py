#!/usr/bin/env python3
'''
    Script Documentation
'''

import numpy as np


def one_hot_decode(one_hot):
    '''
        Converts a one-hot matrix into a vector of labels:
        one_hot is a one-hot encoded numpy.ndarray with shape (classes, m)
            classes is the maximum number of classes
            m is the number of examples
        Returns: a numpy.ndarray with shape (m, ) containing
            the numeric labels for each example, or None on failure
    '''
    if type(one_hot) is not np.ndarray\
       or len(one_hot.shape) != 2:
        return None
    return np.array([np.where(i == 1)[0][0]
                     for i in one_hot.T])
