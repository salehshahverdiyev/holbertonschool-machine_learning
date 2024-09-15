#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np


def normalization_constants(X):
    '''
        Calculates the normalization (standardization) constants of a matrix:
        X is the numpy.ndarray of shape (m, nx) to normalize
            m is the number of data points
            nx is the number of features
        Returns: the mean and standard deviation of each feature, respectively
    '''
    mean = X.mean(axis=0)
    stdrt = X.std(axis=0)
    return mean, stdrt
