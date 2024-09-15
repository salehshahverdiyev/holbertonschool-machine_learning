#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    '''
        normalizes an unactivated output of a neural
            network using batch normalization:
        Z is a numpy.ndarray of shape (m, n) that should be normalized
            m is the number of data points
            n is the number of features in Z
        gamma is a numpy.ndarray of shape (1, n) containing
            the scales used for batch normalization
        beta is a numpy.ndarray of shape (1, n) containing
            the offsets used for batch normalization
        epsilon is a small number used to avoid division by zero
        Returns: the normalized Z matrix
    '''
    mean = np.mean(Z, axis=0)
    var_new = np.var(Z, axis=0)
    Z_n = (Z - mean) / np.sqrt(var_new + epsilon)
    return gamma * Z_n + beta
