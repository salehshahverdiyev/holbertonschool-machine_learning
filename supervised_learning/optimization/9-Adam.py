#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np


def create_RMSProp_op(alpha, beta2, epsilon):
    '''
        updates a variable in place using the Adam optimization algorithm:
        alpha is the learning rate
        beta1 is the weight used for the first moment
        beta2 is the weight used for the second moment
        epsilon is a small number to avoid division by zero
        var is a numpy.ndarray containing the variable to be updated
        grad is a numpy.ndarray containing the gradient of var
        v is the previous first moment of var
        s is the previous second moment of var
        t is the time step used for bias correction
        Returns: the updated variable,
            the new first moment, and the new second moment, respectively
    '''
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * np.square(grad)
    v_new = v / (1 - beta1 ** t)
    s_new = s / (1 - beta2 ** t)
    var = var - (alpha / (np.sqrt(s_new) + epsilon)) * v_new
    return var, v, s
