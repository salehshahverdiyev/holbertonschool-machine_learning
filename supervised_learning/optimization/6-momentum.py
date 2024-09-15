#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np


def create_momentum_op(alpha, beta1):
    '''
        sets up the gradient descent
            with momentum optimization algorithm in TensorFlow:
        alpha is the learning rate.
        beta1 is the momentum weight.
        Returns: optimizer
    '''
    # different way
    # optimizer = tf.compat.v1.train.MomentumOptimizer(alpha, beta1)
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
