#!/usr/bin/env python3
'''
    Script Documentation
'''
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_train_op(loss, alpha):
    '''
        Creates the training operation for the network:
        loss is the loss of the network’s prediction
        alpha is the learning rate
        Returns: an operation that trains the network using gradient descent
    '''
    opt = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    return opt
