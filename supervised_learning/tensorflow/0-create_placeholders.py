#!/usr/bin/env python3
'''
    Script Documentation
'''
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    '''
        Returns two placeholders, x and y, for the neural network:
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
        Returns: placeholders named x and y, respectively
            x is the placeholder for the input data to the neural network
            y is the placeholder for the one-hot labels for the input data

    '''
    x = tf.placeholder(float, shape=[None, nx], name='x')
    y = tf.placeholder(float, shape=[None, classes], name='y')
    return (x, y)
