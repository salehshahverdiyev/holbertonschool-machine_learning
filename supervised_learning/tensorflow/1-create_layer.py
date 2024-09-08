#!/usr/bin/env python3
'''
    Script Documentation
'''
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_layer(prev, n, activation):
    '''
        prev is the tensor output of the previous layer
        n is the number of nodes in the layer to create
        activation is the activation function that the layer should use
        use tf.keras.initializers.VarianceScaling(mode='fan_avg') to implement
            He et. al initialization for the layer weights
        each layer should be given the name layer
        Returns: the tensor output of the layer
    '''
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer, name='layer')

    output = layer(prev)
    return output
