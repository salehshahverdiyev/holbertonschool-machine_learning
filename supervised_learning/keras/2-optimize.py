#!/usr/bin/env python3
'''
    Script Documentation
'''
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    '''
        Sets up Adam optimization for a keras model
            with categorical crossentropy loss and accuracy metrics:
        network is the model to optimize
        alpha is the learning rate
        beta1 is the first Adam optimization parameter
        beta2 is the second Adam optimization parameter
        Returns: None
    '''
    network.compile(optimizer=K.optimizers.Adam(alpha, beta1, beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

    return None
