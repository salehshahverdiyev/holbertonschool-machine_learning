#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''
        Y is a one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the data.
            classes is the number of classes
            m is the number of data points
        weights is a dictionary of the weights and biases of the neural network
        cache is a dictionary of the outputs of
            each layer of the neural network
        alpha is the learning rate
        lambtha is the L2 regularization parameter
        L is the number of layers of the network
        The neural network uses tanh activations on each layer
            except the last, which uses a softmax activation.
        The weights and biases of the network should be updated in place
    '''
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        A = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        dW = (1 / m) * np.matmul(dZ, A.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dZ = np.matmul(W.T, dZ) * (1 - A ** 2)
        weights['W' + str(i)] = weights['W' + str(i)] - alpha * dW
        weights['b' + str(i)] = weights['b' + str(i)] - alpha * db
