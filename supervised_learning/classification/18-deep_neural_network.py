#!/usr/bin/env python3
'''
    Script Documentation
'''

import numpy as np


class DeepNeuralNetwork:
    '''
    Defines a deep neural network for performing binary classification.

    Attributes:
        nx (int): Number of input features.
        layers (list): List representing the number of nodes
        in each layer of the network.
        L (int): Number of layers in the neural network.
        cache (dict): Dictionary to hold the intermediary
        values of the network (i.e., the activations).
        weights (dict): Dictionary to hold the weights
        and biases of the network.

    Methods:
        __init__(self, nx, nodes)
        Initializes the neural network with given input features and nodes in
        the hidden layer.
    '''

    def __init__(self, nx, layers):
        '''
        Initializes the deep neural network.

        Args:
            nx (int): Number of input features.
            layers (list): List representing the number of nodes
            in each layer of the network.

        Raises:
            TypeError: If `nx` is not an integer.
            ValueError: If `nx` is less than 1.
            TypeError: If `layers` is not a list of positive integers.
        '''
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        self.__nx = nx
        self.__layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            W_key = f"W{i + 1}"
            b_key = f"b{i + 1}"

            self.__weights[b_key] = np.zeros((layers[i], 1))

            if i == 0:
                f = np.sqrt(2 / nx)
                self.__weights[W_key] = np.random.randn(layers[i], nx) * f
            else:
                f = np.sqrt(2 / layers[i - 1])
                self.__weights[W_key] = \
                    np.random.randn(layers[i], layers[i - 1]) * f

    @property
    def L(self):
        '''
            Getter
        '''
        return self.__L

    @property
    def cache(self):
        '''
            Getter
        '''
        return self.__cache

    @property
    def weights(self):
        '''
            Getter
        '''
        return self.__weights

    def forward_prop(self, X):
        '''
            Public method
            DeepNeuralNetwork Forward Propagation
        '''
        self.__cache['A0'] = X

        for i in range(self.__L):
            W_key = "W{}".format(i + 1)
            b_key = "b{}".format(i + 1)
            A_key_prev = "A{}".format(i)
            A_key_forw = "A{}".format(i + 1)
            Z = np.matmul(self.__weights[W_key], self.__cache[A_key_prev]) \
                + self.__weights[b_key]
            self.__cache[A_key_forw] = 1 / (1 + np.exp(-Z))
        return self.__cache[A_key_forw], self.__cache
