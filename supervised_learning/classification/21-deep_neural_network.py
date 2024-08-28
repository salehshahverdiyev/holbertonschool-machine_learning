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

    def cost(self, Y, A):
        '''
            Public method
            DeepNeuralNetwork Cost
        '''
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        '''
            Public method
            Evaluate DeepNeuralNetwork
        '''
        A_final = self.forward_prop(X)[0]
        A_adjus = np.where(A_final >= 0.5, 1, 0)
        cost = self.cost(Y, A_final)
        return A_adjus, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        '''
            Public method
            DeepNeuralNetwork Gradient Descent
        '''
        weights = self.__weights.copy()
        m = Y.shape[1]
        for i in reversed(range(self.__L)):
            if i == self.__L - 1:
                dZ = cache['A{}'.format(i + 1)] - Y
                dW = np.matmul(cache['A{}'.format(i)], dZ.T) / m
            else:
                dZa = np.matmul(weights['W{}'.format(i + 2)].T, dZ)
                dZb = (cache['A{}'.format(i + 1)]
                       * (1 - cache['A{}'.format(i + 1)]))
                dZ = dZa * dZb
                dW = (np.matmul(dZ, cache['A{}'.format(i)].T)) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            if i == self.__L - 1:
                self.__weights['W{}'.format(i + 1)] = \
                    (weights['W{}'.format(i + 1)]
                     - (alpha * dW).T)
            else:
                self.__weights['W{}'.format(i + 1)] = \
                    weights['W{}'.format(i + 1)] \
                    - (alpha * dW)
            self.__weights['b{}'.format(i + 1)] = \
                weights['b{}'.format(i + 1)] \
                - (alpha * db)
