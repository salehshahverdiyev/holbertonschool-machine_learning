#!/usr/bin/env python3
'''
    Script Documentation
'''

import numpy as np


class NeuralNetwork:
    '''
        Private instance attributes:
            W1: The weights vector for the hidden layer. Upon instantiation,
                it should be initialized using a random normal distribution.
            b1: The bias for the hidden layer. Upon instantiation,
                it should be initialized with 0's.
            A1: The activated output for the hidden layer.
                Upon instantiation, it should be initialized to 0.
            W2: The weights vector for the output neuron. Upon instantiation,
                it should be initialized using a random normal distribution.
            b2: The bias for the output neuron. Upon instantiation,
                it should be initialized to 0.
            A2: The activated output for the output neuron (prediction).
                Upon instantiation, it should be initialized to 0.
    '''
    def __init__(self, nx, nodes):
        '''
            class constructor: def __init__(self, nx, nodes):
                nx is the number of input features
                    If nx is not an integer, raise a TypeError
                        with the exception: nx must be an integer
                    If nx is less than 1, raise a ValueError
                        with the exception: nx must be a positive integer
                nodes is the number of nodes found in the hidden layer
                    If nodes is not an integer, raise a TypeError
                        with the exception: nodes must be an integer
                    If nodes is less than 1, raise a ValueError
                        with the exception: nodes must be a positive integer
                All exceptions should be raised in the order listed above
        '''
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if isinstance(nodes, int) is False:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        '''
            Getter
        '''
        return self.__W1

    @property
    def b1(self):
        '''
            Getter
        '''
        return self.__b1

    @property
    def A1(self):
        '''
            Getter
        '''
        return self.__A1

    @property
    def W2(self):
        '''
            Getter
        '''
        return self.__W2

    @property
    def b2(self):
        '''
            Getter
        '''
        return self.__b2

    @property
    def A2(self):
        '''
            Getter
        '''
        return self.__A2

    def forward_prop(self, X):
        '''
            Public method
            NeuralNetwork Forward Propagation
        '''
        C1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-C1))
        C2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-C2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''
            Public method
            NeuralNetwork Cost
        '''
        cost = -np.sum((Y * np.log(A)) +
                       ((1 - Y) * np.log(1.0000001 - A))) / Y.shape[1]
        return cost

    def evaluate(self, X, Y):
        '''
            Public method
            Evaluate NeuralNetwork
        '''
        self.forward_prop(X)
        A2 = np.where(self.__A2 >= .5, 1, 0)
        cost = self.cost(Y, self.__A2)
        return A2, cost
