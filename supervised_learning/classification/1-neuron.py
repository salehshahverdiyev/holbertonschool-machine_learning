#!/usr/bin/env python3
'''
    Script Documentation
'''

import numpy as np


class Neuron:
    '''
        Private instance attributes:
            W: The weights vector for the neuron. Upon instantiation,
               it should be initialized using a random normal distribution.
            b: The bias for the neuron. Upon instantiation,
               it should be initialized to 0.
            A: The activated output of the neuron (prediction).
               Upon instantiation, it should be initialized to 0.
    '''
    def __init__(self, nx):
        '''
            class constructor: def __init__(self, nx):
                nx is the number of input features to the neuron
            If nx is not an integer,
                raise a TypeError with the exception: nx must be an integer
            If nx is less than 1,
                raise a ValueError with the exception:
                nx must be a positive integer
            All exceptions should be raised in the order listed above
        '''
        if isinstance(nx, int) is False:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        '''
            Getter function
            Returns:
                numpy.ndarray
                The weight vector of the neuron.
        '''
        return self.__W

    @property
    def b(self):
        '''
            Getter function
            Returns:
                float
                The bias term of the neuron.
        '''
        return self.__b

    @property
    def A(self):
        '''
            Getter function
             Returns:
            float
                The activated output of the neuron.
        '''
        return self.__A
