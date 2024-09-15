#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np


def moving_average(data, beta):
    '''
        Calculates the weighted moving average of a data set:
        data is the list of data to calculate the moving average of
        beta is the weight used for the moving average
        Your moving average calculation should use bias correction
        Returns: a list containing the moving averages of data
    '''
    value = 0
    move_avg = []
    for i in range(len(data)):
        value = beta * value + (1 - beta) * data[i]
        move_avg.append(value / (1 - beta ** (i + 1)))
    return move_avg
