#!/usr/bin/env python3
'''
    Script Documentation
'''


def matrix_shape(matrix):
    '''
        Function Documentation
    '''
    shape = []
    row = len(matrix)
    column = len(matrix[0])
    shape.append(row)
    shape.append(column)
    if isinstance(matrix[0][0], int):
        return shape
    elif isinstance(matrix[0][0], list):
        count = len(matrix[0][0])
        shape.append(count)
        return shape
