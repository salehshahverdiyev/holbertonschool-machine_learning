#!/usr/bin/env python3
'''
    Script Documentation
'''


def matrix_shape(matrix):
    '''
        Function Documentation
    '''
    shape = []
    if isinstance(matrix[0], int):
        shape.append(len(matrix))
        return shape
    if isinstance(matrix[0][0], int):
        row = len(matrix)
        column = len(matrix[0])
        shape.append(row)
        shape.append(column)
        return shape
    if isinstance(matrix[0][0], list) and isinstance(matrix[0][0][0], int):
        row = len(matrix)
        column = len(matrix[0])
        count = len(matrix[0][0])
        shape.append(row)
        shape.append(column)
        shape.append(count)
        return shape
