#!/usr/bin/env python3
'''
    Script Documentation
'''


def matrix_transpose(matrix):
    '''
        Function Documentation
    '''
    new_mat = []
    empty = []
    i = 0
    j = 0
    for j in range(len(matrix[0])):
        for i in range(len(matrix)):
            empty.append(matrix[i][j])
            i += 1
        new_mat.append(empty)
        empty = []
        j += 1
    return new_mat
