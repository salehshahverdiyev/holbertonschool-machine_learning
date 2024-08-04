#!/usr/bin/env python3
'''
    Script Documentation
'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''
        Function Documentation
    '''
    new_mat = []
    i = 0
    if len(mat1) != len(mat2):
        new_mat = mat1 + mat2
        return new_mat
    if axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            temp = mat1[i] + mat2[i]
            new_mat.append(temp)
            i += 1
        return new_mat
