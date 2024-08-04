#!/usr/bin/env python3
'''
    Script Documentation
'''


def cat_matrices2D(mat1, mat2, axis=0):
    '''
        Function Documentation
    '''
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        new_mat = []
        for row in mat1:
            new_mat.append(row)
        for row in mat2:
            new_mat.append(row)
        return new_mat
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        new_mat = []
        for i in range(len(mat1)):
            new_row = mat1[i] + mat2[i]
            new_mat.append(new_row)
        return new_mat
    else:
        return None
