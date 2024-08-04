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
    j = 0
    if axis == 0:
        new_mat = mat1 + mat2
        return new_mat
    if axis == 1:
        for i in range(len(mat2)):
            for j in range(len(mat2[0])):
                mat1[i].append(mat2[i][j])
                j += 1
            i += 1
        new_mat = mat1
        return new_mat
