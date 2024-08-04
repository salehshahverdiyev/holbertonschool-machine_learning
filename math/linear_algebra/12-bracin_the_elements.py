#!/usr/bin/env python3
'''
    Script Documentation
'''


def np_elementwise(mat1, mat2):
    '''
        Function Documentation
    '''
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    elementwise = (add, sub, mul, div)
    return elementwise
