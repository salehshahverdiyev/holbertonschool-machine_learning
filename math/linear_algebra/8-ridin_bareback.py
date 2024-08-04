#!/usr/bin/env python3
'''
    Script Documentation
'''


def mat_mul(mat1, mat2):
    '''
        Function Documentation
    '''
    if len(mat1[0]) != len(mat2):
        return None
    num_rows_mat1 = len(mat1)
    num_cols_mat2 = len(mat2[0])
    result = []
    for i in range(num_rows_mat1):
        new_row = [0] * num_cols_mat2
        for j in range(num_cols_mat2):
            sum_product = 0
            for k in range(len(mat2)):
                sum_product += mat1[i][k] * mat2[k][j]
            new_row[j] = sum_product
        result.append(new_row)
    return result
