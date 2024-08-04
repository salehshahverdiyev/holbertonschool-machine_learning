#!/usr/bin/env python3
'''
    Script Documentation
'''


def add_matrices2D(mat1, mat2):
    '''
        Function Documentation
    '''
    m1 = mat1
    m2 = mat2
    new_arr = []
    empty = []
    i = 0
    j = 0
    if len(m1[0]) > 0 and len(m2[0]) > 0:
        if isinstance(m1[0], list) and isinstance(m2[0], list):
            if isinstance(m1[0][0], int) or isinstance(m1[0][0], float):
                if isinstance(m2[0][0], int) or isinstance(m2[0][0], float):
                    if len(m1) == len(m2) and len(m1[0]) == len(m2[0]):
                        for i in range(len(m1)):
                            for j in range(len(m1[0])):
                                sum = m1[i][j] + m2[i][j]
                                empty.append(sum)
                                j += 1
                            new_arr.append(empty)
                            empty = []
                            i += 1
                        return new_arr
                    else:
                        return None
    else:
        return []
