#!/usr/bin/env python3
'''
    Script Documentation
'''


def add_arrays(arr1, arr2):
    '''
        Function Documentation
    '''
    new_arr = []
    if isinstance(arr1[0], int) or isinstance(arr1[0], float):
        if isinstance(arr2[0], int) or isinstance(arr2[0], float):
            if len(arr1) > 0 and len(arr2) > 0:
                if len(arr1) == len(arr2):
                    for i in range(len(arr1)):
                        sum = arr1[i] + arr2[i]
                        new_arr.append(sum)
                        i += 1
                    return new_arr
                elif len(arr1) != len(arr2):
                    return None
