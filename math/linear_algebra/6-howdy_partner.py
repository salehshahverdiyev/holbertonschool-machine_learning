#!/usr/bin/env python3
'''
    Script Documentation
'''


def cat_arrays(arr1, arr2):
    '''
        Function Documentation
    '''
    new_arr = []
    if isinstance(arr1[0], int) or isinstance(arr1[0], float):
        if isinstance(arr2[0], int) or isinstance(arr2[0], float):
            if len(arr1) > 0 and len(arr2) > 0:
                new_arr = arr1 + arr2
                return new_arr
            if len(arr1) == 0 and len(arr2) == 0:
                return []
            if len(arr1) > 0 and len(arr2) == 0:
                new_arr = arr1
                return new_arr
            elif len(arr1) == 0 and len(arr2) > 0:
                new_arr = arr2
                return new_arr
