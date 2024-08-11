#!/usr/bin/env python3
'''
    Script Documentation
'''


def summation_i_squared(n):
    '''
        Function Documentation
    '''
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
