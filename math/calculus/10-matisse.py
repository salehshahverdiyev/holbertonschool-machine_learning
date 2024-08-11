#!/usr/bin/env python3
'''
    Script Documentation
'''


def poly_derivative(poly):
    '''
        Function Documentation
    '''
    if not isinstance(poly, list):
        return None
    for co_eff in poly:
        if not isinstance(co_eff, (int, float)):
            return None
    if len(poly) == 0:
        return None
    derivative = []
    for i in range(1, len(poly)):
        derivative.append(i * poly[i])
    if len(derivative) == 0:
        return [0]
    else:
        return derivative
