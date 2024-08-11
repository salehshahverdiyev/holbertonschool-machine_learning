#!/usr/bin/env python3
'''
    Script Documentation
'''


def poly_integral(poly, C=0):
    '''
        Function Documentation
    '''
    if not isinstance(poly, list):
        return None
    for c in poly:
        if not isinstance(c, (int, float)):
            return None
    if not isinstance(C, (int, float)):
        return None
    integral = [C]
    for i in range(len(poly)):
        coeff = poly[i]
        integral_coeff = coeff / (i + 1)
        if integral_coeff.is_integer():
            integral.append(int(integral_coeff))
        else:
            integral.append(integral_coeff)
    if len(integral) == 1 and integral[0] == 0:
        return None
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    return integral
