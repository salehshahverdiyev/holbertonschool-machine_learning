#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np


def f1_score(confusion):
    '''
        calculates the F1 score of a confusion matrix:
        confusion is a confusion numpy.ndarray of shape (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels.
                classes is the number of classes
        Returns: a numpy.ndarray of shape (classes,)
            containing the F1 score of each class.
        You must use sensitivity = __import__('1-sensitivity').sensitivity and
            precision = __import__('2-precision').precision create previously.
    '''
    sens = sensitivity(confusion)
    prec = precision(confusion)
    F1 = 2 * (sens * prec) / (sens + prec)
    return F1
