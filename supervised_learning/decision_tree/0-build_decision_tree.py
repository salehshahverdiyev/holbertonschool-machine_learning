#!/usr/bin/env python3
'''
    Script Documentation
'''

import numpy as np


class Node:
    '''
        Class Documentation
    '''
    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        '''
            Function Documentation
        '''
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        '''
            Function Documentation
        '''
        max_depth_left = 0
        max_depth_right = 0
        if self.left_child:
            max_depth_left = self.left_child.max_depth_below()
        if self.right_child:
            max_depth_right = self.right_child.max_depth_below()
        return max(max_depth_left, max_depth_right) + 1


class Leaf(Node):
    '''
        Class Documentation
    '''
    def __init__(self, value, depth=None):
        '''
            Function Documentation
        '''
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        return self.depth


class Decision_Tree():
    '''
        Class Documentation
    '''
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        '''
            Function Documentation
        '''
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        '''
            Function Documentation
        '''
        return self.root.max_depth_below()
