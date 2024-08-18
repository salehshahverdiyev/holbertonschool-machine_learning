#!/usr/bin/env python3
'''
    Script Documentation
'''

import numpy as np


def left_child_add_prefix(text):
    '''
        Function Documentation
    '''
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "    |  " + line + "\n"
    return new_text


def right_child_add_prefix(text):
    '''
        Function Documentation
    '''
    lines = text.split("\n")
    new_text = "    +--" + lines[0] + "\n"
    for line in lines[1:]:
        new_text += "       " + line + "\n"
    return new_text


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

    def count_nodes_below(self, only_leaves=False):
        '''
            Function Documentation
        '''
        if only_leaves:
            if self.is_leaf:
                return 1
            count_left = (self.left_child.count_nodes_below(only_leaves)
                          if self.left_child else 0)
            count_right = (self.right_child.count_nodes_below(only_leaves)
                           if self.right_child else 0)
            return count_left + count_right
        else:
            count_left = (self.left_child.count_nodes_below(only_leaves)
                          if self.left_child else 0)
            count_right = (self.right_child.count_nodes_below(only_leaves)
                           if self.right_child else 0)
            return 1 + count_left + count_right

    def __str__(self):
        '''
            Function Documentation
        '''
        text = f"[feature={self.feature}, threshold={self.threshold}]"
        if self.left_child:
            text += "\n" + left_child_add_prefix(str(self.left_child))
        if self.right_child:
            text += "\n" + right_child_add_prefix(str(self.right_child))
        return text


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
        '''
            Function Documentation
        '''
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        '''
            Function Documentation
        '''
        return 1 if not only_leaves or self.is_leaf else 0

    def __str__(self):
        '''
            Function Documentation
        '''
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
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

    def count_nodes(self, only_leaves=False):
        '''
            Function Documentation
        '''
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        '''
            Function Documentation
        '''
        return str(self.root)
