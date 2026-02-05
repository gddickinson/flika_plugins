# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 13:30:03 2019

@author: George
"""
import numpy as np

dict1 = {'a': np.array([1,2,3]), 'b': 2, 'c': 3, 'd': 4, 'e': 5}

# Check for items greater than 2
dict1_cond = {k:v for (k,v) in dict1.items() if dict1['a'].any() > 4}

print(dict1_cond)