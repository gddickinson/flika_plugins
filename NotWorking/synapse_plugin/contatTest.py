# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:49:12 2019

@author: George
"""

import numpy as np

ch1hull = np.array([[ 85, 173],
 [ 85, 290],
 [ 30, 218],
 [192, 173],
 [192, 218],
 [233,  81],
 [233, 329],
 [ 88, 290],
 [ 88, 329],
 [150,  30],
 [149,  81],
 [149, 150]])

ch2hull=np.array([[14, 11],
 [15, 11],
 [ 3, 10],
 [ 3, 14],
 [16, 10],
 [16, 15]])

test = np.concatenate((ch1hull,ch2hull))