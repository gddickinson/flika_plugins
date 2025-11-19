#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 11:10:39 2024

@author: george
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
import pandas as pd
import math


#A = np.random.randint(800, 1000, size=[100, 100])
A = np.array([[0., 0.], [5., 0.],[5. ,5.],[0. ,5.]])

df = pd.DataFrame(data = A, columns = ['x', 'y'])


plt.scatter(df['x'],df['y'])

# #rotate
# B = df.to_numpy()
# rot = rotate(B, angle=45, reshape=False)
# rotDF = pd.DataFrame(data = rot, columns = ['x', 'y'])


# Define the rotation angle in degrees
angle = 45


# Get the center of the dataframe
center = df[['x', 'y']].mean()

def rotate_around_point(x,y, angle, origin=(0,0)):
    radians = angle * math.pi / 180
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    dx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    dy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return dx, dy


df['X_rotated'],df['Y_rotated'] = rotate_around_point(df['x'],df['y'], angle, origin=center)

plt.scatter(df['X_rotated'],df['Y_rotated'])
