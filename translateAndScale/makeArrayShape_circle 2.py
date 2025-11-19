#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:32:03 2024

@author: george
"""

import numpy as np

# Create a 20x20 array of zeros
array = np.zeros((21, 21))

# Set the center of the circle
center = (10, 10)

# Set the radius of the circle
radius = 7

# Set the stroke width of the circle
stroke_width = 3

# Iterate over the array and set the pixels inside the circle to 1
for i in range(array.shape[0]):
    for j in range(array.shape[1]):
        distance_from_center = np.sqrt((i - center[0])**2 + (j - center[1])**2)
        if distance_from_center <= radius + stroke_width and distance_from_center >= radius - stroke_width:
            array[i, j] = 1

# Print the array
print(array)
