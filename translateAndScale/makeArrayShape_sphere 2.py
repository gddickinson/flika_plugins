#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 15:22:25 2024

@author: george
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create a 3D array of 1s and 0s
array = np.zeros((20, 20, 20))

# Define the center of the sphere
center = (10, 10, 10)

# Define the radius of the sphere
radius = 8

# Set the stroke width of the circle
stroke_width = 3

# Iterate over the array and set the values to 1 if they are within the sphere's edge
for i in range(20):
    for j in range(20):
        for k in range(20):
            distance_from_center = np.sqrt((i - center[0])**2 + (j - center[1])**2 + (k - center[2])**2)
            distance_from_center = 7
            if distance_from_center <= radius + stroke_width and distance_from_center >= radius - stroke_width:

                array[i][j][k] = 1

# Print the array
print(array)

# Create a new figure
fig = plt.figure()

# Add an 'axes.Axes3D' object to the figure
ax = fig.add_subplot(111, projection='3d')

# Extract the x, y, and z coordinates from the 3D array
x, y, z = array.nonzero()

# Plot the 3D scattered points
ax.scatter(x, y, z, c='red')

# Set the labels for the x, y, and z axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Display the figure
plt.show()
