#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:18:56 2024

@author: george
"""

import matplotlib.pyplot as plt
import numpy as np

def find_equilateral_triangle_vertices(centroid, side_length):
  """Finds the XY vertices of an equilateral triangle from the centroid and side length.

  Args:
    centroid: A tuple of (x, y) coordinates of the centroid.
    side_length: The length of a side of the equilateral triangle.

  Returns:
    A list of three tuples of (x, y) coordinates of the vertices of the equilateral triangle.
  """

  # Calculate the coordinates of the first vertex.
  vertex1 = (centroid[0] + side_length / 2, centroid[1] + side_length * np.sqrt(3) / 2)

  # Calculate the coordinates of the second vertex.
  vertex2 = (centroid[0] - side_length / 2, centroid[1] + side_length * np.sqrt(3) / 2)

  # Calculate the coordinates of the third vertex.
  vertex3 = (centroid[0], centroid[1] - side_length)

  return [vertex1, vertex2, vertex3]

# Create the centroid.
centroid = (0, 0)

# Create the side length.
side_length = 10

# Find the vertices of the equilateral triangle.
verticies = find_equilateral_triangle_vertices(centroid, side_length)

print(verticies)

# Plot the equilateral triangle.
plt.plot([vertex[0] for vertex in verticies], [vertex[1] for vertex in verticies], 'o-')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Equilateral Triangle')
plt.show()

angles = np.linspace(0, np.pi * 4 / 3, 3)
verticies = (np.array((np.sin(angles), np.cos(angles))).T * 1) / 2.0

print(verticies)
radius = 10

test= np.array([[i[0] + radius/2, i[1] + radius/2] for i in verticies])

print(test)
