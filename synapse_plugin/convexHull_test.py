# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 16:00:56 2019

@author: George
"""
import numpy as np

from scipy.spatial import ConvexHull, convex_hull_plot_2d
points = np.random.rand(3, 2)   # 30 random points in 2-D
hull = ConvexHull(points)

import matplotlib.pyplot as plt
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], 'k-')

plt.show()