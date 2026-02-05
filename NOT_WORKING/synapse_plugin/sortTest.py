# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 11:33:38 2019

@author: George
"""

import numpy as np
from matplotlib import pyplot as plt
import math

test = np.array([[ 34542.1,  25393.1],
    [ 34562.7,  25626.2],
    [ 34467. ,  25506.5],
    [ 34538. ,  25634.6],
    [ 34502.4,  25405.9],
    [ 34502.4,  25405.9],
    [ 34545.4,  25633. ],
    [ 34545.4,  25633. ],
    [ 34582.,  25536.], 
    [ 34582.,  25536.], 
    [ 34459.1,  25630.6], 
    [ 34459.1,  25630.6], 
    [ 34542.1,  25393.1], 
    [ 34467. ,  25506.5], 
    [ 34562.7,  25626.2], 
    [ 34538. ,  25634.6]])

testList = test.tolist()

center = np.average(test,axis=0)


def clockwiseangle_and_distance(point):
    # Vector between point and the origin: v = p - o
    vector = [point[0]-origin[0], point[1]-origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0]/lenvector, vector[1]/lenvector]
    dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
    diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them 
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2*math.pi+angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


origin = testList[0]
refvec = [0, 1]

ans = sorted(testList, key=clockwiseangle_and_distance)

#plt.scatter(test[:,0],test[:,1])
#plt.scatter(center[0],center[1])
#plt.show()