"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
import numpy as np
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import difflib

from .AnalysisIO import *
from .bin2mat import *
from .tifffile import *
from .SettingsWidgets import *
from math import atan2
from scipy.spatial import distance as dist
from scipy.spatial import KDTree
import math

item_name = ""

def copy(item, name):
    global item_name
    item_name = name
    QtGui.QApplication.clipboard().setText("%s" % asStr(item))

def calculate_distances(arr, maximum = -1):
	dists = []
	for i, p1 in enumerate(arr):
		for p2 in arr[i+1:]:
			d = np.linalg.norm(np.subtract(p1,p2))
			if maximum < 0 or d <= maximum:
				dists.append(d)
	return dists

def sort_closest(lst, close_to=''):
    return sorted(lst, key=lambda i: -difflib.SequenceMatcher(None, i, close_to).ratio())

def sortXY(XYList):
    return sorted(XYList , key=lambda k: [k[1], k[0]])

def sortXY_aroundCircle(XYList):
    return XYList.sort(key=lambda c:atan2(c[0], c[1]))
   
def order_points(pts): 
    #get leftmost point for origin
    
    pts = sortXY(pts)
    xSorted = sorted(pts, key=lambda x: x[0])
    leftMost = xSorted[0]   
    origin = leftMost
    refvec = [0, 1]
    
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
        
    return sorted(pts, key=clockwiseangle_and_distance)


def combineClosestHulls(ch1_hulls,ch1_centeroids,ch1_groupPoints,ch2_hulls,ch2_centeroids,ch2_groupPoints, maxDistance):
    combinedHullList = []
    combinedPointsList = []
    for i in range(len(ch1_centeroids)):
        pt = ch1_centeroids[i]
        distance,index = KDTree(ch2_centeroids).query(pt)
        if distance <= maxDistance:
            ch1_hull = ch1_hulls[i]
            ch2_hull = ch2_hulls[index]
            ch1_points = ch1_groupPoints[i]
            ch2_points = ch2_groupPoints[index]                     
            combinedHullList.append(np.concatenate((ch1_hull,ch2_hull)))   
            combinedPointsList.append(np.concatenate((ch1_points,ch2_points)))               
    return combinedHullList, combinedPointsList

class AnalysisException(Exception):
    def __init__(self, parent, title='Analysis Error', s='Something went wrong'):
        super(AnalysisException, self).__init__(s)
        QtGui.QMessageBox.critical(parent, message, title)

def showMessage(parent, title, message):
	QtGui.QMessageBox.information(parent, title, message)

def random_color(high = 255, low=0):
    r, g, b = np.random.random((3,))

    return QtGui.QColor(int(r * high) + low, int(g * high) + low, int(b * high) + low)

def getOption(parent, title, options, label='Option:'):
	result, ok = QtGui.QInputDialog.getItem(parent, title, label, options, editable=False)
	assert ok, "Selection Canceled"
	return str(result)

def getString(parent, title="Please enter a string", label="String:", initial=""):
	s, ok = QtGui.QInputDialog.getText(parent, title, label, text=initial)
	assert ok, 'Action canceled'
	return str(s)

def getInt(parent, title="Please enter an integer", label="Value:", initial=0):
	v, ok = QtGui.QInputDialog.getInt(parent, title, label, value=initial)
	assert ok, "Action canceled"
	return int(v)

def getFloat(parent, title="Please enter an integer", label="Value:", initial=0):
	v, ok = QtGui.QInputDialog.getDouble(parent, title, label, value=initial)
	assert ok, "Action canceled"
	return float(v)

def asStr(obj):
	if isinstance(obj, np.ndarray):
		return str(obj.tolist())
	else:
		return repr(obj)

def getArea(points):
        a = 0
        ox,oy = points[0]
        for x,y in points[1:]:
            a += (x*oy-y*ox)
            ox,oy = x,y
        return abs(a/2)

#volume of polygon poly
def volume(poly):
    if len(poly) < 3: # not a plane - no volume
        return 0

    def dot(a, b):
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

    def cross(a, b):
        x = a[1] * b[2] - a[2] * b[1]
        y = a[2] * b[0] - a[0] * b[2]
        z = a[0] * b[1] - a[1] * b[0]
        return (x, y, z)

    def unit_normal(a, b, c):
        def det(a):
            return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

        x = det([[1,a[1],a[2]],
                 [1,b[1],b[2]],
                 [1,c[1],c[2]]])
        y = det([[a[0],1,a[2]],
                 [b[0],1,b[2]],
                 [c[0],1,c[2]]])
        z = det([[a[0],a[1],1],
                 [b[0],b[1],1],
                 [c[0],c[1],1]])
        magnitude = (x**2 + y**2 + z**2)**.5
        return (x/magnitude, y/magnitude, z/magnitude)

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)
