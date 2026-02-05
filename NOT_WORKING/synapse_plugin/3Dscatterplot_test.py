# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:13:36 2020

@author: George
"""

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
import sys


def plot3DScatter(data):
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.opts['distance'] = 20
    w.show()
    w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')    
    g = gl.GLGridItem()
    w.addItem(g)        
    sp = gl.GLScatterPlotItem(pos=data, color=(1,1,1,.3), size=0.1, pxMode=False)    
    w.addItem(sp)   
    QtGui.QApplication.instance().exec_()
    return
    
pos3 = np.zeros((100,100,3))
pos3[:,:,:2] = np.mgrid[:100, :100].transpose(1,2,0) * [-0.1,0.1]
pos3 = pos3.reshape(10000,3)
d3 = (pos3**2).sum(axis=1)**0.5

plot3DScatter(pos3)
