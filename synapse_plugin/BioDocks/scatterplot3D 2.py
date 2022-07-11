# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 14:13:36 2020

@author: George
"""

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
import numpy as np
import sys

class Plot3D_GL(QtWidgets.QDialog):
    def __init__(self, data1,data2, parent = None):
        
        self.data1 = data1
        self.data2 = data2

        self.app = QtGui.QApplication([])
        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 200
        
        self.w.setWindowTitle('pyqtgraph example: GLScatterPlotItem')    
        
    def plot(self):   
        
        self.g = gl.GLGridItem()
        self.w.addItem(self.g)   
    
        #color1 = np.ones((data.shape[0], 4))
        #color2 = np.ones((data.shape[0], 4))
        
        color1 = (1.0, 0.0, 0.0, 0.5)
        color2 = (0.0, 0.0, 1.0, 0.5)
        
        pxSize = 1
        pxMode = True
         
        self.sp1 = gl.GLScatterPlotItem(pos=self.data1, color=color1, size=pxSize , pxMode=pxMode) 
        self.sp2 = gl.GLScatterPlotItem(pos=self.data2, color=color2, size=pxSize , pxMode=pxMode) 
        self.w.addItem(self.sp1)
        self.w.addItem(self.sp2) 
        #sp1.setGLOptions('opaque')
        
        #center positions to view more easily
        xT = max(self.data1[::,0])
        yT = max(self.data1[::,1])
        zT = max(self.data1[::,2])
        
        self.sp1.translate(-xT, -yT, -zT)
        self.sp2.translate(-xT, -yT, -zT)
        
        #self.sp1.rotate(0, 90, 1, 90)
        #self.sp2.rotate(0, 90, 1, 90)    
        
        #self.sp1.rotate(90, 0,0,1)
        #self.sp1.rotate(-90, 0,1,0)
        
        #self.sp2.rotate(90, 0,0,1)
        #self.sp2.rotate(-90, 0,1,0)
        
        #self.sp1.rotate(-90, 1,0,0)
    
        self.w.show()
    
        QtGui.QApplication.instance().exec_()
        sys.exit(app.exec_())

    

