# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 08:54:05 2020
Synapse3D - Clustering code
@author: George
"""
import os, sys, glob

try:
    from BioDocks import *
except:
    from .BioDocks import *
from pyqtgraph.dockarea import *
from scipy.spatial import ConvexHull
from collections import OrderedDict
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtGui import *
from qtpy.QtWidgets import *
from qtpy.QtCore import *
import flika
from flika import global_vars as g
from flika.window import Window
from distutils.version import StrictVersion
import numpy as np
import pyqtgraph as pg
from pyqtgraph import mkPen
from matplotlib import pyplot as plt
import copy
import pandas as pd
from sklearn.neighbors import KDTree
import random
import time
from mpl_toolkits.mplot3d import Axes3D # <--- This is needed for newer versions of matplotlib


flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

def setMenuUp(menuItem,menu,shortcut='Ctrl+X',statusTip='Exit',connection=None):
    if shortcut != None:
        menuItem.setShortcut(shortcut)
    menuItem.setStatusTip(statusTip)
    menuItem.triggered.connect(connection)
    menu.addAction(menuItem)
    return

class ClusterAnalysis:
    
    def __init__(self):
        #camera option
        self.unitPerPixel = 166
        # data is loaded in nanometers, divided by # according to units
        self.units = {'Pixels': self.unitPerPixel, 'Nanometers': 1}
        self.unit_prefixes = {'Pixels': 'px', 'Nanometers': 'nm'}
        self.unit = 'Nanometers'
        self.ignore = {"Z Rejected"}
        #clustering option
        self.eps = 100          #max distance between points within a cluster
        self.min_samples = 10   #min number of points to form a cluster
        self.maxDistance = 100  #max distance between clusters in differnt channels when forming combined ROI
        self.colors = ((255, 0, 0), (0, 255, 0))
        self.color_dict = {'atto488': self.colors[0], 'Alexa647': self.colors[1]}
        self.ignore = {"Z Rejected"}        
        self.Channels = []
        self.empty_channel = Channel('Empty', [], (1, 1, 1))
    
        #data state
        self.dataLoaded = False
        self.ROI3D_Initiated = False
        self.dataDisplayed = 'original'
        self.clustersGenerated = False
        self.centroidsGenerated = False
        
        
        self.clusterIndex = []
    
        #cluster analysis options
        self.clusterAnaysisSelection = 'All Clusters'
        self.clusterType = '3D'
        
        self.All_ROIs_pointsList = []
        self.channelList = []
        
        #self..multiThreadingFlag = False
        self.displayFlag = False
        
        #init data types
        self.ch1Points_3D = []
        self.ch2Points_3D = []


    def clear(self):
        self.Channels = []
    
    def open_file(self,filename=''):
        if filename == '':
            filename = getFilename(filter='Text Files (*.txt)')
        self.clear()
        self.data = importFile(filename,evaluateLines=False)
        try:        
            for i in range(len(self.data[0])):
                if '\n' in self.data[0][i]:
                    self.data[0][i] = self.data[0][i].split('\n')[0]                
    	
        except:
            pass

        self.colNames = list(self.data[0])
        #print(self.colNames)
        
        #filter 1000 for testing
        #self.data = self.data[0:5000]
    	
        self.data = {d[0]: d[1:] for d in np.transpose(self.data)}
       
        for k in self.data:
            if k != 'Channel Name':
                self.data[k] = self.data[k].astype(float)
        print('Gathering channels...')
        #g.m.statusBar().showMessage('Gathering channels...')        
        self.names = set(self.data['Channel Name'].astype(str)) - self.ignore
        print('Channels Found: %s' % ', '.join(self.names))
        #g.m.statusBar().showMessage('Channels Found: %s' % ', '.join(self.names))
        
        self.data['Xc'] /= self.units[self.unit]
        self.data['Yc'] /= self.units[self.unit]
        self.data['Zc'] /= self.units[self.unit]
        
        #global Channels
        self.Channels = []
        #self.plotWidget.clear()
        self.pts = [ActivePoint(data={k: self.data[k][i] for k in self.data}) for i in range(len(self.data['Channel Name']))]
        for i, n in enumerate(self.names):
            if n in self.color_dict:
                color = self.color_dict[n]
            else:
                color = self.colors[i]
            self.Channels.append(Channel(n, [p for p in self.pts if p['Channel Name'] == n], color))
        #self.plotWidget.addItem(self.Channels[-1])
        #self.legend.addItem(self.Channels[-1], n)
        #self.show_ch1.setText(self.Channels[0].__name__)
        #self.show_ch2.setText(self.Channels[1].__name__)
        #self.ch1_mesh.setText(self.Channels[0].__name__)
        #self.ch2_mesh.setText(self.Channels[1].__name__)
        
        self.ch1Points_3D = self.Channels[0].getPoints(z=True)
        self.ch2Points_3D = self.Channels[1].getPoints(z=True)
        
        self.ch1Points = self.Channels[0].getPoints(z=False)
        self.ch2Points = self.Channels[1].getPoints(z=False)
        self.dataLoaded = True
        return

    def viewerGUI(self):
        '''MAIN GUI'''
                  
        #3D viewer dock
        #self.app = QtWidgets.QApplication([])

        ## Create window with 4 docks
        self.win = QtWidgets.QMainWindow()
        self.win.resize(1800,650)
        self.win.setWindowTitle('Cluster Analysis Window')

        #create Dock area
        self.area = DockArea()
        self.win.setCentralWidget(self.area)

        #define docks
        self.dock1 = Dock("3D View", size=(600,600), closable=False)
        self.dock2 = Dock("2D View", size=(600,600), closable=False)
        self.dock3 = Dock("ROI Data", size=(600,600), closable=True)
        self.dockButtons = Dock("Buttons",size=(1800,50))
        
        #add docks to area
        self.area.addDock(self.dock1, 'left')
        self.area.addDock(self.dock2, 'right', self.dock1)           
        self.area.addDock(self.dock3, 'right', self.dock2) 
        self.area.addDock(self.dockButtons, 'bottom')         

        #initialise image widgets
        self.imv3D = Plot3DWidget()
        self.imv2D = pg.GraphicsLayoutWidget()
        self.imv3 = pg.GraphicsLayoutWidget()

        #add image widgets to docks
        self.dock1.addWidget(self.imv3D, 0, 0, 6, 6)
        self.dock2.addWidget(self.imv2D)
        self.dock3.addWidget(self.imv3)
        
        #create plot windows
        self.w2 = self.imv2D.addPlot()
        self.w3 = self.imv3.addPlot()

        self.state = self.area.saveState()
        self.menubar = self.win.menuBar()
        self.fileMenu1 = self.menubar.addMenu('&Options')
        
        self.resetLayout = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Reset Layout')
        setMenuUp(self.resetLayout,self.fileMenu1,shortcut='Ctrl+R',statusTip='Reset Layout',connection=self.reset_layout)

        self.showTitles = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Show Titles')
        setMenuUp(self.showTitles,self.fileMenu1,shortcut='Ctrl+G',statusTip='Show Titles',connection=self.show_titles)

        self.hideTitles = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Hide Titles')
        setMenuUp(self.hideTitles,self.fileMenu1,shortcut='Ctrl+H',statusTip='Hide Titles',connection=self.hide_titles)                   

        self.dockList = [self.dock1, self.dock2, self.dock3]
        
        ##QUICK BUTTON BAR##
        self.dockButtons.hideTitleBar()
        self.button_ToggleNoise = QtWidgets.QPushButton("Toggle Noise") 
        self.dockButtons.addWidget(self.button_ToggleNoise,0,0)
        self.button_ToggleNoise.clicked.connect(self.toggleNoise)

        self.button_ToggleClusters = QtWidgets.QPushButton("Toggle Clusters") 
        self.dockButtons.addWidget(self.button_ToggleClusters,0,1)
        self.button_ToggleClusters.clicked.connect(self.toggleClusters)
        
        self.button_ToggleCentroids = QtWidgets.QPushButton("Toggle Centroids") 
        self.dockButtons.addWidget(self.button_ToggleCentroids,0,2)
        self.button_ToggleCentroids.clicked.connect(self.toggleCentroids)        
        
        
        self.win.show()        

        return
        
    def reset_layout(self):
        self.area.restoreState(self.state)

    def hide_titles(self,_):
        for dock in self.dockList:
            dock.hideTitleBar()

    def show_titles(self):
        for dock in self.dockList:
            dock.showTitleBar()
        return

    def display2Ddata_allPoints(self):
        #make point data
        point_n1 = len(self.ch1Points_3D[::,0])
        point_n2 = len(self.ch2Points_3D[::,0])
        point_s1 = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120))
        point_s2 = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))        
        point_pos1 = np.array([self.ch1Points_3D[::,0],self.ch1Points_3D[::,1]])
        point_pos2 = np.array([self.ch2Points_3D[::,0],self.ch2Points_3D[::,1]])
        point_spots1 = [{'pos': point_pos1[:,i], 'data': 1} for i in range(point_n1)]
        point_spots2 = [{'pos': point_pos2[:,i], 'data': 1} for i in range(point_n2)]
        point_s1.addPoints(point_spots1)
        point_s2.addPoints(point_spots2)
        self.w2.addItem(point_s1)
        self.w2.addItem(point_s2)
        return

    def display3Ddata_allPoints(self, toggle=False):
        if toggle:
            dist = self.imv3D.opts['distance']
            elev = self.imv3D.opts['elevation'] 
            azim = self.imv3D.opts['azimuth']
            self.imv3D.clear()
        self.imv3D.addArray(self.ch1Points_3D,color=QColor(0, 255, 0),size=2,name='ch1_pts')
        self.imv3D.addArray(self.ch2Points_3D,color=QColor(255, 0, 0),size=2,name='ch2_pts') 
        if toggle:
            self.imv3D.opts['distance'] = dist
            self.imv3D.opts['elevation'] = elev
            self.imv3D.opts['azimuth'] = azim
            return
        self.imv3D.opts['distance'] = 10000
        self.imv3D.orbit(-135,90)
        return


    def getClusters(self):
        t = Timer() 
        t.start()
        #get cluster labels for each channel
        print('--- channel 1 ---')
        self.ch1_labels,self.ch1_numClusters,self.ch1_numNoise = dbscan(self.ch1Points_3D, eps=self.eps, min_samples=self.min_samples, plot=False)
        print('--- channel 2 ---')
        self.ch2_labels,self.ch2_numClusters,self.ch2_numNoise = dbscan(self.ch2Points_3D, eps=self.eps, min_samples=self.min_samples, plot=False)    
        print('-----------------')
            
        t.timeReport('3D clusters created') 
        
        ch1_Name, ch1_pts, ch1_color = self.Channels[0].filterPts(self.ch1_labels)     
        ch2_Name, ch2_pts, ch2_color = self.Channels[1].filterPts(self.ch2_labels) 
        
        self.clusterChannels = []
  
        self.clusterChannels.append(Channel(ch1_Name, ch1_pts, ch1_color))
        self.clusterChannels.append(Channel(ch2_Name, ch2_pts, ch2_color)) 
        
        self.ch1PointsNoNoise_3D = self.clusterChannels[0].getPoints(z=True)
        self.ch2PointsNoNoise_3D = self.clusterChannels[1].getPoints(z=True)
        self.clustersGenerated = True
        return


    def getHulls(self,points,labels):
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        hulls = []
        centroids = []
        groupPoints = []
        for i in range(n_clusters):
            clusterPoints = points[labels==i]
            groupPoints.append(clusterPoints)           
            hulls.append(ConvexHull(clusterPoints).simplices)
            centroids.append(np.average(clusterPoints,axis=0)) 
        return np.array(hulls), np.array(centroids), np.array(groupPoints)

    def getCentroids(self):
        t = Timer() 
        t.start()        
        #get 3D centeroids for cluster analysis
        _, self.ch1_centeroids_3D, _ = self.getHulls(self.ch1Points_3D,self.ch1_labels)
        _, self.ch2_centeroids_3D, _ = self.getHulls(self.ch2Points_3D,self.ch2_labels)        
        t.timeReport('3D clusters created') 
        self.centroidsGenerated = True
        return

    def getHulls2(self,pointList):
        hullList = []
        for points in pointList:
            hullList.append(ConvexHull(points).simplices)
        return hullList

    def makeHulls(self):
        t = Timer()
        #get hulls for each channels clusters  
        t.start()
        if self.clusterType == '3D':
            ch1_hulls, ch1_centeroids, self.ch1_groupPoints = self.getHulls(self.ch1Points_3D,self.ch1_labels)
            #self.plotHull(self.ch1_groupPoints[0],ch1_hulls[0])
            ch2_hulls, ch2_centeroids, self.ch2_groupPoints = self.getHulls(self.ch2Points_3D,self.ch2_labels)
            
        else:
            ch1_hulls, ch1_centeroids, self.ch1_groupPoints = self.getHulls(self.ch1Points,self.ch1_labels)
            #self.plotHull(self.ch1_groupPoints[0],ch1_hulls[0])
            ch2_hulls, ch2_centeroids, self.ch2_groupPoints = self.getHulls(self.ch2Points,self.ch2_labels)
                 
        #combine nearest roi between channels
        combinedHulls, combinedPoints, self.combined_ch1_Centeroids, self.combined_ch2_Centeroids = combineClosestHulls(ch1_hulls,ch1_centeroids,self.ch1_groupPoints,ch2_hulls,ch2_centeroids,self.ch2_groupPoints, self.maxDistance)
        
        #get new hulls for combined points
        newHulls = self.getHulls2(combinedPoints)         
        #self.plotHull(combinedPoints[0],newHulls[0])       
        t.timeReport('hulls created')


    def display2Dcentroids(self):
        #make centeroid data
        self.centeroid_s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.centeroid_s2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        #combined clusters
        centeroid_n1 = len(self.combined_ch1_Centeroids[::,0])
        centeroid_n2 = len(self.combined_ch2_Centeroids[::,0])
        centeroid_pos1 = np.array([self.combined_ch1_Centeroids[::,0],self.combined_ch1_Centeroids[::,1]])
        centeroid_pos2 = np.array([self.combined_ch2_Centeroids[::,0],self.combined_ch2_Centeroids[::,1]])
        #all clusters
        #centeroid_n1 = len(self.ch1_centeroids_3D[::,0])
        #centeroid_n2 = len(self.ch2_centeroids_3D[::,0])
        #centeroid_pos1 = np.array([self.ch1_centeroids_3D[::,0],self.ch1_centeroids_3D[::,1]])
        #centeroid_pos2 = np.array([self.ch2_centeroids_3D[::,0],self.ch2_centeroids_3D[::,1]])
        self.centeroid_spots1 = [{'pos': centeroid_pos1[:,i], 'data': 1} for i in range(centeroid_n1)]
        self.centeroid_spots2 = [{'pos': centeroid_pos2[:,i], 'data': 1} for i in range(centeroid_n2)]
        self.centeroid_s1.addPoints(self.centeroid_spots1)
        self.centeroid_s2.addPoints(self.centeroid_spots2)
        self.w2.addItem(self.centeroid_s1)
        self.w2.addItem(self.centeroid_s2) 
        return


    def display2Ddata_noNoise(self):
        #make point data
        self.w2.clear()
        point_n1 = len(self.ch1PointsNoNoise_3D[::,0])
        point_n2 = len(self.ch2PointsNoNoise_3D[::,0])
        point_s1 = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120))
        point_s2 = pg.ScatterPlotItem(size=3, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))        
        point_pos1 = np.array([self.ch1PointsNoNoise_3D[::,0],self.ch1PointsNoNoise_3D[::,1]])
        point_pos2 = np.array([self.ch2PointsNoNoise_3D[::,0],self.ch2PointsNoNoise_3D[::,1]])
        point_spots1 = [{'pos': point_pos1[:,i], 'data': 1} for i in range(point_n1)]
        point_spots2 = [{'pos': point_pos2[:,i], 'data': 1} for i in range(point_n2)]
        point_s1.addPoints(point_spots1)
        point_s2.addPoints(point_spots2)
        self.w2.addItem(point_s1)
        self.w2.addItem(point_s2)
        return

    def display3Ddata_noNoise(self, toggle=True):
        dist = self.imv3D.opts['distance']
        elev = self.imv3D.opts['elevation'] 
        azim = self.imv3D.opts['azimuth']         
        self.imv3D.clear()
        self.imv3D.addArray(self.ch1PointsNoNoise_3D,color=QColor(0, 255, 0),size=2,name='ch1_pts')
        self.imv3D.addArray(self.ch2PointsNoNoise_3D,color=QColor(255, 0, 0),size=2,name='ch2_pts') 
        if toggle:
            self.imv3D.opts['distance'] = dist
            self.imv3D.opts['elevation'] = elev
            self.imv3D.opts['azimuth'] = azim
            return
        self.imv3D.opts['distance'] = 10000
        self.imv3D.orbit(-135,90)
        return

    def display3Dcentroids(self):
        dist = self.imv3D.opts['distance']
        elev = self.imv3D.opts['elevation'] 
        azim = self.imv3D.opts['azimuth']   
        self.imv3D.addArray(self.combined_ch1_Centeroids,color=QColor(255, 255, 255),size=10,name='ch1_cent')
        self.imv3D.addArray(self.combined_ch2_Centeroids,color=QColor(255, 255, 255),size=10,name='ch2_cent')
        self.imv3D.opts['distance'] = dist
        self.imv3D.opts['elevation'] = elev
        self.imv3D.opts['azimuth'] = azim        
        return
        

    def toggleNoise(self):
        if self.dataLoaded == False:
            print('No Data Loaded!')
            return
        if self.clustersGenerated == False:
            print('Generate Clusters!')
            return
        if self.dataDisplayed == 'original':
            self.display2Ddata_noNoise()
            self.display3Ddata_noNoise(toggle=True)
            if self.centroidsGenerated:
                self.display2Dcentroids()                  
                self.display3Dcentroids()            
            self.dataDisplayed = 'no noise'

        else:
            self.display2Ddata_allPoints()
            self.display3Ddata_allPoints(toggle=True)
            if self.centroidsGenerated:
                self.display2Dcentroids()  
                self.display3Dcentroids()
            self.dataDisplayed = 'original'            
        return

    def toggleCentroids(self):
        oldDisplay = self.dataDisplayed        
        if self.dataLoaded == False:
            print('No Data Loaded!')
            return
        if self.clustersGenerated == False:
            print('Generate Clusters!')
            return
        if self.centroidsGenerated == False:
            print('Generate Centroids!')
            return    
        
        
        return    

    def toggleClusters(self):
        if self.dataLoaded == False:
            print('No Data Loaded!')
            return
        if self.clustersGenerated == False:
            print('Generate Clusters!')
            return
        if self.dataDisplayed == 'original':
            self.display2Ddata_noNoise()
            self.display3Ddata_noNoise(toggle=True)
            self.dataDisplayed = 'no noise'
        else:
            self.display2Ddata_allPoints()
            self.display3Ddata_allPoints(toggle=True)
            self.dataDisplayed = 'original'            
        return  

clusterAnalysis = ClusterAnalysis()



### TESTING ####
def test():
    fileName = r"C:\Users\George\Desktop\batchTest\0_trial_1_superes_cropped.txt"
    fileName = r"C:\Users\George\Desktop\ianS-synapse\trial_1_superes_fullfield.txt"
    clusterAnalysis.viewerGUI()
    clusterAnalysis.open_file(fileName)
    clusterAnalysis.display2Ddata_allPoints()
    clusterAnalysis.display3Ddata_allPoints()
    clusterAnalysis.getClusters()  
    clusterAnalysis.getCentroids()
    clusterAnalysis.makeHulls()  
    clusterAnalysis.display2Dcentroids()    
    clusterAnalysis.display3Dcentroids()    
    
    
    return     
    
test() 