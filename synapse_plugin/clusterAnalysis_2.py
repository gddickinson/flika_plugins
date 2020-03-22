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


class DataWidget(pg.TableWidget):
    __name__ = "Data Widget"
    def __init__(self, viewer, sortable=False, **args):
        if 'name' in args:
            self.__name__ = args.pop('name')
        super(DataWidget, self).__init__(**args)
        self.viewer = viewer
        self.setSortingEnabled(sortable)
        self.itemSelectionChanged.connect(self.sendRowSignal)

    def sendRowSignal(self):
        items = self.selectedItems()     
        self.viewer.displayROI(items)
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
        
        #display options
        self.centroidSymbolSize = 10

    
        #data state
        self.dataLoaded = False
        self.ROI3D_Initiated = False
        self.dataDisplayed = 'original'
        self.clustersGenerated = False
        self.centroidsGenerated = False
        self.centroidsDisplayed = False      
        
        self.clusterIndex = []
    
        #cluster analysis options
        self.clusterAnaysisSelection = 'All Clusters'
        self.clusterType = '3D'
        
        self.All_ROIs_pointsList = []
        self.channelList = []
        
        self.multiThreadingFlag = False
        self.displayFlag = False
        
        #init data types
        self.ch1Points_3D = []
        self.ch2Points_3D = []

        #init roi flags
        self.ROI2D_flag = False


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
        #data
        self.dock1 = Dock("3D View", size=(600,600), closable=False)
        self.dock2 = Dock("2D View", size=(600,600), closable=False)
        self.dock3 = Dock("ROI 2D", size=(600,600), closable=True)
        #random
        self.dock4 = Dock("3D View - Random", size=(600,600), closable=False)
        self.dock5 = Dock("2D View - Random", size=(600,600), closable=False)
        self.dock6 = Dock("ROI 3D", size=(600,600), closable=False)        
        
        #buttons and results docks        
        self.dockButtons = Dock("Buttons",size=(1800,50))
        self.dockResults = Dock("Results",size=(1800,200))
        
        #add docks to area
        self.area.addDock(self.dock1, 'left')
        self.area.addDock(self.dock2, 'right', self.dock1)           
        self.area.addDock(self.dock3, 'right', self.dock2) 
        self.area.addDock(self.dock4, 'below', self.dock1)
        self.area.addDock(self.dock5, 'below', self.dock2)           
        self.area.addDock(self.dock6, 'above', self.dock3)    
        self.area.addDock(self.dockResults, 'bottom')              
        self.area.addDock(self.dockButtons, 'top',self.dockResults)   
           

        #initialise image widgets
        self.imv3D = Plot3DWidget()
        self.imv2D = pg.GraphicsLayoutWidget()
        self.imv3 = pg.GraphicsLayoutWidget()
        
        self.imv3D_rnd = Plot3DWidget()
        self.imv2D_rnd = pg.GraphicsLayoutWidget()
        self.imv3D_roi = Plot3DWidget()       

        #initialise table widget
        self.resultsTable = DataWidget(self,sortable=True)
        
        #add widgets to docks
        self.dock1.addWidget(self.imv3D, 0, 0, 6, 6)
        self.dock2.addWidget(self.imv2D)
        self.dock3.addWidget(self.imv3)
        self.dock4.addWidget(self.imv3D_rnd, 0, 0, 6, 6)
        self.dock5.addWidget(self.imv2D_rnd)
        self.dock6.addWidget(self.imv3D_roi)        

        self.dockResults.addWidget(self.resultsTable)

        
        #make sure data plots on top
        self.area.moveDock(self.dock1, 'above', self.dock4)  
        self.area.moveDock(self.dock2, 'above', self.dock5)  
        self.area.moveDock(self.dock6, 'above', self.dock3)  
                        
        #create plot windows
        self.w2 = self.imv2D.addPlot()
        self.w3 = self.imv3.addPlot()
        self.w4 = self.imv2D_rnd.addPlot()
       
        
        #add menu options
        self.menubar = self.win.menuBar()
        self.fileMenu1 = self.menubar.addMenu('&Display Options')        
        self.resetLayout = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Reset Layout')
        setMenuUp(self.resetLayout,self.fileMenu1,shortcut='Ctrl+R',statusTip='Reset Layout',connection=self.reset_layout)
        self.showTitles = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Show Titles')
        setMenuUp(self.showTitles,self.fileMenu1,shortcut='Ctrl+G',statusTip='Show Titles',connection=self.show_titles)
        self.hideTitles = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Hide Titles')
        setMenuUp(self.hideTitles,self.fileMenu1,shortcut='Ctrl+H',statusTip='Hide Titles',connection=self.hide_titles)                   

        self.fileMenu2 = self.menubar.addMenu('&File Options')
        self.menu_openFile = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Open File')
        setMenuUp(self.menu_openFile,self.fileMenu2,shortcut='Ctrl+O',statusTip='Open File',connection=lambda f: self.openFileAndDisplay())

        self.fileMenu3 = self.menubar.addMenu('&Cluster Options')
        self.clusterMenu = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Cluster Option Window')
        setMenuUp(self.clusterMenu,self.fileMenu3,shortcut='Ctrl+o',statusTip='Cluster Option Window',connection=self.openClusterOptionWin)



        self.dockList = [self.dock1, self.dock2, self.dock3]
        
        ##QUICK BUTTON BAR##
        self.dockButtons.hideTitleBar()
        
        self.button_getClusters = QtWidgets.QPushButton("Get clusters") 
        self.dockButtons.addWidget(self.button_getClusters,0,0)
        self.button_getClusters.clicked.connect(self.runAnalysis)   
        
        self.button_ToggleNoise = QtWidgets.QPushButton("Toggle Noise") 
        self.dockButtons.addWidget(self.button_ToggleNoise,0,1)
        self.button_ToggleNoise.clicked.connect(self.toggleNoise)

        self.button_ToggleClusters = QtWidgets.QPushButton("Toggle Clusters") 
        self.dockButtons.addWidget(self.button_ToggleClusters,0,2)
        self.button_ToggleClusters.clicked.connect(self.toggleClusters)
        
        self.button_ToggleCentroids = QtWidgets.QPushButton("Toggle Centroids") 
        self.dockButtons.addWidget(self.button_ToggleCentroids,0,3)
        self.button_ToggleCentroids.clicked.connect(self.toggleCentroids)        
        
        self.state = self.area.saveState()        
        self.win.show()        

        return


    def openFileAndDisplay(self):
        #open txt file and setup data channels
        self.open_file()
        #display data
        self.display2Ddata_allPoints()
        self.display3Ddata_allPoints()
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

    def openClusterOptionWin(self):
        self.clusterOptionDialog = ClusterOptions_win(self)
        self.clusterOptionDialog.show()
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
            pos = self.imv3D.cameraPosition() 
            dist = self.imv3D.opts['distance']
            elevation =self.imv3D.opts['elevation']
            azimuth = self.imv3D.opts['azimuth']
            self.imv3D.clear()
        self.imv3D.addArray(self.ch1Points_3D,color=QColor(0, 255, 0),size=2,name='ch1_pts_all')
        self.imv3D.addArray(self.ch2Points_3D,color=QColor(255, 0, 0),size=2,name='ch2_pts_all') 
        if toggle:
            self.imv3D.setCameraPosition(pos=pos,distance = dist, elevation =elevation, azimuth=azimuth)            
            return
        self.imv3D.opts['distance'] = 20000
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
        self.combinedHulls, self.combinedPoints, self.combined_ch1_Centeroids, self.combined_ch2_Centeroids = combineClosestHulls(ch1_hulls,ch1_centeroids,self.ch1_groupPoints,ch2_hulls,ch2_centeroids,self.ch2_groupPoints, self.maxDistance)
        
        #get new hulls for combined points
        self.newHulls = self.getHulls2(self.combinedPoints)              
        t.timeReport('hulls created')


    def createROIFromHull(self,points, hull):
        #t = Timer()
        #t.start()
        '''add roi to display from hull points'''
        #make points list
        pointsList = []     
        for simplex in hull:
            pointsList.append((points[simplex][0])) 
        for simplex in hull:            
            pointsList.append((points[simplex][1])) 

        #order points list
        pointsList = order_points(pointsList)

        #convert list to np array
        pointsList = np.array(pointsList)
               
        #add create ROIs from points
        #self.plotWidget.getViewBox().createROIFromPoints(pointsList)
        
        #add points to All_ROI_pointsList
        #self.All_ROIs_pointsList.append(pointsList)
        self.All_ROIs_pointsList.append(points)
        
        
        #make channel list for all points
        self.makeChannelList()

        #t.timeReport('ROI made')
        return


    def makeChannelList(self):
        self.channelList = []
        ch1_pts = self.Channels[0].getPoints(z=True).tolist() #cast as list to ensure logic test works
        #ch2_pts = self.Channels[1].getPoints()
        for roi in self.All_ROIs_pointsList:
            roiList = []
            for pts in roi:
                if list(pts) in ch1_pts:
                    roiList.append(self.Channels[0].__name__)
                else:
                    roiList.append(self.Channels[1].__name__) 
            self.channelList.append(np.array(roiList))
        return

    def makeROIs(self):
        #single thread
        for i in range(len(self.combinedHulls)):
            self.createROIFromHull(self.combinedPoints[i],self.newHulls[i]) ### THIS IS SLOW! ###
            print('\r', 'creating rois: {:0.2f}'.format((i/len(self.combinedHulls))*100),'%', end='\r', flush=True)
        return

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
        self.centroidsDisplayed = True
        
        #add text labels
        ## Create text object, use HTML tags to specify color/size
        for i in range(len(self.centeroid_spots1)):
            text = pg.TextItem(str(i), anchor=(0,0))#, angle=45, border='w', fill=(0, 0, 255, 100))
            text.setPos(self.centeroid_spots1[i]['pos'][0],self.centeroid_spots1[i]['pos'][1])
            self.w2.addItem(text)        
        self.centroidLabelsDisplayed = True        
        return


    def display2Ddata_noNoise(self):
        #make point data
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
        pos = self.imv3D.cameraPosition()  
        dist = self.imv3D.opts['distance']
        elevation =self.imv3D.opts['elevation']
        azimuth = self.imv3D.opts['azimuth']
        self.imv3D.clear()
        self.imv3D.addArray(self.ch1PointsNoNoise_3D,color=QColor(0, 255, 0),size=2,name='ch1_pts_noNoise')
        self.imv3D.addArray(self.ch2PointsNoNoise_3D,color=QColor(255, 0, 0),size=2,name='ch2_pts_noNoise') 
        if toggle:
            self.imv3D.setCameraPosition(pos=pos,distance=dist, elevation =elevation, azimuth=azimuth)   
            return

        return

    def display3Dcentroids(self):
        pos = self.imv3D.cameraPosition() 
        dist = self.imv3D.opts['distance']
        elevation =self.imv3D.opts['elevation']
        azimuth = self.imv3D.opts['azimuth']
        self.imv3D.addArray(self.combined_ch1_Centeroids,color=QColor(255, 255, 255),size=10,name='ch1_cent')
        self.imv3D.addArray(self.combined_ch2_Centeroids,color=QColor(255, 255, 255),size=10,name='ch2_cent')
        self.imv3D.setCameraPosition(pos=pos,distance=dist, elevation =elevation, azimuth=azimuth)       
        self.centroidsDisplayed = True
        return

    def removeCentroids(self):
        self.imv3D.deleteItem('ch1_cent')
        self.imv3D.deleteItem('ch2_cent')         
        return        


    def clear2Ddisplay(self):
        self.w2.clear()
        return        


    def display2Dcentroids_rnd(self):
        #make centeroid data
        self.centeroid_s1_rnd = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.centeroid_s2_rnd = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        #combined clusters
        centeroid_n1 = len(self.ch1_random[::,0])
        centeroid_n2 = len(self.ch2_random[::,0])
        centeroid_pos1 = np.array([self.ch1_random[::,0],self.ch1_random[::,1]])
        centeroid_pos2 = np.array([self.ch2_random[::,0],self.ch2_random[::,1]])

        self.centeroid_spots1_rnd = [{'pos': centeroid_pos1[:,i], 'data': 1} for i in range(centeroid_n1)]
        self.centeroid_spots2_rnd = [{'pos': centeroid_pos2[:,i], 'data': 1} for i in range(centeroid_n2)]
        self.centeroid_s1_rnd.addPoints(self.centeroid_spots1_rnd)
        self.centeroid_s2_rnd.addPoints(self.centeroid_spots2_rnd)
        self.w4.addItem(self.centeroid_s1_rnd)
        self.w4.addItem(self.centeroid_s2_rnd) 
        self.centroidsDisplayed = True
        return



    def display3Dcentroids_rnd(self):
        pos = self.imv3D_rnd.cameraPosition() 
        dist = self.imv3D_rnd.opts['distance']
        elevation =self.imv3D_rnd.opts['elevation']
        azimuth = self.imv3D_rnd.opts['azimuth']
        self.imv3D_rnd.addArray(self.ch1_random,color=QColor(255, 255, 255),size=10,name='ch1_cent')
        self.imv3D_rnd.addArray(self.ch2_random,color=QColor(255, 255, 255),size=10,name='ch2_cent')
        self.imv3D_rnd.setCameraPosition(pos=pos,distance=dist, elevation =elevation, azimuth=azimuth)       
        self.centroidsDisplayed = True
        return

    def randRGB(self):
        r=0
        g=0
        b=0
        while r==g==b==0:
        #no black
            r = int(random.random() * 256)
            g = int(random.random() * 256)
            b = int(random.random() * 256)          
        return [r,g,b]

    def displayROIpoints_2D(self, roiNum = 'ALL'):
        #add roi points
        #ch1_pts = np.vstack(self.AllPoints_ch1)
        #ch2_pts = np.vstack(self.AllPoints_ch2)

        if roiNum == 'ALL':
        
            for i in range(len(self.AllPoints_ch1)):                
                colour = self.randRGB()                   
                ch1_pts = self.AllPoints_ch1[i]
                ch2_pts = self.AllPoints_ch2[i]                    
                roi_point_n1 = len(ch1_pts[::,0])
                roi_point_n2 = len(ch2_pts[::,0])
                roi_point_s1 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(colour[0], colour[1], colour[2], 120))
                roi_point_s2 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(colour[0], colour[1], colour[2], 120))        
                roi_point_pos1 = np.array([ch1_pts[::,0],ch1_pts[::,1]])
                roi_point_pos2 = np.array([ch2_pts[::,0],ch2_pts[::,1]])
                roi_point_spots1 = [{'pos': roi_point_pos1[:,i], 'data': 1} for i in range(roi_point_n1)]
                roi_point_spots2 = [{'pos': roi_point_pos2[:,i], 'data': 1} for i in range(roi_point_n2)]
                roi_point_s1.addPoints(roi_point_spots1)
                roi_point_s2.addPoints(roi_point_spots2)
                self.w3.addItem(roi_point_s1)
                self.w3.addItem(roi_point_s2)
                return
        else:
                self.w3.clear()               
                ch1_pts = self.AllPoints_ch1[roiNum]
                ch2_pts = self.AllPoints_ch2[roiNum]                    
                roi_point_n1 = len(ch1_pts[::,0])
                roi_point_n2 = len(ch2_pts[::,0])
                roi_point_s1 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0, 120))
                roi_point_s2 = pg.ScatterPlotItem(size=5, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))        
                roi_point_pos1 = np.array([ch1_pts[::,0],ch1_pts[::,1]])
                roi_point_pos2 = np.array([ch2_pts[::,0],ch2_pts[::,1]])
                roi_point_spots1 = [{'pos': roi_point_pos1[:,i], 'data': 1} for i in range(roi_point_n1)]
                roi_point_spots2 = [{'pos': roi_point_pos2[:,i], 'data': 1} for i in range(roi_point_n2)]
                roi_point_s1.addPoints(roi_point_spots1)
                roi_point_s2.addPoints(roi_point_spots2)
                self.w3.addItem(roi_point_s1)
                self.w3.addItem(roi_point_s2) 
                self.w3.autoRange(padding=0.4)
        return

    def display2Dcentroids_roi(self, roiNum = 'ALL'):
        #make centeroid data
        self.centeroid_s1_roi = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.centeroid_s2_roi = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))        
        
        if roiNum == 'ALL':
            #combined clusters
            centeroid_n1 = len(self.combined_ch1_Centeroids[::,0])
            centeroid_n2 = len(self.combined_ch2_Centeroids[::,0])
            centeroid_pos1 = np.array([self.combined_ch1_Centeroids[::,0],self.combined_ch1_Centeroids[::,1]])
            centeroid_pos2 = np.array([self.combined_ch2_Centeroids[::,0],self.combined_ch2_Centeroids[::,1]])
    
            self.centeroid_spots1_roi = [{'pos': centeroid_pos1[:,i], 'data': 1} for i in range(centeroid_n1)]
            self.centeroid_spots2_roi = [{'pos': centeroid_pos2[:,i], 'data': 1} for i in range(centeroid_n2)]
            self.centeroid_s1_roi.addPoints(self.centeroid_spots1_roi)
            self.centeroid_s2_roi.addPoints(self.centeroid_spots2_roi)
            self.w3.addItem(self.centeroid_s1_roi)
            self.w3.addItem(self.centeroid_s2_roi) 
            return
        else:
            #combined clusters
            centeroid_pos1 = np.array([self.combined_ch1_Centeroids[roiNum][0],self.combined_ch1_Centeroids[roiNum]][1])
            centeroid_pos2 = np.array([self.combined_ch2_Centeroids[roiNum][0],self.combined_ch2_Centeroids[roiNum]][1])    
            self.centeroid_spots1_roi = [{'pos': centeroid_pos1, 'data': 1}]
            self.centeroid_spots2_roi = [{'pos': centeroid_pos2, 'data': 1}]
            self.centeroid_s1_roi.addPoints(self.centeroid_spots1_roi)
            self.centeroid_s2_roi.addPoints(self.centeroid_spots2_roi)
            self.w3.addItem(self.centeroid_s1_roi)
            self.w3.addItem(self.centeroid_s2_roi)             
        return


    def display3Ddata_roi(self, roiNum):
        if roiNum == 'ALL':
            ch1_pts = np.vstack(self.AllPoints_ch1)
            ch2_pts = np.vstack(self.AllPoints_ch2)
            self.imv3D_roi.addArray(ch1_pts,color=QColor(0, 255, 0),size=2,name='ch1_pts')
            self.imv3D_roi.addArray(ch2_pts,color=QColor(255, 0, 0),size=2,name='ch2_pts') 
            return
        else:
            self.imv3D_roi.clear()
            ch1_pts = self.AllPoints_ch1[roiNum]
            ch2_pts = self.AllPoints_ch2[roiNum]
            self.imv3D_roi.addArray(ch1_pts,color=QColor(0, 255, 0),size=2,name='ch1_pts')
            self.imv3D_roi.addArray(ch2_pts,color=QColor(255, 0, 0),size=2,name='ch2_pts')             
        return

    def display3Dcentroids_roi(self, roiNum):
        if roiNum == 'ALL':        
            pos = self.imv3D_roi.cameraPosition() 
            dist = self.imv3D_roi.opts['distance']
            elevation =self.imv3D_roi.opts['elevation']
            azimuth = self.imv3D_roi.opts['azimuth']
            self.imv3D_roi.addArray(self.combined_ch1_Centeroids,color=QColor(255, 255, 255),size=10,name='ch1_cent')
            self.imv3D_roi.addArray(self.combined_ch2_Centeroids,color=QColor(255, 255, 255),size=10,name='ch2_cent')
            self.imv3D_roi.setCameraPosition(pos=pos,distance=dist, elevation =elevation, azimuth=azimuth)       
            return
        else:
            print('display roi: ',roiNum)
            pos = self.imv3D_roi.cameraPosition() 
            dist = self.imv3D_roi.opts['distance']
            elevation =self.imv3D_roi.opts['elevation']
            azimuth = self.imv3D_roi.opts['azimuth']
            
            item1 = np.array([np.array(self.combined_ch1_Centeroids[roiNum])])
            item2 = np.array([np.array(self.combined_ch2_Centeroids[roiNum])])
            
            print(item1)
            
            self.imv3D_roi.addArray(item1,color=QColor(255, 255, 255),size=10,name='ch1_cent')
            self.imv3D_roi.addArray(item2,color=QColor(255, 255, 255),size=10,name='ch2_cent')
            self.imv3D_roi.setCameraPosition(pos=pos,distance=dist, elevation =elevation, azimuth=azimuth)             
        return


    def analyze_roi(self, roi, channelList, roiIndex):
        '''analyse roi pts'''
        #channels = [self.Channels[0],self.Channels[1]]
        
        ch1_pts = roi[channelList == self.Channels[0].__name__]
        ch2_pts = roi[channelList == self.Channels[1].__name__]  

        self.AllPoints_ch1.append(np.array(ch1_pts))
        self.AllPoints_ch2.append(np.array(ch2_pts))
                        
        roi_data = OrderedDict([('ROI #', roiIndex), \
                                 ('Mean Distance (%s)' % self.unit_prefixes[self.unit], 0), \
                                 ('%s N' % self.Channels[0].__name__, 0), \
                                 ('%s N' % self.Channels[1].__name__, 0), \
                                 ('%s Volume (%s^3)' % (self.Channels[0].__name__, self.unit_prefixes[self.unit]), 0), \
                                 ('%s Volume (%s^3)' % (self.Channels[1].__name__, self.unit_prefixes[self.unit]), 0)])


        roi_data['%s N' % self.Channels[0].__name__] = len(ch1_pts)
        roi_data['%s N' % self.Channels[1].__name__] = len(ch2_pts)        

        try:        
            if len(ch1_pts) >= 4:
                roi_data['%s Volume (%s^3)' % (self.Channels[0].__name__, self.unit_prefixes[self.unit])] = round(convex_volume(ch1_pts), 2)
    
            else:
                roi_data['%s Volume (%s^3)' % (self.Channels[0].__name__, self.unit_prefixes[self.unit])] = 0

        except:
                roi_data['%s Volume (%s^3)' % (self.Channels[0].__name__, self.unit_prefixes[self.unit])] = 0


        try:
            if len(ch2_pts) >= 4:
    
                roi_data['%s Volume (%s^3)' % (self.Channels[1].__name__, self.unit_prefixes[self.unit])] = round(convex_volume(ch2_pts), 2)
    
            else:
                roi_data['%s Volume (%s^3)' % (self.Channels[1].__name__, self.unit_prefixes[self.unit])] = 0
    
        except:
                roi_data['%s Volume (%s^3)' % (self.Channels[1].__name__, self.unit_prefixes[self.unit])] = 0


            #g.m.statusBar().showMessage('Cannot get Volume of %s in roi %d with %d points' % (ch.__name__, roi.id, ch.getCount())) 
        
        
        roi_data['Mean Distance (%s)' % self.unit_prefixes[self.unit]] = round(np.linalg.norm(np.average(ch1_pts, 0) - np.average(ch2_pts, 0)), 2)
        roi_data['%s Centeroid' % (self.Channels[0].__name__)] = np.average(ch1_pts, 0)
        roi_data['%s Centeroid' % (self.Channels[1].__name__)] = np.average(ch2_pts, 0)


        height = round(max((max(ch1_pts[::,0]),max(ch2_pts[::,0]))) - min((min(ch1_pts[::,0]),min(ch2_pts[::,0]))), 2)  
        width = round(max((max(ch1_pts[::,1]),max(ch2_pts[::,1]))) - min((min(ch1_pts[::,1]),min(ch2_pts[::,1]))) , 2)
        depth = round(max((max(ch1_pts[::,2]),max(ch2_pts[::,2]))) - min((min(ch1_pts[::,2]),min(ch2_pts[::,2]))), 2)
        
        roi_data['center'] = (np.average(ch1_pts, 0) + np.average(ch2_pts, 0))/2
        roi_data['height'] = height  
        roi_data['width'] =  width     
        roi_data['depth'] =  depth   
        roi_data['box volume'] = height*width*depth     
        return roi_data


    def makeROI_DF(self):
        '''pass each roi to analyze_roi(), compile results in table'''
        dictList = []
        #reset AllPoints_ch lists
        self.AllPoints_ch1 = []
        self.AllPoints_ch2 = []
        for i in range(len(self.All_ROIs_pointsList)):
            roi_data = self.analyze_roi(self.All_ROIs_pointsList[i],self.channelList[i],i)
            dictList.append(roi_data)
            print('\r', 'analysing rois: {:0.2f}'.format((i/len(self.All_ROIs_pointsList))*100),'%', end='\r', flush=True)
        #make df
        self.roiAnalysisDF = pd.DataFrame(dictList)
        #print(self.roiAnalysisDF.head())
        return
    
    def toggleNoise(self):
        if self.dataLoaded == False:
            print('No Data Loaded!')
            return
        if self.clustersGenerated == False:
            print('Generate Clusters!')
            return
        self.clear2Ddisplay()
        if self.dataDisplayed == 'original':
            self.display2Ddata_noNoise()
            self.display3Ddata_noNoise(toggle=True)
            if self.centroidsDisplayed:
                self.display2Dcentroids()                  
                self.display3Dcentroids()            
            self.dataDisplayed = 'no noise'

        else:
            self.display2Ddata_allPoints()
            self.display3Ddata_allPoints(toggle=True)
            if self.centroidsDisplayed:
                self.display2Dcentroids()  
                self.display3Dcentroids()
            self.dataDisplayed = 'original'            
        return

    def toggleCentroids(self):      
        if self.dataLoaded == False:
            print('No Data Loaded!')
            return
        if self.clustersGenerated == False:
            print('Generate Clusters!')
            return
        if self.centroidsGenerated == False:
            print('Generate Centroids!')
            return    
        self.clear2Ddisplay()
        #print(self.imv3D.cameraPosition())

        if self.dataDisplayed == 'no noise':
            self.display2Ddata_noNoise()
            self.display3Ddata_noNoise(toggle=True)
            if self.centroidsDisplayed == False:
                self.display2Dcentroids()                  
                self.display3Dcentroids()                 
                self.centroidsDisplayed = True
            else:
                self.removeCentroids()
                self.centroidsDisplayed = False
                
        else:
            self.display2Ddata_allPoints()
            self.display3Ddata_allPoints(toggle=True)
            if self.centroidsDisplayed == False:
                self.display2Dcentroids()  
                self.display3Dcentroids()
                self.centroidsDisplayed = True
            else:
                self.removeCentroids()
                self.centroidsDisplayed = False

        #print(self.imv3D.cameraPosition())
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


    def getRandomXYZ(self, minX, minY, minZ, maxX, maxY, maxZ, n):
        randomList = []
        while len(randomList) < n:
            p = np.array([random.uniform(minX,maxY),
                 random.uniform(minY,maxY),
                 random.uniform(minZ,maxZ)])
            randomList.append(p)       
        return np.array(randomList)

    def getNearestNeighbors(self,train,test,k=1):
        tree = KDTree(train, leaf_size=5)   
        dist, ind = tree.query(test, k=k)         
        return dist.reshape(np.size(dist),)

    def randomPointAnalysis(self):
        '''generate random points distributed in same dimensions as data'''       
        self.dist_clusters = self.getNearestNeighbors(self.combined_ch1_Centeroids,self.combined_ch2_Centeroids)
        #print(self.dist_clusters)        
        #print(min(self.data['Xc']), min(self.data['Yc']), min(self.data['Zc']))
        #print(max(self.data['Xc']), max(self.data['Yc']), max(self.data['Zc']))
        self.ch1_random = self.getRandomXYZ(min(self.data['Xc']),
                                       min(self.data['Yc']),
                                       min(self.data['Zc']),
                                       max(self.data['Xc']),
                                       max(self.data['Yc']),
                                       max(self.data['Zc']), len(self.combined_ch1_Centeroids))


        self.ch2_random = self.getRandomXYZ(min(self.data['Xc']),
                                       min(self.data['Yc']),
                                       min(self.data['Zc']),
                                       max(self.data['Xc']),
                                       max(self.data['Yc']),
                                       max(self.data['Zc']), len(self.combined_ch2_Centeroids))


        self.dist_random = self.getNearestNeighbors(self.ch1_random,self.ch2_random)

        self.distAll_clusters = self.getNearestNeighbors(self.combined_ch1_Centeroids,self.combined_ch2_Centeroids, k=len(self.combined_ch2_Centeroids))
        self.distAll_random = self.getNearestNeighbors(self.ch1_random,self.ch2_random, k=len(self.ch1_random))
        
        return


    def printStats(self):
        '''print stats'''
        print('----------------------------------------------')
        print(self.clusterAnaysisSelection)        
        print('----------------------------------------------')
        print('Channel 1: Number of clusters: ', str(len(self.combined_ch1_Centeroids)))
        print('Channel 2: Number of clusters: ', str(len(self.combined_ch2_Centeroids)))        
        print('Number of nearest neighbor distances:', str(np.size(self.dist_clusters)))
        print('Mean nearest neighbor distance:', str(round(np.mean(self.dist_clusters),2)))
        print('StDev nearest neighbor distance:', str(round(np.std(self.dist_clusters),2)))        
        print('Number of All distances:', str(np.size(self.distAll_clusters)))
        print('Mean All distance:', str(round(np.mean(self.distAll_clusters),2)))
        print('StDev All distance:', str(round(np.std(self.distAll_clusters),2)))       
        print('----------------------------------------------')
        print('Random 1: Number of clusters: ', str(len(self.ch1_random)))
        print('Random 2: Number of clusters: ', str(len(self.ch2_random)))        
        print('Random: Number of nearest neighbor distances:', str(np.size(self.dist_random)))
        print('Random: Mean nearest neighbor distance:', str(round(np.mean(self.dist_random),2)))
        print('Random: StDev nearest neighbor distance:', str(round(np.std(self.dist_random),2)))       
        print('Random: Number of All distances:', str(np.size(self.distAll_random)))
        print('Random: Mean All distance:', str(round(np.mean(self.distAll_random),2)))  
        print('Random: Stdev All distance:', str(round(np.std(self.distAll_random),2)))          
        print('----------------------------------------------')
        return


    def saveStats(self ,savePath='',fileName=''):
        '''save clustering stats as csv'''
        d= {
            'Channel 1: Number of clusters': (len(self.combined_ch1_Centeroids)),
            'Channel 2: Number of clusters': (len(self.combined_ch2_Centeroids)),
            'Channel 1: Number of noise points': self.ch1_numNoise,
            'Channel 2: Number of noise points': self.ch2_numNoise,
            'Number of nearest neighbor distances': (np.size(self.dist_clusters)),
            'Mean nearest neighbor distance': (np.mean(self.dist_clusters)),
            'StDev nearest neighbor distance': (np.std(self.dist_clusters)),       
            'Number of All distances': (np.size(self.distAll_clusters)),
            'Mean All distance': (np.mean(self.distAll_clusters)),
            'StDev All distance': (np.std(self.distAll_clusters)),      
            'Random 1: Number of clusters': (len(self.ch1_random)),
            'Random 2: Number of clusters': (len(self.ch2_random)),      
            'Random: Number of nearest neighbor distances': (np.size(self.dist_random)),
            'Random: Mean nearest neighbor distance': (np.mean(self.dist_random)),
            'Random: StDev nearest neighbor distance': (np.std(self.dist_random)),      
            'Random: Number of All distances': (np.size(self.distAll_random)),
            'Random: Mean All distance': (np.mean(self.distAll_random)),
            'Random: Stdev All distance': (np.std(self.distAll_random)) 
                }
        
        statsDF = pd.DataFrame(data=d,index=[0])
        saveName = os.path.join(savePath, fileName + '_stats.csv')
        statsDF.to_csv(saveName)
        print('stats saved as:', saveName)
        return

    def saveResults(self, savePath='',fileName=''):
        '''save centeroids and distances'''
        d1 = {'clusters_nearest':self.dist_clusters,'random_nearest':self.dist_random}
        d2 = {'clusters_All':self.distAll_clusters,'random_All':self.distAll_random}
        d3 = {'ch1_centeroids_x':self.combined_ch1_Centeroids[::,0],
              'ch1_centeroids_y':self.combined_ch1_Centeroids[::,1],
              'ch1_centeroids_z':self.combined_ch1_Centeroids[::,2]}
        
        d4 = {'ch2_centeroids_x':self.combined_ch2_Centeroids[::,0],
              'ch2_centeroids_y':self.combined_ch2_Centeroids[::,1],
              'ch2_centeroids_z':self.combined_ch2_Centeroids[::,2]}
        
        d5 = {'ch1_centeroids_rnd_x':self.ch1_random[::,0],
              'ch1_centeroids_rnd_y':self.ch1_random[::,1],
              'ch1_centeroids_rnd_z':self.ch1_random[::,2]}
              
        d6 = {'ch2_centeroids_rnd_x':self.ch2_random[::,0],
              'ch2_centeroids_rnd_y':self.ch2_random[::,1],
              'ch2_centeroids_rnd_z':self.ch2_random[::,2]}
        
        
        nearestNeighborDF = pd.DataFrame(data=d1)
        allNeighborDF = pd.DataFrame(data=d2)   
        ch1_centeroids_clusters_DF = pd.DataFrame(data=d3)  
        ch2_centeroids_clusters_DF = pd.DataFrame(data=d4)                 
        ch1_centeroids_random_DF = pd.DataFrame(data=d5)   
        ch2_centeroids_random_DF = pd.DataFrame(data=d6) 
        
        saveName1 = os.path.join(savePath, fileName + '_clusterAnalysis_nearestNeighbors.csv')
        saveName2 = os.path.join(savePath, fileName + '_clusterAnalysis_AllNeighbors.csv')  
        saveName3 = os.path.join(savePath, fileName + '_ch1_clusters_centeroids.csv')
        saveName4 = os.path.join(savePath, fileName + '_ch2_clusters_centeroids.csv')   
        saveName5 = os.path.join(savePath, fileName + '_ch1_random_centeroids.csv')
        saveName6 = os.path.join(savePath, fileName + '_ch2_random_centeroids.csv')   
        
        nearestNeighborDF.to_csv(saveName1)
        allNeighborDF.to_csv(saveName2)
        ch1_centeroids_clusters_DF.to_csv(saveName3) 
        ch2_centeroids_clusters_DF.to_csv(saveName4)        
        ch1_centeroids_random_DF .to_csv(saveName5)         
        ch2_centeroids_random_DF .to_csv(saveName6)   

       
        print('nearest neighbor distances saved as:', saveName1)
        print('all neighbor distances saved as:', saveName2)
        print('ch1_cluster centeroids saved as:', saveName3) 
        print('ch1_cluster centeroids saved as:', saveName4)               
        print('ch1_random centeroids saved as:', saveName5)         
        print('ch2_random centeroids saved as:', saveName6)  



    def displayROIresults(self):
        #self.write_df_to_qtable(self.roiAnalysisDF,self.resultsTable)
        data = self.roiAnalysisDF.to_records(index=False)
        self.resultsTable.setData(data)                
        return

    def displayROI(self, items):
        height = float(items[9].text())
        width = float(items[10].text())
        roi_number = int(items[0].text())
        x = str(items[8].text()).split(' ')[1]
        y = str(items[8].text()).split(' ')[3]        
      
        #print('#: ', str(roi_number), 'x: ', x, 'y: ', y, 'width: ', str(width), 'height: ', str(height))
        
        startX = float(x)-(height/2)
        startY = float(y)-(width/2)

        if self.ROI2D_flag == False:
            self.ROI2D_pen = mkPen('r', width=3,style=QtCore.Qt.DashLine)
            self.ROI_2D = pg.RectROI([startX, startY], [height,width], pen=self.ROI2D_pen, movable=False) 
            #self.ROI_2D = pg.CircleROI([int(width/2),int(self.height)], [ch1_x , ch1_y], pen=(4,9))
            handles = self.ROI_2D.getHandles()
            handles[0].disconnectROI(self.ROI_2D)
            handles[0].pen = mkPen(None)
            #self.ROI_2D.setState(self.ROIState)            
            #self.w3.addItem(self.ROI_2D)
            #add ROI around 2D selection
            self.w2.addItem(self.ROI_2D)   
            self.ROI2D_flag = True
            #display ROI selection only - 2D
            self.displayROIpoints_2D(roi_number)
            self.display2Dcentroids_roi(roi_number)            
            #display ROI selection only - 3D  
            self.display3Ddata_roi(roi_number)
            self.display3Dcentroids_roi(roi_number)
            
        else:
            #update 2D ROI position
            self.ROI_2D.setPos([startX, startY])
            self.ROI_2D.setSize([height,width])
            #update 2D ROI selection display
            self.displayROIpoints_2D(roi_number)
            self.display2Dcentroids_roi(roi_number)            
            #update 3D ROI selection display
            self.display3Ddata_roi(roi_number)
            self.display3Dcentroids_roi(roi_number)
            
        #set window focus
        #self.w3.view.setRect(self.ROI_2D.mapRectFromView())
        return



    def runAnalysis(self):
        '''performs clustering, gets centeroids combines centeroids and displays result'''
        if self.dataLoaded == False:
            print('No Data Loaded!')
            return
        if self.clustersGenerated == True:
            print('Clusters already generated!') #TODO update to clear cluster/centroid data
            return
        self.getClusters()  
        self.getCentroids()
        self.makeHulls()  
        self.display2Dcentroids()    
        self.display3Dcentroids()    
        self.randomPointAnalysis()
        self.display2Dcentroids_rnd()   
        self.display3Dcentroids_rnd() 
        self.printStats()
        self.makeROIs()
        self.makeROI_DF()
        self.displayROIresults()

                       
        return
        
        
clusterAnalysis = ClusterAnalysis()

class ClusterOptions_win(QtWidgets.QDialog):
    def __init__(self, viewerInstance, parent = None):
        super(ClusterOptions_win, self).__init__(parent)

        self.viewer = viewerInstance
        self.eps  = self.viewer.eps
        self.min_samples = self.viewer.min_samples
        self.maxDistance = self.viewer.maxDistance
        self.unitPerPixel = self.viewer.unitPerPixel
        self.centroidSymbolSize = self.viewer.centroidSymbolSize
#        self.multiThreadingFlag = self.viewer.multiThreadingFlag
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 200

        #labels
        self.clusterTitle = QtWidgets.QLabel("----- Cluster Parameters -----") 
        self.label_eps = QtWidgets.QLabel("max distance between points:") 
        self.label_minSamples = QtWidgets.QLabel("minimum number of points:") 
        self.label_maxDistance = QtWidgets.QLabel("max distance between clusters:") 
        self.displayTitle = QtWidgets.QLabel("----- Display Parameters -----") 
        self.label_unitPerPixel = QtWidgets.QLabel("nanometers/pixel:") 
        self.label_centroidSymbolSize = QtWidgets.QLabel("centroid symbol size:")  
        
        self.analysisTitle = QtWidgets.QLabel("----- Cluster Analysis -----") 
        self.label_analysis = QtWidgets.QLabel("Clusters to analyse: ")         
        
#        self.multiThreadTitle = QtWidgets.QLabel("----- Multi-Threading -----") 
#        self.label_multiThread = QtWidgets.QLabel("Multi-Threading On: ")        
        
        #self.label_displayPlot = QtWidgets.QLabel("show plot")         

        #spinboxes
        self.epsBox = QtWidgets.QSpinBox()
        self.epsBox.setRange(0,10000)
        self.epsBox.setValue(self.eps)
        self.minSampleBox = QtWidgets.QSpinBox()    
        self.minSampleBox.setRange(0,10000)
        self.minSampleBox.setValue(self.min_samples)
        self.maxDistanceBox = QtWidgets.QSpinBox()    
        self.maxDistanceBox.setRange(0,10000)
        self.maxDistanceBox.setValue(self.maxDistance)    
        self.unitPerPixelBox = QtWidgets.QSpinBox()    
        self.unitPerPixelBox.setRange(0,1000)
        self.unitPerPixelBox.setValue(self.unitPerPixel)  
        self.centroidSymbolSizeBox = QtWidgets.QSpinBox()    
        self.centroidSymbolSizeBox.setRange(0,1000)
        self.centroidSymbolSizeBox.setValue(self.centroidSymbolSize)          

        #combobox
        self.analysis_Box = QtWidgets.QComboBox()
        self.analysis_Box.addItems(["All Clusters", "Paired Clusters"])
        
#        #tickbox
#        self.multiThread_checkbox = CheckBox()
#        self.multiThread_checkbox.setChecked(self.multiThreadingFlag)
#        self.multiThread_checkbox.stateChanged.connect(self.multiThreadClicked)


        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.clusterTitle, 0, 0, 1, 2)        
        layout.addWidget(self.label_eps, 1, 0)
        layout.addWidget(self.epsBox, 1, 1)       
        layout.addWidget(self.label_minSamples, 2, 0)        
        layout.addWidget(self.minSampleBox, 2, 1)     
        layout.addWidget(self.label_maxDistance, 3, 0)        
        layout.addWidget(self.maxDistanceBox, 3, 1)
        layout.addWidget(self.displayTitle, 4, 0, 1, 2)          
        layout.addWidget(self.label_unitPerPixel, 5, 0)  
        layout.addWidget(self.unitPerPixelBox, 5, 1) 
        layout.addWidget(self.label_centroidSymbolSize, 6, 0)  
        layout.addWidget(self.centroidSymbolSizeBox, 6, 1) 
        
        layout.addWidget(self.analysisTitle, 8, 0, 1, 2)  
        layout.addWidget(self.label_analysis, 9, 0)  
        layout.addWidget(self.analysis_Box, 9, 1)     
        
#        layout.addWidget(self.multiThreadTitle, 10, 0, 1, 2)  
#        layout.addWidget(self.label_multiThread, 11, 0)  
#        layout.addWidget(self.multiThread_checkbox, 11, 1)          

        
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Clustering Options")

        #connect spinboxes
        self.epsBox.valueChanged.connect(self.epsValueChange)
        self.minSampleBox.valueChanged.connect(self.minSampleChange) 
        self.maxDistanceBox.valueChanged.connect(self.maxDistanceChange)    
        self.unitPerPixelBox.valueChanged.connect(self.unitPerPixelChange) 
        self.centroidSymbolSizeBox.valueChanged.connect(self.centroidSymbolSizeChange)  
        #connect combobox
        self.analysis_Box.setCurrentIndex(0)
        self.analysis_Box.currentIndexChanged.connect(self.analysisChange)         
        
    def epsValueChange(self,value):
        self.epsBox = value
        self.viewer.eps = self.epsBox
        return
    
    def minSampleChange(self,value):
        self.min_samples = value
        self.viewer.min_samples = self.min_samples 
        return
        
    def maxDistanceChange(self,value):
        self.maxDistance = value
        self.viewer.maxDistance = self.maxDistance
        return

    def unitPerPixelChange(self,value):
        self.unitPerPixel = value
        self.viewer.unitPerPixel = self.unitPerPixel
        return
    
    def centroidSymbolSizeChange(self,value):
        self.centroidSymbolSize = value
        self.viewer.centroidSymbolSize = self.centroidSymbolSize
        return   
    
    def analysisChange(self):
        self.viewer.clusterAnaysisSelection = self.analysis_Box.currentText()
        return
    
#    def multiThreadClicked(self):
#        self.viewer.multiThreadingFlag = self.multiThread_checkbox.isChecked()
#        return





### TESTING ####
def test():
    fileName = r"C:\Users\George\Desktop\batchTest\0_trial_1_superes_cropped.txt"
    #fileName = r"C:\Users\George\Desktop\ianS-synapse\trial_1_superes_fullfield.txt"
    clusterAnalysis.viewerGUI()
    clusterAnalysis.open_file(fileName)
    clusterAnalysis.display2Ddata_allPoints()
    clusterAnalysis.display3Ddata_allPoints()
    clusterAnalysis.getClusters()  
    clusterAnalysis.getCentroids()
    clusterAnalysis.makeHulls()  
    clusterAnalysis.display2Dcentroids()    
    clusterAnalysis.display3Dcentroids()  
    clusterAnalysis.randomPointAnalysis()
    clusterAnalysis.display2Dcentroids_rnd()   
    clusterAnalysis.display3Dcentroids_rnd()
    clusterAnalysis.printStats()
    clusterAnalysis.makeROIs()
    clusterAnalysis.makeROI_DF()    
    #clusterAnalysis.displayROIpoints_2D('ALL')
    #clusterAnalysis.display2Dcentroids_roi('ALL')
    #clusterAnalysis.display3Ddata_roi('ALL')
    #clusterAnalysis.display3Dcentroids_roi('ALL')
    clusterAnalysis.displayROIresults()
    return     
    
test() 

#clusterAnalysis.viewerGUI()