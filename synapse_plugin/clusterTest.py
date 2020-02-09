# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 08:54:05 2020
Synapse3D - Clustering code
@author: George
"""

"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
import os,sys

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



class Test:
    
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
        self.clustersGeneated = False
        
        self.clusterIndex = []

        #cluster analysis options
        self.clusterAnaysisSelection = 'All Clusters'
        
        self.All_ROIs_pointsList = []
        
   
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


    def clear(self):
    	self.Channels = []

    def getClusters(self):
        t = Timer()
        #get 3D points
        t.start()
        ch1Points_3D = self.Channels[0].getPoints(z=True)
        ch2Points_3D = self.Channels[1].getPoints(z=True)
        
        print('\n-----------------')
        print(':::TIME:::: 3D point data collected: {0:1.3f}'.format(t.stop()))
        print('-----------------\n')
        
        #get cluster labels for each channel
        t.start()
        print('--- channel 1 ---')
        self.ch1_labels = dbscan(ch1Points_3D, eps=self.eps, min_samples=self.min_samples, plot=False)
        print('--- channel 2 ---')
        self.ch2_labels = dbscan(ch2Points_3D, eps=self.eps, min_samples=self.min_samples, plot=False)    
        print('-----------------')

        print('\n-----------------')        
        print(':::TIME:::: 2D clusters created: {0:1.3f}'.format(t.stop()))
        print('-----------------\n')        
        
        #get 2D points
        t.start()
        ch1Points = self.Channels[0].getPoints(z=False)
        ch2Points = self.Channels[1].getPoints(z=False) 

        print('\n-----------------')        
        print(':::TIME:::: 2D point data collected: {0:1.3f}'.format(t.stop()))     
        print('-----------------\n')
        
        #get 3D centeroids for cluster analysis
        t.start()
        _, self.ch1_centeroids_3D, _ = self.getHulls(ch1Points_3D,self.ch1_labels)
        _, self.ch2_centeroids_3D, _ = self.getHulls(ch2Points_3D,self.ch2_labels)
        
        print('\n-----------------')        
        print(':::TIME:::: 3D clusters created: {0:1.3f}'.format(t.stop()))      
        print('-----------------\n')
        
        #get hulls for each channels clusters  
        t.start()
        ch1_hulls, ch1_centeroids, ch1_groupPoints = self.getHulls(ch1Points,self.ch1_labels)
        #self.plotHull(ch1_groupPoints[0],ch1_hulls[0])
        ch2_hulls, ch2_centeroids, ch2_groupPoints = self.getHulls(ch2Points,self.ch2_labels)

        print('\n-----------------')        
        print(':::TIME:::: hulls created: {0:1.3f}'.format(t.stop()))         
        print('-----------------\n')  
        
        #combine nearest roi between channels
        t.start()
        combinedHulls, combinedPoints = combineClosestHulls(ch1_hulls,ch1_centeroids,ch1_groupPoints,ch2_hulls,ch2_centeroids,ch2_groupPoints, self.maxDistance)

        print('\n-----------------')            
        print(':::TIME:::: hulls created: {0:1.3f}'.format(t.stop()))
        print('-----------------\n')      
        
        #get new hulls for combined points
        t.start()
        newHulls = self.getHulls2(combinedPoints) 

        print('\n-----------------')            
        print(':::TIME:::: new hulls created: {0:1.3f}'.format(t.stop()))       
        print('-----------------\n')          
        
        #self.plotHull(combinedPoints[0],newHulls[0])       


        #draw rois around combined hulls
        #self.createROIFromHull(combinedPoints[0],newHulls[0])
        t.start()
        print('--- combined channels ---')
        
        #single thread
        for i in range(len(combinedHulls)):
            self.createROIFromHull(combinedPoints[i],newHulls[i]) ### THIS IS SLOW! ###

        # #multi-thread
        # t2 = Timer()
        # t2.start()
        # self.threadpool = QThreadPool()
        # print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())


        # def progress_fn(n):
        #     print("%d%% done" % n)
        
        # def makeROIs(progress_callback):
        #     for i in range(len(combinedHulls)):
        #         self.createROIFromHull(combinedPoints[i],newHulls[i])
        #         progress_callback.emit((i/len(combinedHulls))*100)
        #     return "Done."

        # def thread_complete():
        #     print("THREAD COMPLETE! - time taken:{0:1.3f}".format(t2.stop()))

        
        # # Pass the function to execute
        # worker = Worker(makeROIs) # Any other args, kwargs are passed 
        # worker.signals.finished.connect(thread_complete)
        # worker.signals.progress.connect(progress_fn)                
        # #start threads
        # self.threadpool.start(worker)
       
        
        print('\n-----------------')                 
        print(':::TIME:::: ROI created: {0:1.3f}'.format(t.stop()))       
        print('-----------------\n')        
        
        print('{} ROI created'.format(str(len(combinedHulls))))
        #g.m.statusBar().showMessage('{} ROI created'.format(str(len(combinedHulls))))
        
        #self.updateClusterData()
        t.start()
        if len(combinedHulls) > 0:
            self.clustersGeneated = True

        print('\n-----------------')              
        print(':::TIME:::: ROI data updated: {0:1.3f}'.format(t.stop()))       
        print('-----------------\n')          
        
        return


    def getHulls(self,points,labels):
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        hulls = []
        centeroids = []
        groupPoints = []
        for i in range(n_clusters):
            clusterPoints = points[labels==i]
            groupPoints.append(clusterPoints)           
            hulls.append(ConvexHull(clusterPoints).simplices)
            centeroids.append(np.average(clusterPoints,axis=0)) 
        return np.array(hulls), np.array(centeroids), np.array(groupPoints)


    def getHulls2(self,pointList):
        hullList = []
        for points in pointList:
            hullList.append(ConvexHull(points).simplices)
        return hullList


    def createROIFromHull(self,points, hull):
        t = Timer()
        t.start()
        '''add roi to display from hull points'''
        #make points list
        pointsList = []     
        for simplex in hull:
            pointsList.append((points[simplex][0])) 
        for simplex in hull:            
            pointsList.append((points[simplex][1])) 

        #print('\n-----------------')
        #print(':::TIME:::: Points list made: {0:1.3f}'.format(t.stop()))
        #print('-----------------\n')
        
        #t.start()
        #order points list
        pointsList = order_points(pointsList)
        
        #print('\n-----------------')
        #print(':::TIME:::: Points list ordered: {0:1.3f}'.format(t.stop()))
        #print('-----------------\n')


        #t.start()
        #convert list to np array
        pointsList = np.array(pointsList)
        
        #print('\n-----------------')
        #print(':::TIME:::: Array made: {0:1.3f}'.format(t.stop()))
        #print('-----------------\n')        


        #t.start()        
        #add create ROIs from points
        #self.plotWidget.getViewBox().createROIFromPoints(pointsList)
        
        #add points to All_ROI_pointsList
        self.All_ROIs_pointsList.append(pointsList)

        print('\n-----------------')
        print(':::TIME:::: ROI made: {0:1.3f}'.format(t.stop()))
        print('-----------------\n')   


test = Test()
test.open_file(filename=r"C:\Users\George\Desktop\ianS-synapse\trial_1_superes_cropped.txt")
test.getClusters()