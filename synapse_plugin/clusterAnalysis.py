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
        self.clustersGeneated = False
        
        self.clusterIndex = []

        #cluster analysis options
        self.clusterAnaysisSelection = 'All Clusters'
        self.clusterType = '3D'
        
        self.All_ROIs_pointsList = []
        self.channelList = []
        
        #self..multiThreadingFlag = False
   
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


    def clear(self):
    	self.Channels = []
     

    def getClusters(self):
        t = Timer()
        #get 3D points
        t.start()
        self.ch1Points_3D = self.Channels[0].getPoints(z=True)
        self.ch2Points_3D = self.Channels[1].getPoints(z=True)
       
        #get cluster labels for each channel
        print('--- channel 1 ---')
        self.ch1_labels,self.ch1_numClusters,self.ch1_numNoise = dbscan(self.ch1Points_3D, eps=self.eps, min_samples=self.min_samples, plot=False)
        print('--- channel 2 ---')
        self.ch2_labels,self.ch2_numClusters,self.ch2_numNoise = dbscan(self.ch2Points_3D, eps=self.eps, min_samples=self.min_samples, plot=False)    
        print('-----------------')
        
        t.timeReport('2D clusters created')  
        #get 2D points
        t.start()
        self.ch1Points = self.Channels[0].getPoints(z=True)
        self.ch2Points = self.Channels[1].getPoints(z=True) 
    
        #get 3D centeroids for cluster analysis
        _, self.ch1_centeroids_3D, _ = self.getHulls(self.ch1Points_3D,self.ch1_labels)
        _, self.ch2_centeroids_3D, _ = self.getHulls(self.ch2Points_3D,self.ch2_labels)
        
        t.timeReport('3D clusters created')    
        
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
 
        #t.timeReport('hulls created')
                
        #combine nearest roi between channels
        #t.start()
        combinedHulls, combinedPoints = combineClosestHulls(ch1_hulls,ch1_centeroids,self.ch1_groupPoints,ch2_hulls,ch2_centeroids,self.ch2_groupPoints, self.maxDistance)
  
        #t.timeReport('new hulls created')
        
        #get new hulls for combined points
        #t.start()
        newHulls = self.getHulls2(combinedPoints)         
        #self.plotHull(combinedPoints[0],newHulls[0])       
        t.timeReport('hulls created')

        #draw rois around combined hulls
        #self.createROIFromHull(combinedPoints[0],newHulls[0])
        t.start()
        print('--- combined channels ---')
        
        #single thread
        for i in range(len(combinedHulls)):
            self.createROIFromHull(combinedPoints[i],newHulls[i]) ### THIS IS SLOW! ###
            print('\r', 'creating rois: {:0.2f}'.format((i/len(combinedHulls))*100),'%', end='\r', flush=True)
        
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
       
              
        t.timeReport('ROI created')

        #g.m.statusBar().showMessage('{} ROI created'.format(str(len(combinedHulls))))
        
        #self.updateClusterData()
        t.start()
        if len(combinedHulls) > 0:
            self.clustersGeneated = True
        
        t.timeReport('ROI data updated')
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
        self.All_ROIs_pointsList.append(pointsList)
        
        #make channel list for all points
        self.makeChannelList()

        #t.timeReport('ROI made')
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
        self.dist_clusters = self.getNearestNeighbors(self.ch1_centeroids_3D,self.ch2_centeroids_3D)
        #print(self.dist_clusters)        
        #print(min(self.data['Xc']), min(self.data['Yc']), min(self.data['Zc']))
        #print(max(self.data['Xc']), max(self.data['Yc']), max(self.data['Zc']))
        self.ch1_random = self.getRandomXYZ(min(self.data['Xc']),
                                       min(self.data['Yc']),
                                       min(self.data['Zc']),
                                       max(self.data['Xc']),
                                       max(self.data['Yc']),
                                       max(self.data['Zc']), len(self.ch1_centeroids_3D))


        self.ch2_random = self.getRandomXYZ(min(self.data['Xc']),
                                       min(self.data['Yc']),
                                       min(self.data['Zc']),
                                       max(self.data['Xc']),
                                       max(self.data['Yc']),
                                       max(self.data['Zc']), len(self.ch2_centeroids_3D))


        self.dist_random = self.getNearestNeighbors(self.ch1_random,self.ch2_random)

        self.distAll_clusters = self.getNearestNeighbors(self.ch1_centeroids_3D,self.ch2_centeroids_3D, k=len(self.ch1_centeroids_3D))
        self.distAll_random = self.getNearestNeighbors(self.ch1_random,self.ch2_random, k=len(self.ch1_random))
        
        return


#    def plotClusters(self):
#        '''3D scatter plots of data points with cluster labels - using matplotlib'''
#        fig = plt.figure()
#        ax1 = fig.add_subplot(111, projection='3d')
#        ax1.scatter(self.ch1Points_3D[::,0], self.ch1Points_3D[::,1], self.ch1Points_3D[::,2], marker='o', c=self.ch1_labels, cmap="RdBu")
#        ax1.scatter(self.ch2Points_3D[::,0], self.ch2Points_3D[::,1], self.ch2Points_3D[::,2], marker='x', c=self.ch2_labels, cmap="RdBu")
#        ax1.view_init(azim=0, elev=90)
#        plt.show()
#        return


    def plotClusters(self):
        '''3D scatter plots of data points with cluster labels - using pyqtgraph'''
        self.app = QtGui.QApplication([])
        self.mw = QtGui.QMainWindow()
        self.mw.resize(800,800)
        view = pg.GraphicsLayoutWidget()  ## GraphicsView with GraphicsLayout inserted by default
        self.mw.setCentralWidget(view)
        self.mw.show()
        self.mw.setWindowTitle('all points and cluster centeroids')
        #create plot window
        w1 = view.addPlot()
        #make point data
        point_n1 = len(self.ch1Points_3D[::,0])
        point_n2 = len(self.ch2Points_3D[::,0])
        point_s1 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 120))
        point_s2 = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(0, 0, 255, 120))        
        point_pos1 = np.array([self.ch1Points_3D[::,0],self.ch1Points_3D[::,1]])
        point_pos2 = np.array([self.ch2Points_3D[::,0],self.ch2Points_3D[::,1]])
        point_spots1 = [{'pos': point_pos1[:,i], 'data': 1} for i in range(point_n1)]
        point_spots2 = [{'pos': point_pos2[:,i], 'data': 1} for i in range(point_n2)]
        point_s1.addPoints(point_spots1)
        point_s2.addPoints(point_spots2)
        w1.addItem(point_s1)
        w1.addItem(point_s2)
        #make centeroid data
        centeroid_n1 = len(self.ch1_centeroids_3D[::,0])
        centeroid_n2 = len(self.ch2_centeroids_3D[::,0])
        centeroid_s1 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        centeroid_s2 = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))        
        centeroid_pos1 = np.array([self.ch1_centeroids_3D[::,0],self.ch1_centeroids_3D[::,1]])
        centeroid_pos2 = np.array([self.ch2_centeroids_3D[::,0],self.ch2_centeroids_3D[::,1]])
        centeroid_spots1 = [{'pos': centeroid_pos1[:,i], 'data': 1} for i in range(centeroid_n1)]
        centeroid_spots2 = [{'pos': centeroid_pos2[:,i], 'data': 1} for i in range(centeroid_n2)]
        centeroid_s1.addPoints(centeroid_spots1)
        centeroid_s2.addPoints(centeroid_spots2)
        w1.addItem(centeroid_s1)
        w1.addItem(centeroid_s2)       
        return

    def plot3DClusters(self):
        '''3D scatter plot using GL ScatterPlot'''
        plot3DScatter = Plot3D_GL(self.ch1Points_3D,self.ch2Points_3D)   
        plot3DScatter.plot()
        return

    def plotAnalysis(self):
        ''''3D scatter plots of centeroids and histograms of distances'''
        fig = plt.figure()
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.scatter(self.ch1_centeroids_3D[::,0], self.ch1_centeroids_3D[::,1], self.ch1_centeroids_3D[::,2], marker='o')
        ax1.scatter(self.ch2_centeroids_3D[::,0], self.ch2_centeroids_3D[::,1], self.ch2_centeroids_3D[::,2], marker='^')

        ax1.set_title('Cluster Centeroids') 
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        ax3 = fig.add_subplot(234, projection='3d')
        ax3.scatter(self.ch1_random[::,0], self.ch1_random[::,1], self.ch1_random[::,2], marker='o')
        ax3.scatter(self.ch2_random[::,0], self.ch2_random[::,1], self.ch2_random[::,2], marker='^')

        ax3.set_title('Random Points')        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        ax2 = fig.add_subplot(232)
        ax2.hist(self.dist_clusters)
        ax2.set_title('Nearest Neighbor')
        ax2.set_ylabel('# of observations')
        ax2.set_xlabel('distance')        
        
        ax5 = fig.add_subplot(233)
        ax5.hist(self.distAll_clusters)
        ax5.set_title('All Neighbors')
        ax5.set_ylabel('# of observations')
        ax5.set_xlabel('distance') 
        
        ax4 = fig.add_subplot(235)
        ax4.hist(self.dist_random)
        #ax4.set_title('Nearest Neighbor')
        ax4.set_ylabel('# of observations')
        ax4.set_xlabel('distance')  

        ax6 = fig.add_subplot(236)
        ax6.hist(self.distAll_random)
        #ax6.set_title('All Neighbors')
        ax6.set_ylabel('# of observations')
        ax6.set_xlabel('distance') 
        
        plt.show()
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


    def analyze_roi(self, roi, channelList, roiIndex):
        '''analyse roi pts'''
        #channels = [self.Channels[0],self.Channels[1]]
        
        ch1_pts = roi[channelList == self.Channels[0].__name__]
        ch2_pts = roi[channelList == self.Channels[1].__name__]  
                        
        roi_data = OrderedDict([('ROI #', roiIndex), ('Mean Distance (%s)' % self.unit_prefixes[self.unit], 0), ('%s N' % self.Channels[0].__name__, 0), \
        ('%s N' % self.Channels[1].__name__, 0), ('%s Volume (%s^3)' % (self.Channels[0].__name__, self.unit_prefixes[self.unit]), 0), ('%s Volume (%s^3)' % (self.Channels[1].__name__, self.unit_prefixes[self.unit]), 0)])


        roi_data['%s N' % self.Channels[0].__name__] = len(ch1_pts)
        roi_data['%s N' % self.Channels[1].__name__] = len(ch2_pts)        

        try:        
            if len(ch1_pts) >= 4:
                roi_data['%s Volume (%s^3)' % (self.Channels[0].__name__, self.unit_prefixes[self.unit])] = convex_volume(ch1_pts)
    
            else:
                roi_data['%s Volume (%s^3)' % (self.Channels[0].__name__, self.unit_prefixes[self.unit])] = 0

        except:
                roi_data['%s Volume (%s^3)' % (self.Channels[0].__name__, self.unit_prefixes[self.unit])] = 0


        try:
            if len(ch2_pts) >= 4:
    
                roi_data['%s Volume (%s^3)' % (self.Channels[1].__name__, self.unit_prefixes[self.unit])] = convex_volume(ch2_pts)
    
            else:
                roi_data['%s Volume (%s^3)' % (self.Channels[1].__name__, self.unit_prefixes[self.unit])] = 0
    
        except:
                roi_data['%s Volume (%s^3)' % (self.Channels[1].__name__, self.unit_prefixes[self.unit])] = 0


            #g.m.statusBar().showMessage('Cannot get Volume of %s in roi %d with %d points' % (ch.__name__, roi.id, ch.getCount())) 
        
        
        roi_data['Mean Distance (%s)' % self.unit_prefixes[self.unit]] = np.linalg.norm(np.average(ch1_pts, 0) - np.average(ch2_pts, 0))
        roi_data['%s Centeroid' % (self.Channels[0].__name__)] = np.average(ch1_pts, 0)
        roi_data['%s Centeroid' % (self.Channels[1].__name__)] = np.average(ch2_pts, 0)
                    
        return roi_data


    def makeROI_DF(self):
        '''pass each roi to analyze_roi(), compile resuts in table'''
        dictList = []
        for i in range(len(self.All_ROIs_pointsList)):
            roi_data = self.analyze_roi(self.All_ROIs_pointsList[i],self.channelList[i],i)
            dictList.append(roi_data)
            print('\r', 'analysing rois: {:0.2f}'.format((i/len(self.All_ROIs_pointsList))*100),'%', end='\r', flush=True)
        #make df
        self.roiAnalysisDF = pd.DataFrame(dictList)
        #print(self.roiAnalysisDF.head())
        return

    def saveROIAnalysis(self, savePath='',fileName=''):
        '''save roi analysis dataframe as csv'''
        self.makeROI_DF()
        saveName = os.path.join(savePath, fileName + '_roiAnalysis.csv')
        self.roiAnalysisDF.to_csv(saveName)   
        print('roi analysis saved as:', saveName)              
        return

    def printStats(self):
        '''print stats'''
        print('----------------------------------------------')
        print(self.clusterAnaysisSelection)        
        print('----------------------------------------------')
        print('Channel 1: Number of clusters: ', str(len(self.ch1_centeroids_3D)))
        print('Channel 2: Number of clusters: ', str(len(self.ch2_centeroids_3D)))        
        print('Number of nearest neighbor distances:', str(np.size(self.dist_clusters)))
        print('Mean nearest neighbor distance:', str(np.mean(self.dist_clusters)))
        print('StDev nearest neighbor distance:', str(np.std(self.dist_clusters)))        
        print('Number of All distances:', str(np.size(self.distAll_clusters)))
        print('Mean All distance:', str(np.mean(self.distAll_clusters)))
        print('StDev All distance:', str(np.std(self.distAll_clusters)))       
        print('----------------------------------------------')
        print('Random 1: Number of clusters: ', str(len(self.ch1_random)))
        print('Random 2: Number of clusters: ', str(len(self.ch2_random)))        
        print('Random: Number of nearest neighbor distances:', str(np.size(self.dist_random)))
        print('Random: Mean nearest neighbor distance:', str(np.mean(self.dist_random)))
        print('Random: StDev nearest neighbor distance:', str(np.std(self.dist_random)))       
        print('Random: Number of All distances:', str(np.size(self.distAll_random)))
        print('Random: Mean All distance:', str(np.mean(self.distAll_random)))  
        print('Random: Stdev All distance:', str(np.std(self.distAll_random)))          
        print('----------------------------------------------')
        return


    def saveStats(self ,savePath='',fileName=''):
        '''save clustering stats as csv'''
        d= {
            'Channel 1: Number of clusters': (len(self.ch1_centeroids_3D)),
            'Channel 2: Number of clusters': (len(self.ch2_centeroids_3D)),
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
        d3 = {'ch1_centeroids_x':self.ch1_centeroids_3D[::,0],
              'ch1_centeroids_y':self.ch1_centeroids_3D[::,1],
              'ch1_centeroids_z':self.ch1_centeroids_3D[::,2]}
        
        d4 = {'ch2_centeroids_x':self.ch2_centeroids_3D[::,0],
              'ch2_centeroids_y':self.ch2_centeroids_3D[::,1],
              'ch2_centeroids_z':self.ch2_centeroids_3D[::,2]}
        
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


    def runAnalysis(self, file, eps = 100 , min_samples = 10, maxDistance = 100, pathName=''):
        try:
            fileName = file.split('\\')[-1].split('.')[0]
            print('analysing: ', fileName)
            filePath = file
            savePath = pathName + r'\results'
            print('save path: ', savePath, '\n')
            
            ##### set parameters ####
            self.name = fileName
            self.eps = eps
            self.min_samples = min_samples
            self.maxDistance = maxDistance
            
            ##### run #####
            self.open_file(filename=filePath)
            self.getClusters()
            #self.plotClusters()
            #self.plot3DClusters()
            self.randomPointAnalysis()
            #self.plotAnalysis()
            self.printStats()
            self.saveResults(savePath, fileName=fileName)
            self.saveStats(savePath, fileName=fileName)
            self.saveROIAnalysis(savePath, fileName=fileName)
        except:
            print('skipped: ',fileName)
            raise
        print ('finished analysing: ', fileName)
        return

    def runBatch(self, pathName, eps = 100 , min_samples = 10, maxDistance = 100, test=False):
        '''run all txt files in folder'''
        files = [f for f in glob.glob(pathName + "**/*.txt", recursive=True)]

        #FOR TESTING - JUST FIRST FILE IN LIST
        if test:
            files=[files[0]]
                    
        for file in files:            
            self.runAnalysis(file, eps, min_samples, maxDistance, pathName)            
            print('batch finished!')
        return

clusterAnalysis = ClusterAnalysis()


class Synapse3D_batch(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(Synapse3D_batch, self).__init__(parent)
        #add window title
        self.setWindowTitle("Volume Slider GUI")
        
        self.pathName = ''

        self.eps  = clusterAnalysis.eps
        self.min_samples = clusterAnalysis.min_samples
        self.maxDistance = clusterAnalysis.maxDistance
        self.unitPerPixel = clusterAnalysis.unitPerPixel
        #self.multiThreadingFlag = clusterAnalysis.multiThreadingFlag
        
        self.clusterType = '3D'
        
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
        self.label_clustertype = QtWidgets.QLabel("clustering type:")        
        self.displayTitle = QtWidgets.QLabel("----- Display Parameters -----") 
        self.label_unitPerPixel = QtWidgets.QLabel("nanometers/pixel:") 
        #self.label_centroidSymbolSize = QtWidgets.QLabel("centroid symbol size:")  
        
        self.analysisTitle = QtWidgets.QLabel("----- Cluster Analysis -----") 
        self.label_analysis = QtWidgets.QLabel("Clusters to analyse: ")         
        
        #self.multiThreadTitle = QtWidgets.QLabel("----- Multi-Threading -----") 
        #self.label_multiThread = QtWidgets.QLabel("Multi-Threading On: ")        
        
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
        #self.centroidSymbolSizeBox = QtWidgets.QSpinBox()    
        #self.centroidSymbolSizeBox.setRange(0,1000)
        #self.centroidSymbolSizeBox.setValue(self.centroidSymbolSize)          

        #combobox
        #self.analysis_Box = QtWidgets.QComboBox()
        #self.analysis_Box.addItems(["All Clusters", "Paired Clusters"])
        self.clustertype_Box = QtWidgets.QComboBox()
        self.clustertype_Box.addItems(["2D", "3D"])
        self.clustertype_Box.setCurrentText(self.clusterType)
        self.clustertype_Box.currentIndexChanged.connect(self.clusterTypeChange)
        
        #tickbox
        #self.multiThread_checkbox = CheckBox()
        #self.multiThread_checkbox.setChecked(self.multiThreadingFlag)
        #self.multiThread_checkbox.stateChanged.connect(self.multiThreadClicked)

        #buttons
        self.button_start = QtWidgets.QPushButton("Set Folder and Start Batch Analysis")
        
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
        layout.addWidget(self.label_clustertype, 6, 0) 
        layout.addWidget(self.clustertype_Box, 6, 1)  
        layout.addWidget(self.button_start, 7, 0)         
        
        #layout.addWidget(self.label_centroidSymbolSize, 6, 0)  
        #layout.addWidget(self.centroidSymbolSizeBox, 6, 1) 
        
        #layout.addWidget(self.analysisTitle, 8, 0, 1, 2)  
        #layout.addWidget(self.label_analysis, 9, 0)  
        #layout.addWidget(self.analysis_Box, 9, 1)     
        
        #layout.addWidget(self.multiThreadTitle, 10, 0, 1, 2)  
        #layout.addWidget(self.label_multiThread, 11, 0)  
        #layout.addWidget(self.multiThread_checkbox, 11, 1)  
        

        
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Clustering Options")

        #connect spinboxes
        self.epsBox.valueChanged.connect(self.epsValueChange)
        self.minSampleBox.valueChanged.connect(self.minSampleChange) 
        self.maxDistanceBox.valueChanged.connect(self.maxDistanceChange)    
        self.unitPerPixelBox.valueChanged.connect(self.unitPerPixelChange) 
        #self.centroidSymbolSizeBox.valueChanged.connect(self.centroidSymbolSizeChange)  
        #connect combobox
        #self.analysis_Box.setCurrentIndex(0)
        #self.analysis_Box.currentIndexChanged.connect(self.analysisChange)         

        #connect buttons
        self.button_start.clicked.connect(self.run)

        
    def epsValueChange(self,value):
        self.epsBox = value
        clusterAnalysis.eps = self.epsBox
        return
    
    def minSampleChange(self,value):
        self.min_samples = value
        clusterAnalysis.min_samples = self.min_samples 
        return
        
    def maxDistanceChange(self,value):
        self.maxDistance = value
        clusterAnalysis.maxDistance = self.maxDistance
        return

    def unitPerPixelChange(self,value):
        self.unitPerPixel = value
        clusterAnalysis.unitPerPixel = self.unitPerPixel
        return
 
    def clusterTypeChange(self):
        self.clusterType = self.clustertype_Box.currentText
        clusterAnalysis.clusterType = self.clusterType
    
#    def centroidSymbolSizeChange(self,value):
#        self.centroidSymbolSize = value
#        self.viewer.centroidSymbolSize = self.centroidSymbolSize
#        return   
    
#    def analysisChange(self):
#        self.viewer.clusterAnaysisSelection = self.analysis_Box.currentText()
#        return
    
#    def multiThreadClicked(self):
#        self.viewer.multiThreadingFlag = self.multiThread_checkbox.isChecked()
#        return

    def getSavePath(self): 
        folder = QtWidgets.QFileDialog.getExistingDirectory(g.m, "Select batch folder.", os.path.expanduser("~"), QtWidgets.QFileDialog.ShowDirsOnly)
        self.pathName = folder
        return

    def run(self):
        self.getSavePath()
        if not os.path.exists(os.path.join(self.pathName,'results')):
            os.makedirs(os.path.join(self.pathName,'results'))
        clusterAnalysis.runBatch(self.pathName)
        return

    def start(self):
        self.show()
        return


### TESTING ####
def test():
    pathName = r"C:\Users\George\Desktop\batchTest"
    clusterAnalysis.clusterType = '2D'
    clusterAnalysis.runBatch(pathName, test=True)
    clusterAnalysis.plotClusters()   
    return     
    
