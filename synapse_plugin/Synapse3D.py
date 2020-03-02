"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
import os,sys
#from .BioDocks import *
#from .BioDocks.DockWindow import * 
#from .BioDocks.AnalysisIO import *
#from .BioDocks.Tools import *
#from .BioDocks.Channels import *
#from .BioDocks.ClusterMath import *
#from .BioDocks.dbscan import *
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




class Synapse3D(BaseProcess):

    def __init__(self):
        super().__init__()
        
        self.app = QtWidgets.QApplication([])
        
        self.win = DockWindow(addMenu=False)
        self.win.resize(1700, 900)
        self.win.setWindowTitle('Main Window')
        self.plotWidget = PlotWidget(viewBox=ROIViewBox(creatingROI=True, roiSnap=False))
        
        #camera option
        self.unitPerPixel = 166
        
        # data is loaded in nanometers, divided by # according to units
        self.units = {'Pixels': self.unitPerPixel, 'Nanometers': 1}
        self.unit_prefixes = {'Pixels': 'px', 'Nanometers': 'nm'}
        self.unit = 'Nanometers'
        
        self.yAxis = self.plotWidget.getPlotItem().getAxis('left')
        self.xAxis = self.plotWidget.getPlotItem().getAxis('bottom')
        self.yAxis.setLabel(text='Y', units=self.unit)
        self.yAxis.enableAutoSIPrefix(False)
        self.xAxis.enableAutoSIPrefix(False)
        self.xAxis.setLabel(text='X', units=self.unit)
        
        self.legend = self.plotWidget.addLegend()
        
        self.colors = ((255, 0, 0), (0, 255, 0))
        self.color_dict = {'atto488': self.colors[0], 'Alexa647': self.colors[1]}
        self.ignore = {"Z Rejected"}
        
        self.Channels = []
        self.empty_channel = Channel('Empty', [], (1, 1, 1))
        
        #clustering option
        self.eps = 100          #max distance between points within a cluster
        self.min_samples = 10   #min number of points to form a cluster
        self.maxDistance = 100  #max distance between clusters in differnt channels when forming combined ROI
        
        #display options
        self.centroidSymbolSize = 10
        
        #data state
        self.dataLoaded = False
        self.ROI3D_Initiated = False
        self.dataDisplayed = 'original'
        self.clustersGeneated = False
        
        self.clusterIndex = []

        #cluster analysis options
        self.clusterAnaysisSelection = 'All Clusters'

        self.clusterType = '2D'

        #multithreading option
        self.multiThreadingFlag = False
        
    def displayData(self):
    	self.dataWidget.setData(sorted([roi.synapse_data for roi in self.plotWidget.items() if \
    		isinstance(roi, Freehand) and hasattr(roi, 'synapse_data')], key=lambda f: f['ROI #']))
    	self.dataWidget.changeFormat('%.3f')
    
    def subchannels_in_roi(self,roi):
    	channels = []
    	for ch in self.Channels:
    		pts_in = []
    		for syn_pt in ch.pts:
    			if roi.contains(QPointF(syn_pt[0], syn_pt[1])):
    				pts_in.append(syn_pt)
    		channels.append(Channel(ch.__name__, pts_in, ch.color()))
    	return channels


    def getROI_pts(self, roi):
        ch1,ch2 = self.subchannels_in_roi(roi)
        return ch1.pts, ch2.pts
    
    def analyze_roi(self,roi):
    	channels = self.subchannels_in_roi(roi)
    	roi.synapse_data = OrderedDict([('ROI #', roi.id), ('Mean Distance (%s)' % self.unit_prefixes[self.unit], 0), ('%s N' % self.Channels[0].__name__, 0), \
    	('%s N' % self.Channels[1].__name__, 0), ('%s Volume (%s^3)' % (self.Channels[0].__name__, self.unit_prefixes[self.unit]), 0), ('%s Volume (%s^3)' % (self.Channels[1].__name__, self.unit_prefixes[self.unit]), 0)])
    
    	for i, ch in enumerate(channels):
    		roi.synapse_data['%s N' % ch.__name__] = ch.getCount()
    		if ch.getCount() >= 4:
    			roi.synapse_data['%s Volume (%s^3)' % (ch.__name__, self.unit_prefixes[self.unit])] = convex_volume(ch.getPoints(True))
    		else:
    			print('Cannot get Volume of %s in roi %d with %d points' % (ch.__name__, roi.id, ch.getCount()))
    			g.m.statusBar().showMessage('Cannot get Volume of %s in roi %d with %d points' % (ch.__name__, roi.id, ch.getCount())) 
                
    	if hasattr(roi, 'mean_line'):
    		self.plotWidget.removeItem(roi.mean_line)
    
    	if all([ch.getCount() > 0 for ch in channels]):
    		roi.synapse_data['Mean Distance (%s)' % self.unit_prefixes[self.unit]] = np.linalg.norm(channels[1].getCenter(z=True) - channels[0].getCenter(z=True))
    		roi.mean_line = pg.PlotDataItem(np.array([channels[1].getCenter(), channels[0].getCenter()]), symbol='d', symbolSize=self.centroidSymbolSize )
    		roi.mean_line.setParentItem(roi)
    		roi.mean_line.setVisible(False)
    		roi.synapse_data['%s Centeroid' % (self.Channels[0].__name__)] = channels[0].getCenter(z=True)
    		roi.synapse_data['%s Centeroid' % (self.Channels[1].__name__)] = channels[1].getCenter(z=True)
            
    	else:
    		del roi.synapse_data
    		print('Must select exactly 2 channels to calculate distance. Ignoring ROI %d' % roi.id)
    		g.m.statusBar().showMessage('Must select exactly 2 channels to calculate distance. Ignoring ROI %d' % roi.id)            
    	self.displayData()
    
    def plotROIChannels(self,roi):      
    	if not self.synapseDock.isVisible():
    		self.synapseDock.float()
    		self.synapseDock.window().setGeometry(20, 100, 500, 500)
    	ch1, ch2 = self.subchannels_in_roi(roi)
    	if hasattr(self.synapseWidget, 'synapse'):
    		self.synapseWidget.synapse.setChannels(ch1, ch2)
    	else:
    		self.synapseWidget.synapse = Synapse(ch1, ch2)
    		self.synapseWidget.addItem(self.synapseWidget.synapse, name=roi.__name__)
    	cen = np.average(self.synapseWidget.synapse.pos, axis=0)
    	d = max([np.linalg.norm(np.subtract(p, cen)) for p in self.synapseWidget.synapse.pos])
    	self.synapseWidget.moveTo(cen, distance = 2 * d)
    
    def roiCreated(self,roi):
    	roi.sigChanged.connect(self.analyze_roi)
    	roi.sigRemoved.connect(lambda : self.displayData())
    	roi.sigHoverChanged.connect(lambda r, h: r.mean_line.setVisible(h) if hasattr(r, 'mean_line') else None)
    	roi.sigClicked.connect(self.plotROIChannels)
    	self.analyze_roi(roi)
    
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
    	g.m.statusBar().showMessage('Gathering channels...')        
    	self.names = set(self.data['Channel Name'].astype(str)) - self.ignore
    	print('Channels Found: %s' % ', '.join(self.names))
    	g.m.statusBar().showMessage('Channels Found: %s' % ', '.join(self.names))
    
    	self.data['Xc'] /= self.units[self.unit]
    	self.data['Yc'] /= self.units[self.unit]
    	self.data['Zc'] /= self.units[self.unit]
    
    	#global Channels
    	self.Channels = []
    	self.plotWidget.clear()
    	self.pts = [ActivePoint(data={k: self.data[k][i] for k in self.data}) for i in range(len(self.data['Channel Name']))]
    	for i, n in enumerate(self.names):
    		if n in self.color_dict:
    			color = self.color_dict[n]
    		else:
    			color = self.colors[i]
    		self.Channels.append(Channel(n, [p for p in self.pts if p['Channel Name'] == n], color))
    		self.plotWidget.addItem(self.Channels[-1])
    		self.legend.addItem(self.Channels[-1], n)
    	self.show_ch1.setText(self.Channels[0].__name__)
    	self.show_ch2.setText(self.Channels[1].__name__)
    	self.ch1_mesh.setText(self.Channels[0].__name__)
    	self.ch2_mesh.setText(self.Channels[1].__name__)

    	self.dataLoaded = True
              
    	if self.viewBox_tickBox.checkState() == 2:
            self.make3DROI()


    def updateClusterData(self):
    	'''filter data points by cluster labels'''
    	#clusters & noise    	
    	if self.dataDisplayed == 'cluster':
    	    	self.clusterChannels = [] 

    	    	ch1_Name, ch1_pts, ch1_color = self.Channels[0].filterPts(self.ch1_labels)     
    	    	ch2_Name, ch2_pts, ch2_color = self.Channels[1].filterPts(self.ch2_labels) 
              
    	    	self.clusterChannels.append(Channel(ch1_Name, ch1_pts, ch1_color))
    	    	self.clusterChannels.append(Channel(ch2_Name, ch2_pts, ch2_color)) 

    	else:
    	    	#combined clusters        
    	    	self.combinedClusterChannels = []
    	    	ch1_Name = self.Channels[0].__name__
    	    	ch2_Name = self.Channels[1].__name__
    	    	ch1_color = self.Channels[0].__color__
    	    	ch2_color = self.Channels[1].__color__                
    	    	ch1_list = []
    	    	ch2_list = []
 	        
    	    	for roi in self.plotWidget.items():
    	    	    	if isinstance(roi, Freehand) and hasattr(roi, 'synapse_data'):	    	
    	    	            ch1_ROI_pts, ch2_ROI_pts = self.getROI_pts(roi)
    	    	            ch1_list.extend(ch1_ROI_pts)
    	    	            ch2_list.extend(ch2_ROI_pts)

    	    	self.combinedClusterChannels.append(Channel(ch1_Name, np.array(ch1_list), ch1_color))
    	    	self.combinedClusterChannels.append(Channel(ch2_Name, np.array(ch2_list), ch2_color))          
    	return


    def refreshData(self):
    	'''Update main window with either clustered or all points '''
        
    	if self.dataDisplayed == 'original':        
        	self.plotWidget.clear()
        	self.plotWidget.addItem(self.Channels[0])
        	self.plotWidget.addItem(self.Channels[1])            
      
    	if self.dataDisplayed == 'cluster':  
        	self.updateClusterData()
        	self.plotWidget.clear()
        	self.plotWidget.addItem(self.clusterChannels[0])
        	self.plotWidget.addItem(self.clusterChannels[1])
             
        
    	if self.dataDisplayed == 'combined':   
        	self.updateClusterData()
        	self.plotWidget.clear()
        	self.plotWidget.addItem(self.combinedClusterChannels[0])
        	self.plotWidget.addItem(self.combinedClusterChannels[1]) 

    	return
        

    def make3DROI(self):        
    	#3D window ROI
    	#print('min Xc: {}, max Xc: {} | min Yc: {}, max Yc: {}'.format(min(data['Xc']),max(data['Xc']),min(data['Yc']),max(data['Yc'])))          
    	self.ROI_3Dview = pg.RectROI([min(self.data['Xc']), min(self.data['Yc'])], [5000,5000], pen='r')
    	self.plotWidget.addItem(self.ROI_3Dview)
    	self.viewerDock.show()
    	self.viewerDock.float()
    	self.viewerDock.window().setGeometry(20, 100, 500, 500)   
    	self.update3D_viewer()  
    	self.ROI_3Dview.sigRegionChanged.connect(self.update3D_viewer)  
    	self.ROIState = self.ROI_3Dview.getState()
    	self.ROI3D_Initiated = True
    	return    

    def update3D_viewer(self):
    	if self.viewBox_tickBox.checkState == False:
    		return     	       
    	self.viewerWidget.clear()

    	x = self.ROI_3Dview.parentBounds().x()
    	y = self.ROI_3Dview.parentBounds().y()
    	        
    	xSize = self.ROI_3Dview.parentBounds().width()        
    	ySize = self.ROI_3Dview.parentBounds().height()  
        
    	ch1 =self.Channels[0].getPoints(z=True)
    	ch2 =self.Channels[1].getPoints(z=True)          	  

    	#filter x
    	displayCh1 = ch1[(ch1[:,0] > x)]
    	displayCh2 = ch2[(ch2[:,0] > x)]
    	displayCh1 = displayCh1[(displayCh1[:,0] < (x+xSize))]
    	displayCh2 = displayCh2[(displayCh2[:,0] < (x+xSize))]
        
    	#filter y
    	displayCh1 = displayCh1[(displayCh1[:,1] > y)]
    	displayCh2 = displayCh2[(displayCh2[:,1] > y)]
    	displayCh1 = displayCh1[(displayCh1[:,1] < y+ySize)]
    	displayCh2 = displayCh2[(displayCh2[:,1] < y+ySize)]        

    	self.viewerWidget.addArray(displayCh1,color=QColor(255, 0, 0)) 
    	self.viewerWidget.addArray(displayCh2,color=QColor(0, 255, 0))    	        

    
    def show_mesh(self,i):
    	if i == None:
    		self.synapseWidget.synapse.mesh.setVisible(False)
    		self.synapseWidget.synapse.update()
    		return
    	else:
    		ps = self.synapseWidget.synapse.channels[self.Channels[i].__name__].getPoints(True)
    		c = self.synapseWidget.synapse.channels[self.Channels[i].__name__].color()
    	if len(ps) == 0:
    		return
    	tri = ConvexHull(ps)
    	self.synapseWidget.synapse.mesh.setColor(c)
    	self.synapseWidget.synapse.mesh.setMeshData(vertexes=ps, faces=tri.simplices)
    	self.synapseWidget.synapse.mesh.meshDataChanged()
    	self.synapseWidget.synapse.mesh.setVisible(True)
    	self.synapseWidget.synapse.update()
    
    def hide_channel(self,i):
    	#global Channels
    	self.Channels[0].setVisible(True)
    	self.Channels[1].setVisible(True)
    	try:  
    	    	self.clusterChannels[0].setVisible(True)
    	    	self.clusterChannels[1].setVisible(True)
    	except:    
    	    	pass            
                
    	try:
    	    	self.combinedClusterChannels[0].setVisible(True)  
    	    	self.combinedClusterChannels[1].setVisible(True)         
    	except:    
    	    	pass            
        
    	if i != None:
    		self.Channels[i].setVisible(False)
            
    		try:  
    	    	    	self.clusterChannels[i].setVisible(False)
    		except:    
    	    	    	pass            
                
    		try:
    	    	    	self.combinedClusterChannels[i].setVisible(False)        
    		except:    
    	    	    	pass             
            
            
    	self.connectROIMeanLines(connect=False)

    def connectROIMeanLines(self,connect=True):
        for roi in self.plotWidget.items():
            if isinstance(roi, Freehand) and hasattr(roi, 'synapse_data'):
                roi.mean_line.setVisible(connect)
                if connect :
                    roi.sigHoverChanged.connect(lambda r, h: r.mean_line.setVisible(True) if hasattr(r, 'mean_line') else None) 
                else:
                    roi.sigHoverChanged.connect(lambda r, h: r.mean_line.setVisible(h) if hasattr(r, 'mean_line') else None) 
        
    def show_centroids(self):
    	self.Channels[0].setVisible(False)
    	self.Channels[1].setVisible(False)    
    	self.connectROIMeanLines(connect=True)       
    
    def clear(self):
    	#global Channels
    	for ch in self.Channels:
    		self.legend.removeItem(ch.__name__)
    	self.Channels = []
    	self.plotWidget.clear()
    	for i in self.plotWidget.getViewBox().addedItems:
    		if isinstance(i, Freehand):
    			i.delete()

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
            

    def plotHull(self,points,hull): 
        plt.plot(points[:,0], points[:,1], 'o')          
        for simplex in hull:
            #print(points[simplex, 0],points[simplex, 1])
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
            plt.show()            
                        
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

        # print('\n-----------------')
        # print(':::TIME:::: Points list made: {0:1.3f}'.format(t.stop()))
        # print('-----------------\n')
        
        #t.start()
        #order points list
        pointsList = order_points(pointsList)
        
        # print('\n-----------------')
        # print(':::TIME:::: Points list ordered: {0:1.3f}'.format(t.stop()))
        # print('-----------------\n')


        #t.start()
        #convert list to np array
        pointsList = np.array(pointsList)
        
        # print('\n-----------------')
        # print(':::TIME:::: Array made: {0:1.3f}'.format(t.stop()))
        # print('-----------------\n')        


        #t.start()        
        #add create ROIs from points
        self.plotWidget.getViewBox().createROIFromPoints(pointsList)

        print('\n-----------------')
        print(':::TIME:::: ROI made: {0:1.3f}'.format(t.stop()))
        print('-----------------\n')         

        
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
        self.ch1_labels,self.ch1_numClusters,self.ch1_numNoise = dbscan(ch1Points_3D, eps=self.eps, min_samples=self.min_samples, plot=False)
        print('--- channel 2 ---')
        self.ch2_labels,self.ch2_numClusters,self.ch2_numNoise  = dbscan(ch2Points_3D, eps=self.eps, min_samples=self.min_samples, plot=False)    
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
        if self.clusterType == '3D': 
            ch1_hulls, ch1_centeroids, ch1_groupPoints = self.getHulls(ch1Points_3D,self.ch1_labels)
            #self.plotHull(ch1_groupPoints[0],ch1_hulls[0])
            ch2_hulls, ch2_centeroids, ch2_groupPoints = self.getHulls(ch2Points_3D,self.ch2_labels)
        else:
            ch1_hulls, ch1_centeroids, ch1_groupPoints = self.getHulls(ch1Points,self.ch1_labels)
            #self.plotHull(ch1_groupPoints[0],ch1_hulls[0])
            ch2_hulls, ch2_centeroids, ch2_groupPoints = self.getHulls(ch2Points,self.ch2_labels)

        print('\n-----------------')        
        print(':::TIME:::: hulls created: {0:1.3f}'.format(t.stop()))         
        print('-----------------\n')  
        
        #combine nearest roi between channels
        t.start()
        combinedHulls, combinedPoints, self.combined_ch1_Centeroids, self.combined_ch2_Centeroids = combineClosestHulls(ch1_hulls,ch1_centeroids,ch1_groupPoints,ch2_hulls,ch2_centeroids,ch2_groupPoints, self.maxDistance)

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
        

        if self.multiThreadingFlag:
        # #multi-thread
            t2 = Timer()
            t2.start()
            self.threadpool = QThreadPool()
            print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
    
    
            def progress_fn(n):
                print("%d%% done" % n)
            
            def makeROIs(progress_callback):
                for i in range(len(combinedHulls)):
                    self.createROIFromHull(combinedPoints[i],newHulls[i])
                    progress_callback.emit((i/len(combinedHulls))*100)
                return "Done."
    
            def thread_complete():
                print("THREAD COMPLETE! - time taken:{0:1.3f}".format(t2.stop()))
    
            
            # Pass the function to execute
            worker = Worker(makeROIs) # Any other args, kwargs are passed 
            worker.signals.finished.connect(thread_complete)
            worker.signals.progress.connect(progress_fn)                
            #start threads
            self.threadpool.start(worker)

        else:
        #single thread
            for i in range(len(combinedHulls)):
                self.createROIFromHull(combinedPoints[i],newHulls[i]) ### THIS IS SLOW! ###
      
        
        print('\n-----------------')                 
        print(':::TIME:::: ROI created: {0:1.3f}'.format(t.stop()))       
        print('-----------------\n')        
        
        print('{} ROI created'.format(str(len(combinedHulls))))
        g.m.statusBar().showMessage('{} ROI created'.format(str(len(combinedHulls))))
        
        #self.updateClusterData()
        t.start()
        if len(combinedHulls) > 0:
            self.clustersGeneated = True

        print('\n-----------------')              
        print(':::TIME:::: ROI data updated: {0:1.3f}'.format(t.stop()))       
        print('-----------------\n')          
        
        return


    def saveROIs(self):
        self.plotWidget.getViewBox().export_rois()
        print('ROIs saved')
        return


    def loadROIs(self):
        self.plotWidget.getViewBox().import_rois()
        self.clustersGeneated = True
        print('ROIs loaded')
        return

    def clearClusters(self):
        while (len(self.plotWidget.getViewBox().addedItems)) > 2:
            for i in self.plotWidget.getViewBox().addedItems:
                if isinstance(i, Freehand):
                    i.delete()
        self.clustersGeneated = False
        return

    def clusterOptions(self):
        self.clusterOptionDialog = ClusterOptions_win(self)
        self.clusterOptionDialog.show()
        return

    def show_scaleBar(self,value):
        self.synapseWidget.synapse.scaleBar_x.setVisible(value)
        self.synapseWidget.synapse.scaleBar_y.setVisible(value)
        self.synapseWidget.synapse.scaleBar_z.setVisible(value)        
        self.synapseWidget.synapse.update()        
        return

    def show_viewBox(self,value):
        if self.dataLoaded == False:
            return
        
        if value == False:
            #self.viewerDock.hide()
            self.viewerDock.close()
            self.ROIState = self.ROI_3Dview.getState()
            self.ROI_3Dview.parentItem().getViewBox().removeItem(self.ROI_3Dview)

        else:

            self.viewerDock.show()
            self.viewerDock.float()
            
            if self.ROI3D_Initiated == False:
                self.make3DROI()
            
            else:
                self.ROI_3Dview = pg.RectROI([0, 0], [5000,5000], pen='r')
                self.ROI_3Dview.setState(self.ROIState)            
                self.plotWidget.addItem(self.ROI_3Dview)
                self.ROI_3Dview.sigRegionChanged.connect(self.update3D_viewer)             
           
        return

    def viewerFrameCloseAction(self):
        print('Viewer closed')
        #self.viewerDock.close()
        self.ROIState = self.ROI_3Dview.getState()
        self.ROI_3Dview.parentItem().getViewBox().removeItem(self.ROI_3Dview)
        self.viewBox_tickBox.stateChanged.disconnect(self.show_viewBox)
        self.viewBox_tickBox.setChecked(False)
        self.viewBox_tickBox.stateChanged.connect(self.show_viewBox)
        return

    def toggleNoise(self):
        '''Toggle man window view between data points identified as noise by clustering and all imported points'''
        #oldSetting = self.dataDisplayed
        if self.dataLoaded == False:
            print('No data loaded')
            g.m.statusBar().showMessage('No data loaded')
            return
        
        if self.clustersGeneated:
            #print('toggle: ', self.dataDisplayed)
            if self.dataDisplayed == 'original' or self.dataDisplayed == 'combined':
                self.dataDisplayed = 'cluster'
            else:
                self.dataDisplayed = 'original'                                      
            self.refreshData()
        
        else:
            print('no clusters!')
            g.m.statusBar().showMessage('no clusters!')
        return

    def toggleClusters(self):
        '''Toggle main window view between data points in ROIs and all imported points'''
        #oldSetting = self.dataDisplayed
        if self.dataLoaded == False:
            print('No data loaded')
            g.m.statusBar().showMessage('No data loaded')
            return
        
        if self.clustersGeneated:
            #print('toggle: ', self.dataDisplayed)
            if self.dataDisplayed == 'original' or self.dataDisplayed == 'cluster':
                self.dataDisplayed = 'combined'
            else:
               self.dataDisplayed = 'original'
            self.refreshData()
        
        else:
            print('no clusters!')
            g.m.statusBar().showMessage('no clusters!')
        return

    def dictToCSV(self, dict_data_A, dict_data_B, filePath = "export.csv"):
        try:                        
               df_A = pd.DataFrame.from_dict(dict_data_A)
               df_B = pd.DataFrame.from_dict(dict_data_B)
               df = df_A.append(df_B)
               #self.colNames = ['Channel Name', 'X', 'Y', 'Xc', 'Yc', 'Height', 'Area', 'Width', 'Phi', 'Ax', 'BG', 'I', 'Frame', 'Length', 'Link', 'Valid', 'Z', 'Zc', 'Photons', 'Lateral Localization Accuracy', 'Xw', 'Yw', 'Xwc', 'Ywc', 'Zw', 'Zwc']
               df = df = df.ix[:, self.colNames]
               #print(df.head())                                    
               export_csv = df.to_csv (filePath, index = None, header=True)
               print('Export finished')
               g.m.statusBar().showMessage('Export finished')                                 
        except IOError:
                print("I/O error")                
        return



    def exportWindow(self):
        '''Export data points shown in main window'''
        #Get saveName
        saveName = getSaveFilename()
        
        #Get main window data
        if self.dataDisplayed == 'original':
            data1 = self.Channels[0].getDataAsDict() 
            data2 = self.Channels[1].getDataAsDict()             
        
        if self.dataDisplayed == 'cluster':
            data1 = self.clusterChannels[0].getDataAsDict()  
            data2 = self.clusterChannels[1].getDataAsDict()              
        
        if self.dataDisplayed == 'combined':            
            data1 = self.combinedClusterChannels[0].getDataAsDict()
            data2 = self.combinedClusterChannels[1].getDataAsDict()            
        
        #Export     
        self.dictToCSV(data1,data2,filePath=saveName)
        return

    
    def clusterAnalysis(self):
        if self.clustersGeneated != True:
            print('no clusters!')
            g.m.statusBar().showMessage('no clusters!')
            return

        if self.clusterAnaysisSelection == 'Paired Clusters':            
            #print(self.clusterChannels[0].getDataAsDict())
            #print(self.Channels[0].getCenter())
            #print(self.dataWidget.getData())
            ch1_centeroids = []
            ch2_centeroids = []
            for roi in self.plotWidget.items():
                if isinstance(roi, Freehand) and hasattr(roi, 'synapse_data'):
                    ch1_centeroids.append(roi.synapse_data['%s Centeroid' % (self.Channels[0].__name__)])
                    ch2_centeroids.append(roi.synapse_data['%s Centeroid' % (self.Channels[1].__name__)])


        else:
            ch1_centeroids = self.ch1_centeroids_3D
            ch2_centeroids = self.ch2_centeroids_3D
                
        ch1_centeroids = np.array(ch1_centeroids)
        ch2_centeroids = np.array(ch2_centeroids)
        dist_clusters = self.getNearestNeighbors(ch1_centeroids,ch2_centeroids)
        #print(dist_clusters)        
        #print(min(self.data['Xc']), min(self.data['Yc']), min(self.data['Zc']))
        #print(max(self.data['Xc']), max(self.data['Yc']), max(self.data['Zc']))
        ch1_random = self.getRandomXYZ(min(self.data['Xc']),
                                       min(self.data['Yc']),
                                       min(self.data['Zc']),
                                       max(self.data['Xc']),
                                       max(self.data['Yc']),
                                       max(self.data['Zc']), len(ch1_centeroids))


        ch2_random = self.getRandomXYZ(min(self.data['Xc']),
                                       min(self.data['Yc']),
                                       min(self.data['Zc']),
                                       max(self.data['Xc']),
                                       max(self.data['Yc']),
                                       max(self.data['Zc']), len(ch2_centeroids))


        dist_random = self.getNearestNeighbors(ch1_random,ch2_random)

        distAll_clusters = self.getNearestNeighbors(ch1_centeroids,ch2_centeroids, k=len(ch1_centeroids))
        distAll_random = self.getNearestNeighbors(ch1_random,ch2_random, k=len(ch1_random))

        
        fig = plt.figure()
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.scatter(ch1_centeroids[::,0], ch1_centeroids[::,1], ch1_centeroids[::,2], marker='o')
        ax1.scatter(ch2_centeroids[::,0], ch2_centeroids[::,1], ch2_centeroids[::,2], marker='^')

        ax1.set_title('Cluster Centeroids') 
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
    
        ax3 = fig.add_subplot(234, projection='3d')
        ax3.scatter(ch1_random[::,0], ch1_random[::,1], ch1_random[::,2], marker='o')
        ax3.scatter(ch2_random[::,0], ch2_random[::,1], ch2_random[::,2], marker='^')

        ax3.set_title('Random Points')        
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        

        ax2 = fig.add_subplot(232)
        ax2.hist(dist_clusters)
        ax2.set_title('Nearest Neighbor')
        ax2.set_ylabel('# of observations')
        ax2.set_xlabel('distance')        
        
        ax5 = fig.add_subplot(233)
        ax5.hist(distAll_clusters)
        ax5.set_title('All Neighbors')
        ax5.set_ylabel('# of observations')
        ax5.set_xlabel('distance') 
        
        ax4 = fig.add_subplot(235)
        ax4.hist(dist_random)
        #ax4.set_title('Nearest Neighbor')
        ax4.set_ylabel('# of observations')
        ax4.set_xlabel('distance')  

        ax6 = fig.add_subplot(236)
        ax6.hist(distAll_random)
        #ax6.set_title('All Neighbors')
        ax6.set_ylabel('# of observations')
        ax6.set_xlabel('distance') 
       
        plt.show()
        
        #print stats
        print('----------------------------------------------')
        print(self.clusterAnaysisSelection)        
        print('----------------------------------------------')
        print('Channel 1: Number of clusters: ', str(len(ch1_centeroids)))
        print('Channel 2: Number of clusters: ', str(len(ch2_centeroids)))        
        print('Number of nearest neighbor distances:', str(np.size(dist_clusters)))
        print('Mean nearest neighbor distance:', str(np.mean(dist_clusters)))
        print('StDev nearest neighbor distance:', str(np.std(dist_clusters)))        
        print('Number of All distances:', str(np.size(distAll_clusters)))
        print('Mean All distance:', str(np.mean(distAll_clusters)))
        print('StDev All distance:', str(np.std(distAll_clusters)))       
        print('----------------------------------------------')
        print('Random 1: Number of clusters: ', str(len(ch1_random)))
        print('Random 2: Number of clusters: ', str(len(ch2_random)))        
        print('Number of nearest neighbor distances:', str(np.size(dist_random)))
        print('Mean nearest neighbor distance:', str(np.mean(dist_random)))
        print('StDev nearest neighbor distance:', str(np.std(dist_random)))       
        print('Number of All distances:', str(np.size(distAll_random)))
        print('Mean All distance:', str(np.mean(distAll_random)))  
        print('Stdev All distance:', str(np.std(distAll_random)))          
        print('----------------------------------------------')
        
        
        #save centeroids and distances
        d1 = {'clusters_nearest':dist_clusters,'random_nearest':dist_random}
        d2 = {'clusters_All':distAll_clusters,'random_All':distAll_random}
        d3 = {'ch1_centeroids_x':ch1_centeroids[::,0],
              'ch1_centeroids_y':ch1_centeroids[::,1],
              'ch1_centeroids_z':ch1_centeroids[::,2]}
        
        d4 = {'ch2_centeroids_x':ch2_centeroids[::,0],
              'ch2_centeroids_y':ch2_centeroids[::,1],
              'ch2_centeroids_z':ch2_centeroids[::,2]}
        
        d5 = {'ch1_centeroids_rnd_x':ch1_random[::,0],
              'ch1_centeroids_rnd_y':ch1_random[::,1],
              'ch1_centeroids_rnd_z':ch1_random[::,2]}
              
        d6 = {'ch2_centeroids_rnd_x':ch2_random[::,0],
              'ch2_centeroids_rnd_y':ch2_random[::,1],
              'ch2_centeroids_rnd_z':ch2_random[::,2]}
        
        
        nearestNeighborDF = pd.DataFrame(data=d1)
        allNeighborDF = pd.DataFrame(data=d2)   
        ch1_centeroids_clusters_DF = pd.DataFrame(data=d3)  
        ch2_centeroids_clusters_DF = pd.DataFrame(data=d4)                 
        ch1_centeroids_random_DF = pd.DataFrame(data=d5)   
        ch2_centeroids_random_DF = pd.DataFrame(data=d6) 
         
        saveName1 = 'clusterAnalysis_nearestNeighbors.csv'
        saveName2 = 'clusterAnalysis_AllNeighbors.csv'  
        saveName3 = 'ch1_clusters_centeroids.csv'
        saveName4 = 'ch2_clusters_centeroids.csv'        
        saveName5 = 'ch1_random_centeroids.csv'       
        saveName6 = 'ch2_random_centeroids.csv'          
        
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

    def start(self):        
        self.menu = self.win.menuBar()
        
        self.fileMenu =self. menu.addMenu('&File')
        self.fileMenu.addAction(QtWidgets.QAction('&Import Channels', self.fileMenu, triggered = lambda : self.open_file()))
        self.fileMenu.addAction(QtWidgets.QAction('&Save ROIs', self.fileMenu, triggered = lambda : self.saveROIs()))
        self.fileMenu.addAction(QtWidgets.QAction('&Load ROIs', self.fileMenu, triggered = lambda : self.loadROIs()))
        
        self.fileMenu.addAction(QtWidgets.QAction('&Close', self.fileMenu, triggered = self.win.close))
        self.optionMenu =self. menu.addMenu('&Options')
        self.optionMenu.addAction(QtWidgets.QAction('&Settings', self.optionMenu, triggered = lambda : self.clusterOptions()))            
        
        self.plotWidget.getViewBox().roiCreated.connect(self.roiCreated)
        self.plotWidget.load_file = self.open_file
        
        self.opsFrame = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout(self.opsFrame)
        self.show_all = QtWidgets.QRadioButton('Show all', checked=True)
        self.show_all.pressed.connect(lambda : self.hide_channel(None))
        self.show_ch1 = QtWidgets.QRadioButton('Channel 1')
        self.show_ch1.pressed.connect(lambda : self.hide_channel(1))
        self.show_ch2 = QtWidgets.QRadioButton('Channel 2')
        self.show_ch2.pressed.connect(lambda : self.hide_channel(0))
        self.show_cent = QtWidgets.QRadioButton('ROIs only')
        self.show_cent.pressed.connect(lambda : self.show_centroids())                
        self.getClusters_button = QtWidgets.QPushButton('Get Clusters')
        self.getClusters_button.pressed.connect(lambda : self.getClusters()) 
        self.clearClusters_button = QtWidgets.QPushButton('Clear Clusters')
        self.clearClusters_button.pressed.connect(lambda : self.clearClusters()) 
        self.toggleClusters_button = QtWidgets.QPushButton('Toggle Noise')
        self.toggleClusters_button.pressed.connect(lambda : self.toggleNoise()) 
        self.toggleClusters2_button = QtWidgets.QPushButton('Toggle Clusters')
        self.toggleClusters2_button.pressed.connect(lambda : self.toggleClusters()) 
        self.export_button = QtWidgets.QPushButton('Export Window')
        self.export_button.pressed.connect(lambda : self.exportWindow()) 
        self.clusterAnalysis_button = QtWidgets.QPushButton('Cluster Analysis')
        self.clusterAnalysis_button.pressed.connect(lambda : self.clusterAnalysis()) 
                
        self.viewBox_tickLabel = QtWidgets.QLabel('3D Viewer: ')
        self.viewBox_tickBox = QtWidgets.QCheckBox()  
        self.viewBox_tickBox.setChecked(False)
        self.viewBox_tickBox.stateChanged.connect(self.show_viewBox)
        
           
        self.layout.addWidget(self.show_all, 0, 0)
        self.layout.addWidget(self.show_ch1, 0, 1)
        self.layout.addWidget(self.show_ch2, 0, 2)
        self.layout.addWidget(self.show_cent, 0, 3)        
        self.layout.addWidget(self.getClusters_button, 0, 4)    
        self.layout.addWidget(self.clearClusters_button, 0, 5) 
        
        self.layout.addWidget(self.clusterAnalysis_button, 0, 6)    
        
        self.layout.addWidget(self.viewBox_tickLabel, 0, 8) 
        self.layout.addWidget(self.viewBox_tickBox, 0, 9) 
        self.layout.addWidget(self.toggleClusters_button, 0, 10)  
        self.layout.addWidget(self.toggleClusters2_button, 0, 11)
        self.layout.addWidget(self.export_button, 0, 12)           
        self.layout.addItem(QtWidgets.QSpacerItem(400, 20), 0, 3, 1, 8)
        
        self.plotWidget.__name__ = '2D Plotted Channels'
        self.plotDock = self.win.addWidget(self.plotWidget, size=(400, 500))
        self.plotDock.addWidget(self.opsFrame)
        
        self.synapseFrame = QtWidgets.QWidget()
        self.layout = QtWidgets.QGridLayout(self.synapseFrame)
        self.synapseWidget = Plot3DWidget()
        self.synapseWidget.load_file = self.open_file        
        self.layout.addWidget(self.synapseWidget, 0, 0, 6, 6)
        self.export_synapse = QtWidgets.QPushButton('Export Coordinates')
        self.export_synapse.pressed.connect(lambda : export_arr(self.synapseWidget.synapse.pos, header='X\tY\tZ'))
        self.layout.addWidget(self.export_synapse, 6, 0)
        self.no_mesh = QtWidgets.QRadioButton('No Mesh', checked=True)
        self.no_mesh.pressed.connect(lambda : self.show_mesh(None))
        self.ch1_mesh = QtWidgets.QRadioButton('Channel 1')
        self.ch1_mesh.pressed.connect(lambda : self.show_mesh(0))
        self.ch2_mesh = QtWidgets.QRadioButton('Channel 2')
        self.ch2_mesh.pressed.connect(lambda : self.show_mesh(1))
        self.scaleBar_tickLabel = QtWidgets.QLabel('Display Scale Bar')
        self.scaleBar_tickBox = QtWidgets.QCheckBox()         
        self.scaleBar_tickBox.stateChanged.connect(self.show_scaleBar)        
        self.layout.addWidget(self.no_mesh, 6, 1)
        self.layout.addWidget(self.ch1_mesh, 6, 2)
        self.layout.addWidget(self.ch2_mesh, 6, 3)
        self.layout.addWidget(self.scaleBar_tickLabel, 6, 4)
        self.layout.addWidget(self.scaleBar_tickBox, 6, 5)        
        
        self.layout.addWidget(QtWidgets.QLabel(\
        '''
        ROI Plot 3D Widget Controls
        	Arrow Keys or Left click and drag to rotate camera
        	Middle click and drag to pan
        	Scroll mouse wheel to zoom
        	Right Click for plotted item options
        '''), 7, 0, 2, 3)
        
        self.synapseDock = self.win.addWidget(size=(300, 100), widget=self.synapseFrame)
        self.synapseDock.hide()
        self.plotDock.window().setGeometry(340, 100, 1400, 800)
        
        self.dataWidget = DataWidget()
        
        
        #################################################
        #3D viewer dock
        self.viewerFrame = QtWidgets.QWidget()
        self.layout2 = QtWidgets.QGridLayout(self.viewerFrame)
        self.viewerWidget = Plot3DWidget()
        self.viewerWidget.load_file = self.open_file        
        self.layout2.addWidget(self.viewerWidget, 0, 0, 6, 6)
        #self.export_viewer = QtWidgets.QPushButton('Export Coordinates')
        #self.export_viewer.pressed.connect(lambda : export_arr(self.viewerWidget.synapse.pos, header='X\tY\tZ'))
        #self.layout2.addWidget(self.export_viewer, 6, 0)
        #self.no_mesh = QtWidgets.QRadioButton('No Mesh', checked=True)
        #self.no_mesh.pressed.connect(lambda : self.show_mesh(None))
        #self.ch1_mesh = QtWidgets.QRadioButton('Channel 1')
        #self.ch1_mesh.pressed.connect(lambda : self.show_mesh(0))
        #self.ch2_mesh = QtWidgets.QRadioButton('Channel 2')
        #self.ch2_mesh.pressed.connect(lambda : self.show_mesh(1))
        #self.scaleBar_tickLabel = QtWidgets.QLabel('Display Scale Bar')
        #self.scaleBar_tickBox = QtWidgets.QCheckBox()         
        #self.scaleBar_tickBox.stateChanged.connect(self.show_scaleBar)        
        #self.layout2.addWidget(self.no_mesh, 6, 1)
        #self.layout2.addWidget(self.ch1_mesh, 6, 2)
        #self.layout2.addWidget(self.ch2_mesh, 6, 3)
        #self.layout2.addWidget(self.scaleBar_tickLabel, 6, 4)
        #self.layout2.addWidget(self.scaleBar_tickBox, 6, 5)        
        
        self.layout2.addWidget(QtWidgets.QLabel(\
        '''
        ROI Plot 3D Widget Controls
        	Arrow Keys or Left click and drag to rotate camera
        	Middle click and drag to pan
        	Scroll mouse wheel to zoom
        	Right Click for plotted item options
        '''), 7, 0, 2, 3)
        
        self.viewerDock = self.win.addWidget(size=(300, 100), widget=self.viewerFrame)
        self.viewerDock.hide()

        #TODO Get this to work
        #self.viewerDock.close = lambda : self.viewerFrameCloseAction()

        #####################################################################################

        self.win.addWidget(self.dataWidget, where=('right', self.plotDock), size=(100, 500))
        #self.win.closeEvent = lambda f: self.app.exit() 
        self.win.closeEvent = lambda f: self.app.closeAllWindows()
        self.win.show()
        #sys.exit(self.app.exec_())


class ClusterOptions_win(QtWidgets.QDialog):
    def __init__(self, viewerInstance, parent = None):
        super(ClusterOptions_win, self).__init__(parent)

        self.viewer = viewerInstance
        self.eps  = self.viewer.eps
        self.min_samples = self.viewer.min_samples
        self.maxDistance = self.viewer.maxDistance
        self.unitPerPixel = self.viewer.unitPerPixel
        self.centroidSymbolSize = self.viewer.centroidSymbolSize
        self.multiThreadingFlag = self.viewer.multiThreadingFlag
        
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
        
        self.multiThreadTitle = QtWidgets.QLabel("----- Multi-Threading -----") 
        self.label_multiThread = QtWidgets.QLabel("Multi-Threading On: ")        
        
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
        
        #tickbox
        self.multiThread_checkbox = CheckBox()
        self.multiThread_checkbox.setChecked(self.multiThreadingFlag)
        self.multiThread_checkbox.stateChanged.connect(self.multiThreadClicked)


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
        
        layout.addWidget(self.multiThreadTitle, 10, 0, 1, 2)  
        layout.addWidget(self.label_multiThread, 11, 0)  
        layout.addWidget(self.multiThread_checkbox, 11, 1)          

        
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
    
    def multiThreadClicked(self):
        self.viewer.multiThreadingFlag = self.multiThread_checkbox.isChecked()
        return