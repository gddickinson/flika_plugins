"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
import os,sys
from .BioDocks import *
from .BioDocks.DockWindow import * 
from .BioDocks.AnalysisIO import *
from .BioDocks.Tools import *
from pyqtgraph.dockarea import *
from scipy.spatial import ConvexHull
from collections import OrderedDict
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *
from qtpy import QtWidgets, QtCore, QtGui
from .Channels import *
from .ClusterMath import *
import flika
from flika import global_vars as g
from flika.window import Window
from distutils.version import StrictVersion
import numpy as np
import pyqtgraph as pg
from pyqtgraph import mkPen
from matplotlib import pyplot as plt
from .dbscan import *
import copy
import pandas as pd


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
        '''add roi to display from hull points'''
        pointsList = []     
        for simplex in hull:
            pointsList.append((points[simplex][0])) 
        for simplex in hull:            
            pointsList.append((points[simplex][1]))       
        pointsList = order_points(pointsList)
        pointsList = np.array(pointsList) 
        self.plotWidget.getViewBox().createROIFromPoints(pointsList)
        
    def getClusters(self):
        #get 2D points
        ch1Points = self.Channels[0].getPoints(z=False)
        ch2Points = self.Channels[1].getPoints(z=False)
        #get cluster labels for each channel
        print('--- channel 1 ---')
        self.ch1_labels = dbscan(ch1Points, eps=self.eps, min_samples=self.min_samples, plot=False)
        print('--- channel 2 ---')
        self.ch2_labels = dbscan(ch2Points, eps=self.eps, min_samples=self.min_samples, plot=False)    
        print('-----------------')
        #get hulls for each channels clusters
        ch1_hulls, ch1_centeroids, ch1_groupPoints = self.getHulls(ch1Points,self.ch1_labels)
        #self.plotHull(ch1_groupPoints[0],ch1_hulls[0])
        ch2_hulls, ch2_centeroids, ch2_groupPoints = self.getHulls(ch2Points,self.ch2_labels)
        #combine nearest roi between channels
        combinedHulls, combinedPoints = combineClosestHulls(ch1_hulls,ch1_centeroids,ch1_groupPoints,ch2_hulls,ch2_centeroids,ch2_groupPoints, self.maxDistance)
        #get new hulls for combined points
        newHulls = self.getHulls2(combinedPoints)        
        #self.plotHull(combinedPoints[0],newHulls[0])       
        #draw rois around combined hulls
        #self.createROIFromHull(combinedPoints[0],newHulls[0])
        print('--- combined channels ---')
        for i in range(len(combinedHulls)):
            self.createROIFromHull(combinedPoints[i],newHulls[i])  
        print('{} ROI created'.format(str(len(combinedHulls))))
        g.m.statusBar().showMessage('{} ROI created'.format(str(len(combinedHulls))))
        #self.updateClusterData()
        if len(combinedHulls) > 0:
            self.clustersGeneated = True
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

    
    def getRandomClusters(self):
        return
    
    def clearRandomClusters(self):
        return    


    def start(self):        
        self.menu = self.win.menuBar()
        
        self.fileMenu =self. menu.addMenu('&File')
        self.fileMenu.addAction(QtWidgets.QAction('&Import Channels', self.fileMenu, triggered = lambda : self.open_file()))
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
        self.getRandomClusters_button = QtWidgets.QPushButton('Random Clusters')
        self.getRandomClusters_button.pressed.connect(lambda : self.getRandomClusters()) 
        self.clearRandomClusters_button = QtWidgets.QPushButton('Clear Random')
        self.clearRandomClusters_button.pressed.connect(lambda : self.clearRandomClusters())         
                
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
        
        self.layout.addWidget(self.getRandomClusters_button, 0, 6)    
        self.layout.addWidget(self.clearRandomClusters_button, 0, 7)         
        
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