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

from matplotlib import pyplot as plt
from .dbscan import *

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
        
        # data is loaded in nanometers, divided by # according to units
        self.units = {'Pixels': 166, 'Nanometers': 1}
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
        self.eps = 400
        self.min_samples = 10
        self.maxDistance = 400
        
        
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
    		roi.mean_line = pg.PlotDataItem(np.array([channels[1].getCenter(), channels[0].getCenter()]), symbol='d')
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
    	data = importFile(filename,evaluateLines=False)
    	data = {d[0]: d[1:] for d in np.transpose(data)}
    	for k in data:
    		if k != 'Channel Name':
    			data[k] = data[k].astype(float)
    	print('Gathering channels...')
    	g.m.statusBar().showMessage('Gathering channels...')        
    	names = set(data['Channel Name'].astype(str)) - self.ignore
    	print('Channels Found: %s' % ', '.join(names))
    	g.m.statusBar().showMessage('Channels Found: %s' % ', '.join(names))
    
    	data['Xc'] /= self.units[self.unit]
    	data['Yc'] /= self.units[self.unit]
    	data['Zc'] /= self.units[self.unit]
    
    	#global Channels
    	self.Channels = []
    	self.plotWidget.clear()
    	pts = [ActivePoint(data={k: data[k][i] for k in data}) for i in range(len(data['Channel Name']))]
    	for i, n in enumerate(names):
    		if n in self.color_dict:
    			color = self.color_dict[n]
    		else:
    			color = self.colors[i]
    		self.Channels.append(Channel(n, [p for p in pts if p['Channel Name'] == n], color))
    		self.plotWidget.addItem(self.Channels[-1])
    		self.legend.addItem(self.Channels[-1], n)
    	self.show_ch1.setText(self.Channels[0].__name__)
    	self.show_ch2.setText(self.Channels[1].__name__)
    	self.ch1_mesh.setText(self.Channels[0].__name__)
    	self.ch2_mesh.setText(self.Channels[1].__name__)
    
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
    	if i != None:
    		self.Channels[i].setVisible(False)
    
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
        ch1_labels = dbscan(ch1Points, eps=self.eps, min_samples=self.min_samples, plot=False)
        print('--- channel 2 ---')
        ch2_labels = dbscan(ch2Points, eps=self.eps, min_samples=self.min_samples, plot=False)    
        print('-----------------')
        #get hulls for each channels clusters
        ch1_hulls, ch1_centeroids, ch1_groupPoints = self.getHulls(ch1Points,ch1_labels)
        print('number of channel 1 hulls: {}'.format(len(ch1_hulls)))
        #self.plotHull(ch1_groupPoints[0],ch1_hulls[0])
        ch2_hulls, ch2_centeroids, ch2_groupPoints = self.getHulls(ch2Points,ch2_labels)
        print('number of channel 2 hulls: {}'.format(len(ch2_hulls)))
        #combine nearest roi between channels
        combinedHulls, combinedPoints = combineClosestHulls(ch1_hulls,ch1_centeroids,ch1_groupPoints,ch2_hulls,ch2_centeroids,ch2_groupPoints, self.maxDistance)
        self.plotHull(combinedPoints[0],combinedHulls[0])       
        #draw rois around combined hulls
        #self.createROIFromHull(combinedPoints[0],combinedHulls[0])
        for i in range(len(combinedHulls)):
            self.createROIFromHull(combinedPoints[i],combinedHulls[i])    #TODO Ordering of roi points needs to be fixed    
        return

    def clusterOptions(self):
        self.clusterOptionDialog = ClusterOptions_win(self)
        self.clusterOptionDialog.show()
        return

    def start(self):        
        self.menu = self.win.menuBar()
        
        self.fileMenu =self. menu.addMenu('&File')
        self.fileMenu.addAction(QtWidgets.QAction('&Import Channels', self.fileMenu, triggered = lambda : self.open_file()))
        self.fileMenu.addAction(QtWidgets.QAction('&Close', self.fileMenu, triggered = self.win.close))
        self.optionMenu =self. menu.addMenu('&Options')
        self.optionMenu.addAction(QtWidgets.QAction('&Clustering', self.optionMenu, triggered = lambda : self.clusterOptions()))            
        
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
        self.getClusters_button = QtWidgets.QPushButton('Get Clusters')
        self.getClusters_button.pressed.connect(lambda : self.getClusters())               
        self.layout.addWidget(self.show_all, 0, 0)
        self.layout.addWidget(self.show_ch1, 0, 1)
        self.layout.addWidget(self.show_ch2, 0, 2)
        self.layout.addWidget(self.getClusters_button, 0, 3)        
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
        self.layout.addWidget(self.no_mesh, 6, 1)
        self.layout.addWidget(self.ch1_mesh, 6, 2)
        self.layout.addWidget(self.ch2_mesh, 6, 3)
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
        self.win.addWidget(self.dataWidget, where=('right', self.plotDock), size=(100, 500))
        self.win.closeEvent = lambda f: self.app.exit()
        
        self.win.show()
        #sys.exit(self.app.exec_())


class ClusterOptions_win(QtWidgets.QDialog):
    def __init__(self, viewerInstance, parent = None):
        super(ClusterOptions_win, self).__init__(parent)

        self.viewer = viewerInstance
        self.eps  = self.viewer.eps
        self.min_samples = self.viewer.min_samples
        self.maxDistance = self.viewer.maxDistance
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 200

        #labels
        self.label_eps = QtWidgets.QLabel("max distance between points:") 
        self.label_minSamples = QtWidgets.QLabel("minimum number of points:") 
        self.label_maxDistance = QtWidgets.QLabel("max distance between clusters:") 
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
        

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.label_eps, 0, 0)
        layout.addWidget(self.epsBox, 0, 1)       
        layout.addWidget(self.label_minSamples, 1, 0)        
        layout.addWidget(self.minSampleBox, 1, 1)     
        layout.addWidget(self.label_maxDistance, 2, 0)        
        layout.addWidget(self.maxDistanceBox, 2, 1)         

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Clustering Options")

        #connect spinboxes
        self.epsBox.valueChanged.connect(self.epsValueChange)
        self.minSampleBox.valueChanged.connect(self.minSampleChange) 
        self.maxDistance.valueChanged.connect(self.maxDistanceChange)         
        
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

