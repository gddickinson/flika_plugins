"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
import os,sys,inspect
from scipy.spatial import Delaunay
import pyqtgraph.console
#from .BioDocks.Channels import ActivePoint, Channel
#from .BioDocks import *
#from .BioDocks.DockWindow import * 
#from .BioDocks.ClusterMath import *
try:
    from BioDocks import *
except:
    from .BioDocks import *

from pyqtgraph.dockarea import *
from collections import OrderedDict
#from PyQt4.QtCore import *
#from PyQt4.QtGui import *
from qtpy import QtWidgets, QtCore, QtGui

import flika
from flika import global_vars as g
from flika.window import Window
from distutils.version import StrictVersion
import numpy as np

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

class Synapse(BaseProcess):

    def __init__(self):
        super().__init__()
        
        self.app = QtWidgets.QApplication([])
        
        self.win = QtWidgets.QMainWindow()
        self.win.resize(1700, 900)
        self.dockArea = DockArea()
        self.win.setCentralWidget(self.dockArea)
        self.win.setWindowTitle('Main Window')
        self.plotWidget = pg.PlotWidget(viewBox=ROIViewBox(creatingROI=True, roiSnap=False))
        
        self.units = {'Pixels': 166, 'Nanometers': 1}
        self.unit_prefixes = {'Pixels': 'px', 'Nanometers': 'nm'}
        self.unit = 'Nanometers'
        
        self.ignore = {'Z Rejected'}
        
        self.yAxis = self.plotWidget.getPlotItem().getAxis('left')
        self.xAxis = self.plotWidget.getPlotItem().getAxis('bottom')
        self.yAxis.setLabel(text='Y', units=self.unit)
        self.yAxis.enableAutoSIPrefix(False)
        self.xAxis.enableAutoSIPrefix(False)
        self.xAxis.setLabel(text='X', units=self.unit)
        
        self.channel_colors = [(255, 0, 0), (0, 255, 0)]
        self.legend = self.plotWidget.addLegend()
        
        self.Channels = []
        
    def clear(self):
    	#global Channels
    	for ch in self.Channels:
    		legend.removeItem(ch.__name__)
    	self.Channels = []
    	self.plotWidget.clear()
    
    def import_channels(self,filename=''):
    	self.clear()
    	if filename == '':
    		filename = getFilename('Select the text file of the channels to import')
    	if filename == '':
    		return
    	data = fileToArray(filename)
    	data['Channel Name'] = data['Channel Name'].astype(str)
    	data['Xc'] /= self.units[self.unit]
    	data['Yc'] /= self.units[self.unit]
    	channel_names = {str(i) for i in data['Channel Name']} - self.ignore
    	assert len(channel_names) == 2, 'Must provide only 2 channels, channels are %s' % channel_names
    	pts = [ActivePoint({k: data[k][i] for k in data}) for i in range(len(data['Channel Name']))]
    
    	for i, ch in enumerate(channel_names):
    		item = Channel(name=ch, points=[p for p in pts if p['Channel Name'] == ch], color=self.channel_colors[i])
    		self.plotWidget.addItem(item, name=ch)
    		self.Channels.append(item)
    		self.legend.addItem(item, ch)
    
    def displayData(self):
    	self.synapseWidget.setData(sorted([roi.synapse_data for roi in self.plotWidget.items() if isinstance(roi, Freehand) and hasattr(roi, 'synapse_data')], key=lambda f: f['ROI #']))
    
    def subchannels_in_roi(self,roi):
    	channels = []
    	for ch in self.Channels:
    		pts_in = []
    		for syn_pt in ch.pts:
    			if roi.contains(QtCore.QPointF(float(syn_pt[0]), float(syn_pt[1]))):
    				pts_in.append(syn_pt)
    		channels.append(Channel(ch.__name__, pts_in, ch.color()))
    	return channels
    
    def analyze_roi(self,roi):
    	channels = self.subchannels_in_roi(roi)
    	roi.synapse_data = OrderedDict([('ROI #', roi.id), ('Mean Distance (%s)' % self.unit_prefixes[self.unit], 0), ('%s N' % self.Channels[0].__name__, 0), \
    	('%s N' % self.Channels[1].__name__, 0), ('%s Area (%s^2)' % (self.Channels[0].__name__, self.unit_prefixes[self.unit]), 0), ('%s Area (%s^2)' % (self.Channels[1].__name__, self.unit_prefixes[self.unit]), 0)])
    
    	for i, ch in enumerate(channels):
    		roi.synapse_data['%s N' % ch.__name__] = ch.getCount()
    		if ch.getCount() >= 3:
    			roi.synapse_data['%s Area (%s^2)' % (ch.__name__, self.unit_prefixes[self.unit])] = concaveArea(ch.getPoints())
    		else:
    			print('Cannot get area of %s in roi %d with %d points' % (ch.__name__, roi.id, ch.getCount()))
    
    	if hasattr(roi, 'mean_line'):
    		self.plotWidget.removeItem(roi.mean_line)
    
    	if all([ch.getCount() > 0 for ch in channels]):
    		roi.synapse_data['Mean Distance (%s)' % self.unit_prefixes[self.unit]] = np.linalg.norm(channels[1].getCenter() - channels[0].getCenter())
    		roi.mean_line = pg.PlotDataItem(np.array([channels[1].getCenter(), channels[0].getCenter()]), symbol='d')
    		roi.mean_line.setParentItem(roi)
    		roi.mean_line.setVisible(False)
    	else:
    		del roi.synapse_data
    		print('Must select exactly 2 channels to calculate distance. Ignoring ROI %d' % roi.id)
    
    	self.displayData()
    
    def show_line(self,roi, hover):
    	if hasattr(roi, 'mean_line'):
    		roi.mean_line.setVisible(hover)
    
    def connect_roi(self,roi):
    	roi.sigChanged.connect(self.analyze_roi)
    	roi.sigRemoved.connect(lambda : self.displayData())
    	roi.sigHoverChanged.connect(self.show_line)
    	roi.sigClicked.connect(self.analyze_roi)
    	self.analyze_roi(roi)

    def start(self):         
        self.menu = self.win.menuBar()
        
        self.fileMenu = QtWidgets.QMenu('&File', self.menu)
        self.fileMenu.addAction(QtWidgets.QAction('&Import Channels', self.fileMenu, triggered = lambda : self.import_channels()))
        self.fileMenu.addAction(QtWidgets.QAction('&Close', self.fileMenu, triggered = self.win.close))
        self.menu.addMenu(self.fileMenu)
        self.plotWidget.load_file = self.import_channels
        self.plotDock = WidgetDock(name='Plot Dock', size=(500, 400), widget=self.plotWidget)
        self.dockArea.addDock(self.plotDock)
        
        self.synapseWidget = DataWidget()
        self.synapseWidget.setFormat("%3.3f")
        self.synapseDock = Dock(name='MeanXY Distances', widget=self.synapseWidget)
        self.dockArea.addDock(self.synapseDock, 'right', self.plotDock)
        
        self.plotWidget.getViewBox().roiCreated.connect(self.connect_roi)
        
        self.win.show()
        #sys.exit(app.exec_())
