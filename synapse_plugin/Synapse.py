"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
import os,sys,inspect
from BioDocks import *
from pyqtgraph.Qt import QtCore, QtGui
from scipy.spatial import Delaunay
from pyqtgraph.dockarea import *
from collections import OrderedDict
import pyqtgraph.console
from Channels import ActivePoint, Channel
from ClusterMath import *

app = QtGui.QApplication([])

win = QtGui.QMainWindow()
win.resize(1700, 900)
dockArea = DockArea()
win.setCentralWidget(dockArea)
win.setWindowTitle('Main Window')
plotWidget = pg.PlotWidget(viewBox=ROIViewBox(creatingROI=True, roiSnap=False))

units = {'Pixels': 166, 'Nanometers': 1}
unit_prefixes = {'Pixels': 'px', 'Nanometers': 'nm'}
unit = 'Nanometers'

ignore = {'Z Rejected'}

yAxis = plotWidget.getPlotItem().getAxis('left')
xAxis = plotWidget.getPlotItem().getAxis('bottom')
yAxis.setLabel(text='Y', units=unit)
yAxis.enableAutoSIPrefix(False)
xAxis.enableAutoSIPrefix(False)
xAxis.setLabel(text='X', units=unit)

channel_colors = [(255, 0, 0), (0, 255, 0)]
legend = plotWidget.addLegend()

Channels = []

def clear():
	global Channels
	for ch in Channels:
		legend.removeItem(ch.__name__)
	Channels = []
	plotWidget.clear()

def import_channels(filename=''):
	clear()
	if filename == '':
		filename = getFilename('Select the text file of the channels to import')
	if filename == '':
		return
	data = fileToArray(filename)
	data['Channel Name'] = data['Channel Name'].astype(str)
	data['Xc'] /= units[unit]
	data['Yc'] /= units[unit]
	channel_names = {str(i) for i in data['Channel Name']} - ignore
	assert len(channel_names) == 2, 'Must provide only 2 channels, channels are %s' % channel_names
	pts = [ActivePoint({k: data[k][i] for k in data}) for i in range(len(data['Channel Name']))]

	for i, ch in enumerate(channel_names):
		item = Channel(name=ch, points=[p for p in pts if p['Channel Name'] == ch], color=channel_colors[i])
		plotWidget.addItem(item, name=ch)
		Channels.append(item)
		legend.addItem(item, ch)

def displayData():
	synapseWidget.setData(sorted([roi.synapse_data for roi in plotWidget.items() if isinstance(roi, Freehand) and hasattr(roi, 'synapse_data')], key=lambda f: f['ROI #']))

def subchannels_in_roi(roi):
	channels = []
	for ch in Channels:
		pts_in = []
		for syn_pt in ch.pts:
			if roi.contains(QPointF(syn_pt[0], syn_pt[1])):
				pts_in.append(syn_pt)
		channels.append(Channel(ch.__name__, pts_in, ch.color()))
	return channels

def analyze_roi(roi):
	channels = subchannels_in_roi(roi)
	roi.synapse_data = OrderedDict([('ROI #', roi.id), ('Mean Distance (%s)' % unit_prefixes[unit], 0), ('%s N' % Channels[0].__name__, 0), \
	('%s N' % Channels[1].__name__, 0), ('%s Area (%s^2)' % (Channels[0].__name__, unit_prefixes[unit]), 0), ('%s Area (%s^2)' % (Channels[1].__name__, unit_prefixes[unit]), 0)])

	for i, ch in enumerate(channels):
		roi.synapse_data['%s N' % ch.__name__] = ch.getCount()
		if ch.getCount() >= 3:
			roi.synapse_data['%s Area (%s^2)' % (ch.__name__, unit_prefixes[unit])] = concaveArea(ch.getPoints())
		else:
			print('Cannot get area of %s in roi %d with %d points' % (ch.__name__, roi.id, ch.getCount()))

	if hasattr(roi, 'mean_line'):
		plotWidget.removeItem(roi.mean_line)

	if all([ch.getCount() > 0 for ch in channels]):
		roi.synapse_data['Mean Distance (%s)' % unit_prefixes[unit]] = np.linalg.norm(channels[1].getCenter() - channels[0].getCenter())
		roi.mean_line = pg.PlotDataItem(np.array([channels[1].getCenter(), channels[0].getCenter()]), symbol='d')
		roi.mean_line.setParentItem(roi)
		roi.mean_line.setVisible(False)
	else:
		del roi.synapse_data
		print('Must select exactly 2 channels to calculate distance. Ignoring ROI %d' % roi.id)

	displayData()

def show_line(roi, hover):
	if hasattr(roi, 'mean_line'):
		roi.mean_line.setVisible(hover)

def connect_roi(roi):
	roi.sigChanged.connect(analyze_roi)
	roi.sigRemoved.connect(lambda : displayData())
	roi.sigHoverChanged.connect(show_line)
	roi.sigClicked.connect(analyze_roi)
	analyze_roi(roi)

menu = win.menuBar()

fileMenu = QtGui.QMenu('&File', menu)
fileMenu.addAction(QtGui.QAction('&Import Channels', fileMenu, triggered = lambda : import_channels()))
fileMenu.addAction(QtGui.QAction('&Close', fileMenu, triggered = win.close))
menu.addMenu(fileMenu)
plotWidget.load_file = import_channels
plotDock = WidgetDock(name='Plot Dock', size=(500, 400), widget=plotWidget)
dockArea.addDock(plotDock)

synapseWidget = DataWidget()
synapseWidget.setFormat("%3.3f")
synapseDock = Dock(name='MeanXY Distances', widget=synapseWidget)
dockArea.addDock(synapseDock, 'right', plotDock)

plotWidget.getViewBox().roiCreated.connect(connect_roi)

win.show()
sys.exit(app.exec_())
