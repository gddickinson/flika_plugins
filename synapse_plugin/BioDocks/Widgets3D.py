"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
from .Tools import *
from .AnalysisIO import *
from .SettingsWidgets import *
import math
import pyqtgraph.opengl as gl
#from PyQt4 import QtCore, QtGui
from qtpy import QtCore, QtGui, QtWidgets
import numpy as np

class Plot3DWidget(gl.GLViewWidget):
	status = QtCore.Signal(str)
	copy = QtCore.Signal(object, str)
	plot = QtCore.Signal(object, str)
	def __init__(self, name='3D Plot Widget'):
		super(Plot3DWidget, self).__init__()
		self.__name__ = name
		self.addedMenu = QtWidgets.QMenu('Other')

	def clear(self):
		while self.items:
			self.removeItem(self.items[0])

	def _make_menu(self):
		menu = QtWidgets.QMenu(self.__name__)
		fileMenu = menu.addMenu('&File')
		fileMenu.addAction(QtWidgets.QAction('&Open', fileMenu, triggered=self.load_file))
		itemsMenu = menu.addMenu('&Items')
		for i in self.items:
			itemsMenu.addMenu(i.menu)
		menu.addAction(QtWidgets.QAction('&Clear Items', itemsMenu, triggered=self.clear))
		if not self.addedMenu.isEmpty():
			menu.addMenu(self.addedMenu)
		return menu

	def add_to_menu(self, item):
		if isinstance(item, QtWidgets.QMenu):
			self.addedMenu.addMenu(item)
		elif isinstance(item, QtWidgets.QAction):
			self.addedMenu.addAction(item)

	def contextMenuEvent(self, ev):
		self._make_menu().exec_(ev.globalPos())

	def translate(self, dX, dY, dZ):
		atX, atY, atZ = self.cameraPosition()
		self.pan(atX + dX, atY + dY, atZ + dZ)
		self.opts['distance'] = 500

	def moveTo(self, pos=(0, 0, 0), item=None, distance = 1000):
		atX, atY, atZ = self.cameraPosition()
		self.pan(-atX, -atY, -atZ)
		self.opts['distance'] = distance
		if item != None:
			pos = list(np.average(item.pos, 0))
		self.opts['center'] = QVector3D(*pos)


	def make_menu_for(self, item):
		item.menu = QtWidgets.QMenu(item.__name__)
		item.menu.addAction(QtWidgets.QAction('Goto', item.menu, triggered=lambda : self.moveTo(item=item)))
		item.menu.addAction(QtWidgets.QAction('Hide Item', item.menu, triggered=lambda f: item.setVisible(not f), checkable=True))
		item.menu.addAction(QtWidgets.QAction('&Copy Data', item.menu, triggered=lambda : self.copy.emit(item.pos, "%s copy" % item.__name__)))
		item.menu.addAction(QtWidgets.QAction('Properties', item.menu, triggered=lambda : self.editItem(item)))
		item.menu.addAction(QtWidgets.QAction('&Remove', item.menu, triggered=lambda : self.removeItem(item)))

	def add_item_from_array(self, array, name='',test=True):
		if test:    
			if isinstance(array, (list, tuple, np.ndarray)):
				if np.shape(array)[0] == 3:
					array = np.transpose(array)
				item = gl.GLScatterPlotItem(pos=array)
				item.__name__ = name
			elif type(array) == dict:
				cols = array.keys()
				self.op = ParameterWidget('Select the columns to use:', [{'key': 'Name', 'value': name}, \
					{'key': 'X Column', 'value': sort_closest(cols, 'X0')}, {'key':'Y Column', 'value': sort_closest(cols, 'Y1')}, \
					{'key':'Z Column', 'value': sort_closest(cols, 'Z2')}, {'key': 'Size', 'value': 4}, {'key': 'Color', 'value': QColor(0, 255, 0)}],\
					about='''Importing a file to a plot widget. Select the options below for how to read the file data''', doneButton=True)
				self.op.done.connect(lambda d: self.addItem(gl.GLScatterPlotItem(pos=np.transpose([array[d['X Column']], array[d['Y Column']], array[d['Z Column']]]), color=d['Color'].getRgbF(), size=d['Size']), name=d['Name']))
				self.op.show()
		else:
				if type(array) == dict:
					self.addItem(gl.GLScatterPlotItem(pos=np.transpose([array['Xc'], array['Yc'], array['Zc']]), color=QColor(0, 255, 0).getRgbF(), size=4))


	def addArray(self, array, color=QColor(0, 255, 0),size=4,name=''):
	#print(array[:,0])        
		array = np.transpose([array[:,0],array[:,1],array[:,2]])
		item = gl.GLScatterPlotItem(pos=array, color=color.getRgbF(), size=size)
		item.__name__ = name
		self.addItem(item)  
        
	#TODO
	def addLine(self,ch1_x,ch1_y,ch1_z,ch2_x,ch2_y,ch2_z,name=''):
		item = gl.GLLinePlotItem()        
		item.setVisible(True)         
		item.setData(pos=np.array([[ch1_x,ch1_y,ch1_z], [ch2_x,ch2_y,ch2_z]]), color=(1, 1, 1, 1))        
		item.__name__ = name 
        
		self.addItem(item) 
		return


	def addItem(self, item, name=''):
		#print('item added: ',item.__name__)        
		if name == '':
			name = 'Item %d' % len(self.items)

		if isinstance(item, (np.ndarray, list, tuple)):
			self.add_item_from_array(item, name)
			return

		if not hasattr(item, '__name__'):
			item.__name__ = name
		else:
			name = item.__name__

		if not hasattr(item, 'menu'):
			self.make_menu_for(item)

		try:
			self.moveTo(list(np.average(item.pos, 0)))
		except Exception as e:
			pass
			#print("Could not move to item pos: %s" % e)

		super(Plot3DWidget, self).addItem(item)


	def deleteItem(self,itemName):
		for item in self.items:
			#print(item.__name__)    
			if item.__name__ == itemName:          
				self.removeItem(item)  

	def load_file(self, f = ''):
		if type(f) != str or f == '':
			f=getFilename('Select a file to open', filter="All files (*.*);;Text Files (*.txt);;Excel Files (*.xls, *.xlsx);;Numpy files (*.np*)")
		if f == '':
			return
		self.status.emit('Opening %s in %s dock' % (f, self.__name__))
		data = fileToArray(f)
		item = self.add_item_from_array(data, name=f)

	def updateItem(self, item, change):
		i = self.items.index(item)
		if 'Name' in change:
			del change['Name']
		if 'color' in change and isinstance(change['color'], QColor):
			col = change['color']
			change['color'] = (col.redF(), col.greenF(), col.blueF(), col.alphaF())
		if isinstance(self.items[i], gl.GLScatterPlotItem):
			self.items[i].setData(**change)
		elif isinstance(self.items[i], gl.GLMeshItem):
			if 'color' in change:
				self.items[i].setColor(change['color'])
				del change['color']
			self.items[i].opts.update(change)
			self.items[i].meshDataChanged()

	def editItem(self, item):
		i = self.items.index(item)
		params = [	{'key': 'Name', 'value': item.__name__, 'readonly':True}]
		if isinstance(self.items[i], gl.GLScatterPlotItem):
			params.extend([{'key': 'color', 'name': 'Color', 'value': QColor(*np.multiply(255, self.items[i].color))}, \
				{'key': 'size', 'name': 'Size', 'value': self.items[i].size}])
		elif isinstance(self.items[i], gl.GLMeshItem):
			for k, v in self.items[i].opts.items():
				if k in ('edgeColor', 'color'):
					params.append({'key': k, 'value': QColor(*np.multiply(255, v)) if not isinstance(v, QColor) else v})
				elif k in ('drawEdges', 'drawFaces'):
					params.append({'key': k, 'value': v})
		elif isinstance(self.items[i], gl.GLLinePlotItem):
			pass
		self.op = ParameterWidget('Change the settings for 3DPlot Items', params, doneButton=True)
		self.op.valueChanged.connect(lambda s, v: self.updateItem(item, {str(s): v}))
		self.op.done.connect(lambda d: self.updateItem(item, d))
		self.op.show()
