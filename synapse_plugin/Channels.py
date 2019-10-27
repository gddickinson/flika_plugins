import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
from qtpy.QtGui import *
from qtpy.QtWidgets import *

class ActivePoint():
	def __init__(self, data):
		self.pos = np.array([data['Xc'], data['Yc'], data['Zc']])
		self.data = data

	def __getitem__(self, item):
		if type(item) == int:
			return self.pos[item]
		if item in self.data:
			return self.data[item]
		else:
			return self.__dict__[item]

class Synapse(gl.GLScatterPlotItem):
	def __init__(self, channelA, channelB):
		super(Synapse, self).__init__()
		self.centers = gl.GLLinePlotItem()
		self.centers.setParentItem(self)
		self.centers.setVisible(False)
		self.mesh = gl.GLMeshItem()
		self.mesh.setParentItem(self)
		self.mesh.setVisible(False)
		self.setChannels(channelA, channelB)
		self._make_menu()

	def setChannels(self, channelA, channelB):
		self.channels = {ch.__name__: ch for ch in [channelA, channelB]}
		colors = [QColor.getRgbF(channelA.color()) for i in range(channelA.getCount())]
		colors.extend([QColor.getRgbF(channelB.color()) for i in range(channelB.getCount())])
		colors = np.array(colors)
		pos = list(channelA.getPoints(z=True))
		pos.extend(channelB.getPoints(z=True))
		self.setData(pos=np.array(pos), color=colors, pxMode=True, size=4)
		if channelA.getCount() > 0 and channelB.getCount() > 0:
			self.centers.setData(pos=np.array([channelA.getCenter(z=True), channelB.getCenter(z=True)]), color=(1, 1, 1, 1))

	def _make_menu(self):
		self.menu = QMenu('Synapse 3D Plot Options')
		self.centerShow = QAction('Show Center Distance', self.menu, triggered=lambda : self.centers.setVisible(self.centerShow.isChecked()), checkable=True)
		self.menu.addAction(self.centerShow)
		self.menu.addAction(QAction('Export Coordinates', self.menu, triggered=lambda : export_arr(np.transpose(self.pos), header='X\tY\tZ', comments='')))

class Channel(pg.ScatterPlotItem):
	def __init__(self, name, points, color):
		super(Channel, self).__init__(x=[p[0] for p in points], y=[p[1] for p in points], brush=color, pen=color, size=2)
		self.__name__ = name
		self.pts = points

	def getPoints(self, z=False):
		if self.getCount() == 0:
			return np.array([])
		if z:
			return np.array([p.pos for p in self.pts])
		else:
			return np.transpose(self.getData())

	def getCount(self):
		return len(self.pts)

	def getCenter(self, z=False):
		return np.average(self.getPoints(z), 0)

	def color(self):
		return self.opts['brush'].color()
