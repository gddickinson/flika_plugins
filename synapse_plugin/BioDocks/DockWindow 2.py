"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
import os, sys
from pyqtgraph.Qt import QtCore, QtGui
from pyqtgraph import GraphicsWidget, setConfigOptions
from pyqtgraph.console import ConsoleWidget
from pyqtgraph.dockarea import *

from .Tools import *
from .PlotWidgets import *
setConfigOptions(useWeave=False)

if '-no3D' not in sys.argv:
	from .Widgets3D import Plot3DWidget

class WidgetDock(Dock):
	def __init__(self, name, widget, size=(500, 500)):
		'''create the base structure of a dock to hold widgets'''
		super(WidgetDock, self).__init__(name, size=size, closable=True)
		self.dockMenu = QtGui.QMenu()
		self.label.sigClicked.connect(self._dockClick)
		self.label.startedDrag = False				# mandatory to handle label
		self.label.pressPos = QtCore.QPoint(0, 0)	# click events
		self._make_base_actions()
		self.addWidget(widget)

	def _dockClick(self, label, event):
		if event.button() == QtCore.Qt.RightButton:
			event.accept()
			self.dockMenu.exec_(event.globalPos())

	def _make_base_actions(self):
		self.dockMenu.addAction(QtGui.QAction("Re&name Dock", self, triggered=lambda : self._rename()))
		self.dockMenu.addAction(QtGui.QAction("&Float Dock", self, triggered=self.float))
		self.dockMenu.addAction(QtGui.QAction("&Paste to", self, triggered=self.paste))

	def paste(self):
		data = QtGui.qApp.clipboard().text()
		try:
			i = data.index('::')
			name = data[:i]
			data = data[i + 2:]
		except:
			name = 'Pasted Data'
		arr = np.array([np.fromstring(row, sep='\t') for row in data.split('\n')])
		if isinstance(self.widget, (pg.ImageWidget, pg.PlotWidget)):
			self.widgets[0].addItem(pg.PlotDataItem(arr, name=name))
		elif isinstance(self.widget, pg.TableWidget):
			self.widgets[0].setData(arr)

	def _rename(self, name=''):
		if name == '':
			name = getString(self, title="Rename Dock...", label='What would you like to rename %s to?' % self.name(), initial=self.name())
		if name != '':
			self.area.docks[name] = self.area.docks.pop(self.name())
			self.label.setText(name)
			self.widgets[0].__name__ = name

	def close(self):
		del self.area.docks[self.name()]
		Dock.close(self)

class DockWindow(QtGui.QMainWindow):
	sigProgressChanged = QtCore.pyqtSignal(str)
	def __init__(self, addMenu=True):
		super(DockWindow, self).__init__()
		self.dockarea = DockArea()
		self.setCentralWidget(self.dockarea)
		self.clipboard = QtGui.QApplication.clipboard()
		if addMenu:
			addMenu = self.menuBar().addMenu("&Add Dock")
			addMenu.addAction(QtGui.QAction("Trace Widget", addMenu, triggered=lambda : self.addWidget(TraceWidget())))
			addMenu.addAction(QtGui.QAction("Plot Widget", addMenu, triggered=lambda : self.addWidget(PlotWidget())))
			addMenu.addAction(QtGui.QAction("Video Widget", addMenu, triggered=lambda : self.addWidget(VideoWidget())))
			addMenu.addAction(QtGui.QAction("Data Widget", addMenu, triggered=lambda : self.addWidget(DataWidget())))
			addMenu.addAction(QtGui.QAction("Console Widget", addMenu, triggered=lambda : self.addWidget(ConsoleWidget(namespace=globals()))))
			if '-no3D' not in sys.argv:
				addMenu.addAction(QtGui.QAction("3D Plot Widget", addMenu, triggered=lambda : self.addWidget(Plot3DWidget())))

	def __getattr__(self, at):
		if at not in dir(self):
			return DockArea.__getattribute__(self.dockarea, at)
		else:
			return QtGui.QMainWindow.__getattr__(self, at)

	def showProgress(self, s):
		t = self.windowTitle()
		try:
			i = t.index(' -- ')
			t = t[:i]
		except:
			pass
		t += ' -- [%s]' % s
		self.sigProgressChanged.emit(s)
		self.setWindowTitle(t)

	def addWidget(self, widget, where=[], **args):
		if not hasattr(widget, '__name__'):
			s = str(widget.__class__)
			a = s.index("'")
			b = s.index("'", a + 1)
			widget.__name__ = s[a+1: b]
		while widget.__name__ in self.dockarea.docks.keys():
			widget.__name__ = "%s - copy" % widget.__name__
		d = WidgetDock(widget.__name__, widget=widget, **args)
		self.addDock(d, *where)
		return d

if __name__ == "__main__":
	app = QtGui.QApplication([])
	dw = DockWindow()
	dw.show()
	app.exec_()
