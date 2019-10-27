"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""
from .Tools import *
from .ROIs import *
import flika
from flika import global_vars as g
from flika.window import Window

class DataWidget(pg.TableWidget):
	__name__ = "Data Widget"
	def __init__(self, sortable=False, **args):
		if 'name' in args:
			self.__name__ = args.pop('name')
		super(DataWidget, self).__init__(**args)
		self.setSortingEnabled(sortable)
		self.addedMenu = QtGui.QMenu('Other')

	def contextMenuEvent(self, ev):
		self._make_menu().exec_(ev.globalPos())

	def _make_menu(self):
		menu = QtGui.QMenu('%s Options' % self.__name__)
		self.contextMenu.setTitle('Edit')
		menu.addMenu(self.contextMenu)
		fileMenu = menu.addMenu("File")
		fileMenu.addAction(QtGui.QAction('Open file...', menu, triggered=lambda : self.load_file(append=False)))
		dataMenu = menu.addMenu("Data Options")
		dataMenu.addAction(QtGui.QAction('Transpose Data', menu, triggered=self.transpose))
		dataMenu.addAction(QtGui.QAction('Remove Selected Column(s)', menu, triggered=lambda : [self.removeColumn(i) for i in sorted({cell.column() for cell in self.selectedItems()})[::-1]]))
		dataMenu.addAction(QtGui.QAction('Remove Selected Row(s)', menu, triggered=lambda : [self.removeRow(i) for i in sorted({cell.row() for cell in self.selectedItems()})[::-1]]))
		dataMenu.addAction(QtGui.QAction('Format All Cells', menu, triggered = self.changeFormat))
		menu.addAction(QtGui.QAction("&Plot", menu, triggered=lambda : copy("%s Plot" % self.__name__, self.getData()) if not self.isEmpty() else None))
		menu.addAction(QtGui.QAction('Clear Table', menu, triggered=self.clear))
		if not self.addedMenu.isEmpty():
			menu.addMenu(self.addedMenu)
		return menu

	def isEmpty(self):
		return self.rowCount() == 0 and self.columnCount() == 0

	def getData(self):
		rs, cs = int(self.rowCount()), int(self.columnCount())
		data = np.zeros((rs, cs))
		for r in range(rs):
			for c in range(cs):
				if self.item(r, c) is not None:
					data[r, c] = self.item(r, c).value
		return np.reshape(data, (rs, cs))

	def changeFormat(self, f=''):
		if f == '':
			f = getString(self, "Formatting cells", 'How would you like to display cell values? (%#.#f/d)', initial=self.items[0]._defaultFormat)
		for item in self.items:
			item.setFormat(f)

	def transpose(self):
		cop = self.getData().transpose()
		hs = [self.horizontalHeaderItem(i).text() for i in range(self.columnCount()) if self.horizontalHeaderItem(i)]
		vs = [self.verticalHeaderItem(i).text() for i in range(self.rowCount()) if self.verticalHeaderItem(i)]
		self.setHorizontalHeaderLabels(vs)
		self.setVerticalHeaderLabels(hs)
		self.setData(cop)

	def load_file(self, append=False):
		fname = getFilename('Select the type of file to open', filter = "All files (*.*);;TXT Files (*.txt);;Excel Files (*.xls, *.xlsx)")
		if fname == '':
			return
		arr = fileToArray(fname)
		self.setData(arr)

class TraceWidget(pg.PlotWidget):
	__name__ = "Trace Widget"
	def __init__(self, **args):
		if 'name' in args:
			self.__name__ = args.pop('name')
		super(TraceWidget, self).__init__(**args)
		self.trace = pg.PlotDataItem()
		self.addItem(self.trace)
		self._make_menu()

	def valAt(self, x_pos):
		x, y = self.trace.getData()
		try:
			if min(x) <= x_pos <= max(x):
				return y[x_pos]
			else:
				return None
		except Exception as e:
			pass

	def getTrace(self):
		return self.trace.getData()[1]

	def mouseMoveEvent(self, ev):
		p = self.mapToView(QtCore.QPointF(ev.pos()))
		self.setTitle('(%.3f, %.3f)' % (p.x(), p.y()))
		super(TraceWidget, self).mouseMoveEvent(ev)

	def setTrace(self, **args):
		'''accepts arguments of setData or a plotdataitem
		'''
		if 'item' in args:
			self.removeItem(self.trace)
			self.trace = args['item']
			self.addItem(self.trace)
		else:
			self.trace.setData(**args)

	def importTrace(self, fname):
		heads = [i.strip() for i in get_headers(fname)]
		head = getOption(self, 'Column select', label='Which column would you like to plot?', options=sort_closest(heads, 'Yc'))
		y = np.loadtxt(fname, skiprows = 1, usecols=[heads.index(head)])
		self.setTrace(y=y)

	def exportTrace(self):
		fname = getSaveFilename(title='Select the file to export the trace to', filter='Text Files (*.txt)')
		np.savetxt(fname, self.trace.getData()[1], header='Yc', comments='')

	def _make_menu(self):
		fileMenu = self.getViewBox().menu.addMenu('&File')
		fileMenu.addAction(QtGui.QAction('&Import Trace', fileMenu, triggered=lambda : self.importTrace(getFilename(title='Select the file to import as a trace'))))
		fileMenu.addAction(QtGui.QAction('&Export Trace', fileMenu, triggered=self.exportTrace))


class PlotWidget(pg.PlotWidget):
	'''plotwidet with menu additions to save and remove items'''
	__name__ = "Plot Widget"
	def __init__(self, **args):
		if 'name' in args:
			self.__name__ = args.pop('name')
		super(PlotWidget, self).__init__(**args)
		self.menu = self.getViewBox().menu
		self.itemsMenu = self.menu.addMenu('&Items')

	def addPlotDataItem(self, item, name, removable=True, *args, **kargs):
		item.__name__ = name
		if hasattr(item, "menu"):
			menu = item.menu
		else:
			item.menu = QtGui.QMenu(name)
		self.itemsMenu.addMenu(item.menu)
		if removable:
			item.menu.addAction(QtGui.QAction("Remove", item.menu, triggered=lambda : self.removeItem(item)))
		super(PlotWidget, self).addItem(item, *args, **kargs)

	def removeItem(self, item):
		for act in self.itemsMenu.actions():
			if act.text() == item.__name__:
				self.itemsMenu.removeAction(act)
		super(PlotWidget, self).removeItem(item)

class VideoWidget(pg.ImageView):
	__name__ = "Video Widget"
	imageChanged = Signal()
	def __init__(self, **args):
		if 'name' in args:
			self.__name__ = args.pop('name')
		super(VideoWidget, self).__init__(**args)
		self.image_item = self.getImageItem()
		self.roi.show = lambda : None
		self.normRoi.show = lambda : None
		try:
			self.ui.roiBtn.hide()
			self.ui.menuBtn.hide()
		except:
			self.ui.normBtn.hide()

		self._make_menu()

	def addItem(self, item, name, removable=True, **args):
		if isinstance(item, (list, tuple, np.ndarray)) and len(np.shape(item)) >= 2 and all([i > 3 for i in np.shape(item)]):
			self.setImage(item)
			return
		elif isinstance(item, (pg.ImageItem, pg.ImageView)):
			self.set(item.image)
			self.dock.rename(name)
			return
		elif isinstance(item, (list, tuple, np.ndarray)) and len(np.shape(item)) in np.shape(item):
			item = pg.PlotDataItem(*item)
			item.__name__ = name
			pg.ImageView.addItem(self, item, **args)
		else:
			pg.ImageView.addItem(self, item, **args)
			item.__name__ = name

		if not hasattr(item, 'menu'):
			menu = QtGui.QAction(name)
		else:
			menu = item.menu
		menu.addAction(QtGui.QAction(name, menu))
		return item

	def _make_menu(self):
		imageMenu = self.view.menu.addMenu("Image")
		imageMenu.addAction(QtGui.QAction("&Open Image", imageMenu, triggered = lambda : self.open_image()))
		imageMenu.addAction(QtGui.QAction("&Transpose", imageMenu, triggered = lambda : self.setImage(np.swapaxes(self.getProcessedImage(), *range(self.getProcessedImage().ndim)[-2:]))))
		imageMenu.addAction(QtGui.QAction('&Save Image', imageMenu, triggered = lambda : self.save_image()))
		genMenu = imageMenu.addMenu("Copy Special")
		genMenu.addAction(QtGui.QAction('&Average Intensity Frame', genMenu, triggered=lambda : \
			copy("%s Average Frame" % self.name, np.average(self.getProcessedImage(), 0))))
		genMenu.addAction(QtGui.QAction('&Average Intensity Plot', genMenu, triggered=lambda : \
			copy("%s Average Trace" % self.name, np.average(self.getProcessedImage(), (2, 1)))))
		genMenu.addAction(QtGui.QAction('&Current Frame', genMenu, triggered=lambda : \
			copy("%s Frame %s" % (self.name, self.currentIndex), self.getProcessedImage()[self.currentIndex])))
		self.itemsMenu = self.view.menu.addMenu("Items...")

	def contextMenuEvent(self, ev):
		self.itemsMenu.clear()
		for item in self.view.addedItems:
			if item.menu != None:
				self.itemsMenu.addMenu(item.menu)
		self.itemsMenu.addAction(QtGui.QAction('&Clear Items', self.itemsMenu, triggered=self.clear_items))
		ev.accept()

	def clear_items(self):
		for item in self.view.addedItems[::-1]:
			if not isinstance(item, pg.ImageItem):
				self.removeItem(item)

	def save_image(self):
		self.saver = TiffExporter(self.getProcessedImage(), self.name)
		self.saver.start()

	def setImage(self, im, signal=True):
		if np.ndim(im) == 2:
			im = np.array([im])
		super(VideoWidget, self).setImage(im)
		if signal:
			self.imageChanged.emit()

	def loaded(self):
		try:
			if self.getProcessedImage().size == 0:
				raise Exception('')
			return True
		except:
			return False

	def open_image(self, filename=''):
		self.opener = ImageImporter(filename)
		self.__name__ = os.path.basename(self.opener.filename)
		def done(image):
			self.setImage(image)
			del image
			del self.opener
		self.opener.done.connect(done)
		if self.opener.filename != "":
			self.opener.start()

class ROIViewBox(pg.ViewBox):
	roiCreated = Signal(object)
	def __init__(self, aspectRatio=False, creatingROI=False, roiMenu=True, roiSnap=1, **kargs):
		super(ROIViewBox, self).__init__(lockAspect = aspectRatio, **kargs)
		self.mouse_connected = False
		self.creatingROI = creatingROI
		self.roiSnap = roiSnap
		if roiMenu:
			self._make_menu()

	def sceneEvent(self, ev):
		if not self.mouse_connected:
			self.mouse_connected = True
			self.scene().sigMouseMoved.connect(self.mouseMoved)
		return super(ROIViewBox, self).sceneEvent(ev)

	def mouseMoved(self, pos):
		pos = self.mapToView(pos)
		self.mouse_x = pos.x()
		self.mouse_y = pos.y()
		g.m.statusBar().showMessage('x={}, y={}'.format(int(self.mouse_x),int(self.mouse_y)))

	def clear_rois(self):
		for roi in self.addedItems[::-1]:
			if isinstance(roi, Freehand):
				roi.delete()

	def _make_menu(self):
		roiMenu = self.menu.addMenu("ROI Options")
		drawing = QtGui.QAction("&Enable ROI Drawing", self.menu, triggered = lambda : setattr(self, 'creatingROI', not self.creatingROI), checkable=True)
		roiMenu.addAction(drawing)
		drawing.setChecked(self.creatingROI)
		roiMenu.addAction(QtGui.QAction('Export ROIs', roiMenu, triggered=self.export_rois))
		roiMenu.addAction(QtGui.QAction('Import ROIs', roiMenu, triggered=self.import_rois))
		roiMenu.addAction(QtGui.QAction('Clear ROIs', roiMenu, triggered=self.clear_rois))

	def export_rois(self):
		fname = getSaveFilename(title='Saving ROIs', filter='Text files (*.txt)')
		with open(fname, 'w') as outfile:
			outfile.write(''.join([repr(roi) for roi in self.addedItems if isinstance(roi, Freehand)]))

	def import_rois(self):
		filename = getFilename('Select the roi text file to import from...', filter='Text Files (*.txt)')
		for roi in Freehand.import_rois(filename):
			self.addItem(roi)
			self.roiCreated.emit(roi)

	def mouseDragEvent(self, ev):
		if ev.button() == 1 and self.creatingROI:
			ev.accept()
			pt = self.mapToView(ev.pos())
			if ev.isStart():
				self.disableAutoRange()
				self.currentROI = Freehand(pt, snap = self.roiSnap)
				self.addItem(self.currentROI)
			elif ev.isFinish():
				if self.currentROI.drawFinished():
					self.roiCreated.emit(self.currentROI)
				self.currentROI = None
			else:
				self.currentROI.extend(pt)
		else:
			ev.accept()
			difference = self.mapToView(ev.lastScenePos().toQPoint()) - self.mapToView(ev.scenePos().toQPoint())
			self.translateBy(difference)
