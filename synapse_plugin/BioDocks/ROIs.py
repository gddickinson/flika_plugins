"""
@author: Brett Settle
@Department: UCI Neurobiology and Behavioral Science
@Lab: Parker Lab
@Date: August 6, 2015
"""

from .Tools import *
import pyqtgraph as pg
#from PyQt4.QtCore import pyqtSignal as Signal
from qtpy.QtCore import Signal
from qtpy.QtWidgets import *
from skimage.draw import polygon

class PixelSelector(pg.RectROI):
	radius = 1
	p = (-radius, -radius)
	def __init__(self, pos=p, max_bounds=[1000, 1000], **kargs):
		maxBounds = QRectF(-self.radius, -self.radius, max_bounds[0], max_bounds[1])
		super(PixelSelector, self).__init__(pos=pos, size=(2 * self.radius + 1, 2 * self.radius + 1), \
						pen=QPen(Qt.red), translateSnap=True, maxBounds=maxBounds, **kargs)

	def setPos(self, pos, **kargs):
		pos = (pos.x() - self.radius, pos.y() - self.radius)
		super(PixelSelector, self).setPos(pos, **kargs)
		self.sigRegionChanged.emit(self)

	def getPosition(self):
		x, y = self.pos()
		return np.array([x + self.radius, y + self.radius])


class ROIRange(pg.LinearRegionItem):
	sigRemoved = Signal(object)
	def __init__(self, bounds = [0, 10]):
		super(ROIRange, self).__init__(bounds)
		self.id = 0
		self.__name__ = "ROI Range %d" % self.id
		self.lines[0].setPos = lambda pos : pg.InfiniteLine.setPos(self.lines[0], int(pos if isinstance(pos, (int, float)) else pos.x()))
		self.lines[1].setPos = lambda pos : pg.InfiniteLine.setPos(self.lines[1], int(pos if isinstance(pos, (int, float)) else pos.x()))
		self.setZValue(20)
		self._make_menu()

	def parentChanged(self):
		if self.parentWidget() != None:
			self.id = 1
			while any([roi.id == self.id for roi in self.getViewBox().items() if isinstance(roi, ROIRange) and roi != self]):
				self.id += 1
			self.__name__ = "ROI Range %d" % self.id
			self.menu.setTitle(self.__name__)
		super(ROIRange, self).parentChanged()

	def _make_menu(self):
		self.menu = QMenu(self.__name__)
		setMenu = self.menu.addMenu("Set Range to...")
		setMenu.addAction(QAction("&Average Value", self.menu, triggered=lambda : self.setTrace(np.average(self.getRegionTrace()[1]))))
		setMenu.addAction(QAction("&Enter Value", self.menu, triggered=lambda : self.setTrace(getFloat(self, "Setting Range Value", "What value would you like to set the region to?"))))
		self.menu.addAction(QAction("&Make Baseline", self.menu, triggered=lambda : self.setTrace(self.getTrace() / np.average(self.getRegionTrace()[1]), portion=False)))
		self.menu.addAction(QAction("&Remove", self.menu, triggered=self.delete))

	def delete(self):
		self.parentItem().getViewBox().removeItem(self)
		self.sigRemoved.emit(self)

	def getRegionTrace(self):
		t = self.getTrace()
		x1, x2 = self.getRegion()
		x1 = int(x1)
		x1 = max(0, x1)
		x2 = int(x2)
		x2 = min(x2, len(t))
		return (np.arange(x1, x2+1), t[x1:x2 + 1])

	def getTrace(self):
		trace = [line for line in self.parentItem().getViewBox().addedItems if isinstance(line, pg.PlotDataItem)][0]
		return np.copy(trace.getData()[1])

	def setTrace(self, val, portion=True):
		trace = [line for line in self.parentItem().getViewBox().addedItems if isinstance(line, pg.PlotDataItem)][0]
		x1, x2 = self.getRegion()
		if portion:
			t = trace.getData()[1]
			t[x1:x2+1] = val
		else:
			t = val
		trace.setData(y=t)

	def getIntegral(self):
		x1, x2 = self.getRegion()
		y = self.getTrace()[x1:x2+1]
		return np.trapz(y)

class Freehand(pg.GraphicsObject):
	sigHoverChanged = Signal(object, bool)
	sigClicked = Signal(object, object)
	sigRemoved = Signal(object)
	sigChanged = Signal(object)

	def __init__(self, pos, snap = 1, color=None):
		super(Freehand, self).__init__()
		self.setAcceptedMouseButtons(Qt.NoButton)
		self.path_item = QGraphicsPathItem(parent=self)
		self.boundingRect = self.path_item.boundingRect
		self.paint = self.path_item.paint
		self.snap = snap
		pos = self.snapPoint(pos)
		self.path = QPainterPath(pos)
		if not color:
			self.color = random_color()
		else:
			self.color = color
		self.pen = QPen(self.color)
		self.path_item.setPath(self.path)
		self.path_item.setPen(self.pen)

		self.id = -1
		self.hover = False
		self.isMoving = False
		self.colorDialog = QColorDialog()
		self.colorDialog.colorSelected.connect(self.colorSelected)

	@staticmethod
	def import_rois(filename):
		kind = None
		pts = []
		for line in open(filename, 'r'):
			if kind == None:
				kind = line
			elif line.strip() == '':
				cur_roi = Freehand(QPointF(*pts[0]))
				for p in pts[1:]:
					cur_roi.extend(QPointF(*p))
				cur_roi.drawFinished()
				yield cur_roi
				kind = None
				pts = []
			else:
				pts.append([float(i) for i in line.split()])

	def __repr__(self):
		return 'freehand\n' + '\n'.join(['%.2f\t%.2f' % (x, y) for x,y in self.getPoints()]) + '\n\n'

	def parentChanged(self):
		self.id = 1
		while any([roi.id == self.id for roi in self.getViewBox().items() if isinstance(roi, Freehand) and roi != self]):
			self.id += 1
		if hasattr(self, 'ti'):
			self.ti.setText("%d" % self.id)
		self.__name__ = "ROI %d" % self.id
		self._make_menu()
		#super(Freehand, self).parentChanged()

	def translate(self, difference):
		self.path.translate(difference.x(), difference.y())
		self.path_item.setPath(self.path)
		self.ti.setPos(*self.getPoints()[0])
		self.update()

	def snapPoint(self, pos):
		if isinstance(pos, (tuple, list)):
			pos = QPointF(*pos)
		if self.snap != False:
			pos = QPointF(pos.x() - (pos.x() % self.snap), pos.y() - (pos.y() % self.snap))
		return pos

	def extend(self, pos):
		pos = self.snapPoint(pos)
		self.path.lineTo(pos)
		self.path_item.setPath(self.path)

	def delete(self):
		self.parentItem().getViewBox().removeItem(self)
		self.sigRemoved.emit(self)

	def getPoints(self):
		points=[]
		for i in np.arange(self.path.elementCount()):
			e=self.path.elementAt(i)
			x=e.x
			y=e.y
			if len(points)==0 or points[-1]!=(x,y):
				points.append((x,y))
		return np.copy(points)

	def drawFinished(self):
		self.path.closeSubpath()
		self.path_item.setPath(self.path)
		self.ti = pg.TextItem(text='%d' % self.id, fill=self.color, color=(0, 0, 0))
		self.ti.setParentItem(self)
		self.ti.setPos(*self.getPoints()[0])
		self.sigChanged.emit(self)
		try:
			if getArea(self.getPoints()) < .2:
				raise Exception('ROI is too small, must have minimum area of .2')
		except:
			self.delete()
			return False
		return True

	def contextMenuEvent(self, ev):
		if isinstance(ev, QGraphicsSceneContextMenuEvent):
			ev.accept()
			pos = ev.screenPos()
			self.menu.popup(pos)

	def hoverEvent(self, ev):
		hover = False
		if not ev.isExit():
			if ev.acceptDrags(Qt.LeftButton):
				hover=True

			for btn in [Qt.LeftButton, Qt.RightButton, Qt.MidButton]:
				if int(self.acceptedMouseButtons() & btn) > 0 and ev.acceptClicks(btn):
					hover=True
			ev.acceptClicks(Qt.RightButton)

		if hover:
			self.setMouseHover(True)
			ev.acceptClicks(Qt.LeftButton)  ## If the ROI is hilighted, we should accept all clicks to avoid confusion.
			ev.acceptClicks(Qt.RightButton)
			ev.acceptClicks(Qt.MidButton)
		else:
			self.setMouseHover(False)

	def setMouseHover(self, hover):
		## Inform the ROI that the mouse is (not) hovering over it
		if self.hover == hover:
			return
		self.sigHoverChanged.emit(self, hover)
		self.hover = hover
		if hover:
			self.pen.setColor(QColor(255, 255, 0))
		else:
			self.pen.setColor(self.color)
		self.path_item.setPen(self.pen)
		self.update()

	def colorSelected(self, color):
		if color.isValid():
			self.color=QColor(color.name())
			self.pen.setColor(self.color)
			self.path_item.setPen(self.pen)
			self.ti.fill = pg.mkBrush(self.color)
		self.update()

	def _make_menu(self):
		self.menu = QMenu("ROI %d" % self.id)
		self.menu.addAction(QAction("Change C&olor", self.menu, triggered =lambda : self.colorDialog.open()))
		self.menu.addAction(QAction("&Delete", self.menu, triggered = self.delete))

	def contains(self, pt):
		return self.path.contains(pt)

	def mouseDragEvent(self, ev):
		if ev.isStart():
			if ev.button() == Qt.LeftButton:
				self.setSelected(True)
				self.isMoving = True
				ev.accept()

		elif ev.isFinish():
			if self.isMoving:
				self.sigChanged.emit(self)
			self.isMoving = False
			return

		if self.isMoving and ev.buttons() == Qt.LeftButton:
			newPos = self.mapToParent(ev.pos()) - self.mapToParent(ev.lastPos())
			self.translate(newPos)

	def mouseClickEvent(self, ev):
		if ev.button() == Qt.RightButton and self.isMoving:
			ev.accept()
		if ev.button() == Qt.RightButton:
			self.contextMenuEvent(ev)
			ev.accept()
		elif ev.button() == Qt.LeftButton:
			self.sigClicked.emit(self, ev)
		elif int(ev.button() & self.acceptedMouseButtons()) > 0:
			ev.accept()
		else:
			ev.ignore()

	def getMaskOnImage(self, tif):
	    pts = self.getPoints()
	    x=np.array([p[0] for p in pts])
	    y=np.array([p[1] for p in pts])
	    nDims=len(tif.shape)
	    if nDims==4: #if this is an RGB image stack
	        tif=np.mean(tif,3)
	        mask=np.zeros(tif[0,:,:].shape,np.bool)
	    elif nDims==3:
	        mask=np.zeros(tif[0,:,:].shape,np.bool)
	    if nDims==2: #if this is a static image
	        mask=np.zeros(tif.shape,np.bool)

	    xx,yy=polygon(x,y,shape=mask.shape)
	    mask[xx,yy]=True
	    return mask
