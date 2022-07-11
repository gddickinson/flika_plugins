import time
tic=time.time()
import os, sys
import numpy as np
from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtCore import Signal
from qtpy.QtWidgets import *
from pyqtgraph import plot, show
import pyqtgraph as pg
import flika.global_vars as g
from collections import OrderedDict
import traceback

class RectSelector(pg.ROI):
	def __init__(self, origin, size):
		pg.ROI.__init__(self, origin, size, translateSnap=True, scaleSnap=True)
		self.setPen(QPen(QColor(255, 0, 0), 0))
		## handles scaling horizontally around center
		leftHandle = self.addScaleHandle([1, 0.5], [0, 0.5])
		rightHandle = self.addScaleHandle([0, 0.5], [1, 0.5])
		
		## handles scaling vertically from opposite edge
		bottomHandle = self.addScaleHandle([0.5, 0], [0.5, 1])
		topHandle = self.addScaleHandle([0.5, 1], [0.5, 0])

		## handles scaling both vertically and horizontally
		self.addScaleHandle([1, 1], [0, 0])
		self.addScaleHandle([0, 0], [1, 1])
		self.sigRegionChanged.connect(self.onTranslate)
		
		self.addPolyfill()

		self.traceLine = None


	def setVisible(self, v):
		pg.ROI.setVisible(self, v)
		if g.m.currentTrace:
			self.traceLine = g.m.currentTrace.rois[0]['p1trace']

	def onTranslate(self):
		if not self.traceLine:
			return
		x, y = self.pos()
		if x < 0:
			self.setPos([0, y])
			return
		w, h = self.size()
		frames = len(self.traceLine.getData()[1])
		if x + w >= frames:
			if frames - x - 1 <= 0:
				return
			self.setSize([frames - x - 1, h])
			return
		self.redraw()

	def redraw(self):
		baseline = self.pos()[1]
		x, y = self.getFrameTrace()
		if x is None:
			return
		y -= baseline
		ftrace = get_polyfit(x, y)
		self.analyze_trace(x, y, ftrace)
		
		self.polyDataItem.setData(x=x, y=ftrace, pen=self.polyPen, fillBrush=QColor(0, 100, 155, 100), fillLevel=0)

	def addPolyfill(self):
		self.polyPen = QPen(QColor(255, 0, 0), 0)
		self.polyPen.setStyle(Qt.DashLine)
		self.polyPen.setDashOffset(5)
		self.polyDataItem = pg.PlotDataItem(pen=self.polyPen)
		self.polyDataItem.setParentItem(self)

	def getFrameRect(self):
		origin = self.pos()
		origin[0] = max(0, origin[0])
		origin[1] = max(0, origin[1])
		size = self.size()
		if self.traceLine and origin[0] + size[0] > max(self.traceLine.getData()[0]):
			size[1] = max(self.traceLine.getData()[0]) - origin[0]
		return np.array([origin[0], origin[1], origin[0] + size[0], origin[1] + size[1]], int)

	def getFrameTrace(self):
		if not self.traceLine:
			return None, None
		t = np.copy(self.traceLine.getData()[1])
		x1, y1, x2, y2 = self.getFrameRect()
		x1 = max(0, round(x1))
		x2 = min(round(x2), len(t))
		t = t[x1:x2 + 1]
		return (np.arange(0, x2+1 - x1), t)

	def setTrace(self, t):
		if t == None:
			return
		if self.parentWidget() != None:
			self.parentWidget().removeItem(self)
		t.parentWidget().addItem(self)
		
		self.traceLine = t
		self.onTranslate()

	def getIntegral(self):
		x1, y1, x2, y2 = self.getFrameRect()
		y = self.getTrace()[x1:x2+1]
		return np.trapz(y)

	def baseline(self):
		return self.pos()[1]

	def analyze_trace(self, x, y, ftrace):
		pos = self.pos()
		size = self.size()
		x_peak = np.argmax(y)
		f_peak = np.argmax(ftrace)
		self.data = OrderedDict([('Baseline', (pos[0], pos[1])), \
							('Peak', (f_peak + pos[0], ftrace[f_peak])),\
							('Delta Peak', (f_peak, ftrace[f_peak] - pos[1]))])
		#yRiseFall = getRiseFall(x, y)
		ftraceRiseFall = getRiseFall(x, ftrace)
		self.data.update(OrderedDict([(k, ftraceRiseFall[k]) for k in ftraceRiseFall.keys()]))
		self.data['area'] = (0, np.trapz(y), 0, np.trapz(ftrace))

def get_polyfit(x, y):
	np.warnings.simplefilter('ignore', np.RankWarning)
	poly=np.poly1d(np.polyfit(x, y, 20))
	ftrace=poly(x)
	return ftrace
	
def getRiseFall(x, y):
	x_peak = np.where(y == max(y))[0][0]
	baseline = y[0]
	dPeak = (x_peak, y[x_peak]-baseline)

	data = OrderedDict([('Rise 20%', [-1, -1]),
		('Rise 50%', [-1, -1]), ('Rise 80%', [-1, -1]),
		('Rise 100%', [-1, -1]), ('Fall 80%', [-1, -1]),
		('Fall 50%', [-1, -1]), ('Fall 20%', [-1, -1])])
	try:
		thresh20=dPeak[1]*.2 + baseline
		thresh50=dPeak[1]*.5 + baseline
		thresh80=dPeak[1]*.8 + baseline

		data['Rise 20%'] = [np.argwhere(y>thresh20)[0][0], thresh20]
		data['Rise 50%'] = [np.argwhere(y>thresh50)[0][0], thresh50]
		data['Rise 80%'] = [np.argwhere(y>thresh80)[0][0], thresh80]
		data['Rise 100%'] = [np.argmax(y), max(y)]

		tmp=np.squeeze(np.argwhere(y<thresh80))
		data['Fall 80%'] = [tmp[tmp>data['Rise 100%'][0]][0], thresh80]
		tmp=np.squeeze(np.argwhere(y<thresh50))
		data['Fall 50%'] = [tmp[tmp>data['Fall 80%'][0]][0], thresh50]
		tmp=np.squeeze(np.argwhere(y<thresh20))
		data['Fall 20%'] = [tmp[tmp>data['Fall 50%'][0]][0], thresh20]
	except Exception as e:
		pass
		#print("Analysis Failed: %s" % traceback.format_exc())
	return data