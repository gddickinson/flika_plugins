from qtpy import uic
import scipy
import flika.global_vars as g
import pyqtgraph as pg
from .GlobalPolyfit import *
from flika.window import Window
from flika.roi import *
from flika.process.measure import measure
from flika.process.file_ import save_file_gui
from flika.tracefig import TraceFig

class AnalysisUI(QtWidgets.QGroupBox):
	ui = None
	log_data = ''
	def __init__(self):
		QtWidgets.QGroupBox.__init__(self)
		uic.loadUi(os.path.join(os.path.dirname(__file__), 'main.ui'), self)
		
		self.traceRectROI = RectSelector([0, 0], [10, 10])
		self.traceRectROI.setVisible(False)

		self.traceROICheck.toggled.connect(self.toggleVisible)
		self.measureButton.clicked.connect(measure.gui)
		self.tableWidget.setFormat("%.3f")
		self.tableWidget.setSortingEnabled(False)
		self.traceComboBox.mousePressEvent = self.comboBoxClicked
		self.logButton.clicked.connect(self.logData)
		
		self.nextButton.clicked.connect(self.nextROI)
		self.prevButton.clicked.connect(self.previousROI)
		self.nextPuffButton.clicked.connect(self.nextPuff)
		self.prevPuffButton.clicked.connect(self.previousPuff)
		
		def saveLogData():
			f = save_file_gui("Save Logged Ploynomial Fit Data", filetypes="*.txt")
			if f:
				self.saveLoggedData(f)

		self.traceRectROI.sigRegionChanged.connect(self.fillDataTable)
		self.saveButton.clicked.connect(saveLogData)
		self.traceComboBox.updating = False
		self.all_rois = []
		self.traceComboBox.activated.connect(self.indexChanged)
		self.puffComboBox.activated.connect(self.puffSelected)
		self.traceComboBox.currentIndexChanged.connect(self.indexChanged)
		self.puffComboBox.currentIndexChanged.connect(self.puffSelected)

		g.m.dialogs.append(self)
		
	def saveLoggedData(self, fname):
		with open(fname, 'w') as outf:
			outf.write("Value\tFtrace Frame\tFtrace Y\n")
			outf.write(AnalysisUI.log_data)
		AnalysisUI.log_data = ''

	def logData(self):
		for name, vals in self.traceRectROI.data.items():
			AnalysisUI.log_data += "%s\t%s\n" % (name, '\t'.join([str(i) for i in vals]))
		AnalysisUI.log_data += "\n"

	def nextROI(self):
		self.buildComboBox()
		self.traceComboBox.setCurrentIndex((self.traceComboBox.currentIndex() + 1) % self.traceComboBox.count())
	def previousROI(self):
		self.buildComboBox()
		self.traceComboBox.setCurrentIndex((self.traceComboBox.currentIndex() - 1) % self.traceComboBox.count())

	def analyzeTrace(self):
		self.loadPuffCombo()
		self.traceRectROI.redraw()

	def indexChanged(self, i=0):
		if self.traceComboBox.updating or i == -1 or i >= len(self.all_rois):
			return

		if hasattr(self, 'oldRoi'):
			try:
				self.oldRoi.sigRegionChanged.disconnect(self.analyzeTrace)
			except:
				pass
			try:
				self.oldRoi.sigRegionChangeFinished.disconnect(self.analyzeTrace)
			except:
				pass

		roi = self.all_rois[i]['roi']
		color = roi.pen.color()
		self.traceComboBox.setStyleSheet("background: %s" % color.name())
		self.traceRectROI.setTrace(self.all_rois[i]['p1trace'])
		self.fillDataTable()
		roi.sigRegionChanged.connect(self.analyzeTrace)
		roi.sigRegionChangeFinished.connect(self.analyzeTrace)
		self.oldRoi = roi
		self.loadPuffCombo()

	def loadPuffCombo(self):
		self.puffComboBox.blockSignals(True)
		trace = self.oldRoi.getTrace()
		blur = scipy.ndimage.filters.gaussian_filter1d(trace, 9)
		highs, lows = peakdetect(trace)
		print(highs)
		highs, lows = peakdetect(blur)
		print(highs)
		ranges = []
		for h in highs:
			h = h[0]
			s = h - 20
			e = h + 20
			ranges.append([s, e])

		self.puffComboBox.clear()
		if len(ranges) == 0:
			self.puffComboBox.addItem("No Trace Selected")
		else:
			for i, rng in enumerate(ranges):
				self.puffComboBox.addItem("Puff #%d" % (i + 1), rng)
		self.puffComboBox.blockSignals(False)

	def puffSelected(self, i):
		lo, hi = self.puffComboBox.itemData(i)
		tr = self.oldRoi.getTrace()
		mi, ma = np.min(tr), np.max(tr)
		self.traceRectROI.setPos([lo, max(0, np.floor(mi))])
		self.traceRectROI.setSize([hi-lo, np.ceil(ma-mi)])


	def nextPuff(self):
		self.puffComboBox.setCurrentIndex((self.puffComboBox.currentIndex()+1) % self.puffComboBox.count())

	def previousPuff(self):
		self.puffComboBox.setCurrentIndex((self.puffComboBox.currentIndex()-1) % self.puffComboBox.count())

	def toggleVisible(self, v):
		#traceWindow.partialThreadUpdatedSignal.connect(buildComboBox)
		self.buildComboBox()
		self.traceRectROI.setVisible(v)
		ymax = 1
		self.traceRectROI.setPos((0, 0))
		self.traceRectROI.setSize((100, ymax))

		self.indexChanged()
		if not v:
			self.tableWidget.clear()
		else:
			self.fillDataTable()

	def closeEvent(self, ev):
		try:
			self.traceRectROI.scene().removeItem(self.traceRectROI)
		except:
			pass
		ev.accept()

	def buildComboBox(self):
		all_rois = []
		for traceWindow in g.m.traceWindows:
			#print(traceWindow.rois)
			all_rois.extend(traceWindow.rois)
		if len(all_rois) == len(self.all_rois):
			return

		self.traceComboBox.blockSignals(True)
		self.traceComboBox.updating = True
		self.traceComboBox.clear()
		self.all_rois = all_rois
		if len(self.all_rois) == 0:
			self.traceComboBox.addItem("No Trace Selected")
		else:
			model = self.traceComboBox.model()
			for i, roiLine in enumerate(self.all_rois):
				item = QStandardItem("ROI #%d" % (i + 1))
				item.setBackground(roiLine['roi'].pen.color())
				model.appendRow(item)
				if roiLine['p1trace'] == self.traceRectROI.traceLine:
					self.traceComboBox.setCurrentIndex(i)

		self.traceComboBox.updating = False
		self.traceComboBox.blockSignals(False)


	def comboBoxClicked(self, ev):
		self.buildComboBox()
		QComboBox.mousePressEvent(self.traceComboBox, ev)

	def fillDataTable(self):
		''' when the region moves, recalculate the polyfit
		data and plot/show it in the table and graph accordingly'''
		if not self.traceRectROI.traceLine:
			return
		self.tableWidget.setData(self.traceRectROI.data)
		self.tableWidget.setHorizontalHeaderLabels(['Ftrace Frames', 'Ftrace Y'])

	@staticmethod
	def show():
		if AnalysisUI.ui is None:
			AnalysisUI.ui = AnalysisUI()
		AnalysisUI.log_data = ''
		QtWidgets.QGroupBox.show(AnalysisUI.ui)

def peakdetect(y_axis, x_axis = None, lookahead = 200, delta=0):
    """
    Converted from/based on a MATLAB script at: 
    http://billauer.co.il/peakdet.html
    
    function for detecting local maxima and minima in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maxima and minima respectively
    
    keyword arguments:
    y_axis -- A list containing the signal over which to find peaks
    
    x_axis -- A x-axis whose values correspond to the y_axis list and is used
        in the return to specify the position of the peaks. If omitted an
        index of the y_axis is used.
        (default: None)
    
    lookahead -- distance to look ahead from a peak candidate to determine if
        it is the actual peak
        (default: 200) 
        '(samples / period) / f' where '4 >= f >= 1.25' might be a good value
    
    delta -- this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            When omitted delta function causes a 20% decrease in speed.
            When used Correctly it can double the speed of the function
    
    
    return: two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tuple
        of: (position, peak_value) 
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do: 
        x, y = zip(*max_peaks)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false
       
    # check input data
    x_axis, y_axis = np.arange(len(y_axis)), y_axis
    # store data length for later use
    length = len(y_axis)
    
    
    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")
    
    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf
    
    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead], 
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x
        
        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]
        
        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found 
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]
    
    
    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass
        
    return [max_peaks, min_peaks]