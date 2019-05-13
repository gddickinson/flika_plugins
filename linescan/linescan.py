from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
from distutils.version import StrictVersion
import flika
from flika import global_vars as g
from flika.window import Window


flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector

class PlotWindow(BaseProcess_noPriorWindow):
    def __init__(self,window):
        BaseProcess_noPriorWindow.__init__(self)
        #use current window to get ROI data
        self.win = window
        self.getLineData()
        #set plot line style
        self.setStyle()
        #initialize plot
        self.initalizePlot()
        #set connections
        self.setUpdateConnections()

    def initalizePlot(self):
        #create window for plot
        self.plotWin = pg.GraphicsWindow(title="Linescan Plot")       
        #add plot object to window
        self.linescanPlot = self.plotWin.addPlot()
        #add plot
        #self.linescanPlot.plot(x, self.linescanData, pen=penType, symbol='o')       
        self.linescanPlot.plot(self.x, self.y, pen=self.penType, clear=True)
        #add title to plot
        #self.linescanPlot.setTitle('ROI #1')       
        return 

    def getLineData(self):
        #get current frame index
        self.index = self.win.imageview.currentIndex 
        #get current frame data
        data = self.win.imageview.getProcessedImage()

        #using data.shape to detect if colour image ##TODO fix this to better deal with movies v single images, bw v colour etc
        if data.shape[2] != 3:
            data = data[self.index]
            linescanData = self.win.rois[0].getArrayRegion(data,self.win.imageview.imageItem)
            self.x = range(linescanData.shape[0])
            self.y = linescanData

        else:
            linescanData = self.win.rois[0].getArrayRegion(data,self.win.imageview.imageItem)

            #colour image line plot
            #red
            redLine = linescanData[:,0]
            #green
            greenLine = linescanData[:,1]
            #blue
            blueLine = linescanData[:,2]
    
            self.x = range(linescanData[:,0].shape[0])
            self.y = redLine #curently only displaying red channel data

    def setStyle(self):
        #set pen type
        #penType = None                                                     ## For Scatter Plot
        #penType = pg.mkPen('y', width=3, style=Qt.DashLine)                ## Make a dashed yellow line 2px wide
        self.penType = pg.mkPen(0.5)                                        ## solid grey line 1px wide
        #penType = pg.mkPen(color=(200, 200, 255), style=Qt.DotLine)        ## Dotted pale-blue line
        return

    def setUpdateConnections(self):
        #connect roi change to update
        self.win.rois[0].sigRegionChanged.connect(self.update)
        #connect frame index change to update
        self.win.sigTimeChanged.connect(self.update)
        return

    def update(self):
        self.getLineData()
        self.linescanPlot.plot(self.x, self.y, clear=True)
        return


class OptionsGUI(QDialog):
    def __init__(self, parent = None):
        super(OptionsGUI, self).__init__(parent)
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 150
        
        #spinboxes
        self.spinLabel1 = QLabel("line width") 
        self.SpinBox1 = QSpinBox()
        self.SpinBox1.setRange(1,10)
        self.SpinBox1.setValue(1)
        
        #ComboBox
        self.channelSelectorBoxLabel = QLabel("channel") 
        self.channelSelectorBox = QComboBox()
        self.channelSelectorBox.addItems(["R", "G", "B"])
        self.channelSelectorBox.currentIndexChanged.connect(self.channelSelectionChange)
        self.channelSelection = self.channelSelectorBox.currentText()
 
        #buttons
        self.closeButton = QPushButton('Close')
        self.closeButton.pressed.connect(self.closeOptions)
        
        #grid layout
        layout = QGridLayout()
        layout.setSpacing(10)
        
        layout.addWidget(self.spinLabel1, 4, 0)        
        layout.addWidget(self.SpinBox1, 4, 1)
        layout.addWidget(self.channelSelectorBoxLabel, 5, 0)  
        layout.addWidget(self.channelSelectorBox, 5, 1)
        layout.addWidget(self.closeButton, 7, 0)
        
        
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("options GUI")

        self.show()

    def channelSelectionChange(self):
        return

    def closeOptions(self):
        return

class Linescan(BaseProcess_noPriorWindow):
    """
    Plot linescan in new window based on line ROI in current window
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)

    def __call__(self):
        '''
        
        '''
        pass
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)
        self.linescanPlot.close()

    def gui(self):
        self.gui_reset()
        self.linescanButton = QPushButton('Start')
        self.linescanButton.pressed.connect(self.linescan)
        self.optionsButton = QPushButton('Show Options Menu')
        self.optionsButton.pressed.connect(self.options)       
        self.items.append({'name': 'linescan', 'string': 'Linescan: ', 'object': self.linescanButton})
        self.items.append({'name': 'options', 'string': 'Options: ', 'object': self.optionsButton})       
        super().gui()

    def linescan(self):
        #create plotWindow instance
        self.linescanPlot = PlotWindow(g.win)      
        return

    def options(self):
        #create optionsGUI instance
        self.optionsWGUI = OptionsGUI()      
        return        
 
linescan = Linescan()
