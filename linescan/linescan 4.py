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
from pyqtgraph.Point import Point
from os.path import expanduser
import os
import math

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector


class FolderSelector(QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a folder.  Once you do, it sets self.folder and it sets the label.
    """
    valueChanged=Signal()
    def __init__(self,filetypes='*.*'):
        QWidget.__init__(self)
        self.button=QPushButton('Select Folder')
        self.label=QLabel('None')
        self.window=None
        self.layout=QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.folder = ''
        
    def buttonclicked(self):
        prompt = 'testing folderSelector'
        self.folder = QFileDialog.getExistingDirectory(g.m, "Select recording folder.", expanduser("~"), QFileDialog.ShowDirsOnly)
        self.label.setText('...'+os.path.split(self.folder)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.folder

    def setValue(self, folder):
        self.folder = str(folder)
        self.label.setText('...' + os.path.split(self.folder)[-1][-20:])    


class PlotWindow(BaseProcess_noPriorWindow):
    def __init__(self,window, channel):
        BaseProcess_noPriorWindow.__init__(self)
        #set current window 
        self.win = window
        #set linescan width
        self.width = 1
        #set integration selection
        self.integrationSelection = 'mean'
        #add ROI
        self.drawROI()
        #set colour channel at start
        self.channelSelection = channel
        #create colour dict
        self.colourChannelDict = {'R': 0, 'G': 1, 'B': 2}
        #get linedata
        #self.getLineData()
        self.getROIdata()
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
        #add label to display xy data
        label = pg.LabelItem(justify='right')
        self.plotWin.addItem(label)
        #add plot
        #self.linescanPlot.plot(x, self.linescanData, pen=penType, symbol='o')       
        self.linescanPlot.plot(self.x, self.y, pen=self.penType, clear=True)
        #add title to plot
        #self.linescanPlot.setTitle('ROI #1') 
        
        #get mouse hover data from line

        #cross hair
        vLine = pg.InfiniteLine(angle=90, movable=False)
        hLine = pg.InfiniteLine(angle=0, movable=False)
        self.linescanPlot.addItem(vLine, ignoreBounds=True)
        self.linescanPlot.addItem(hLine, ignoreBounds=True)
        vb = self.linescanPlot.vb

        def mouseMoved(evt): 
            if self.linescanPlot.sceneBoundingRect().contains(evt):
                mousePoint = vb.mapSceneToView(evt)
                index = int(mousePoint.x())
                if index > 0 and index < len(self.y):
                    label.setText("<span style='font-size: 12pt'>x=%0.2f,   <span style='color: red'>y=%0.2f</span>" % (mousePoint.x(), self.y[index]))
                    vLine.setPos(mousePoint.x())
                    #hLine.setPos(mousePoint.y())
                    hLine.setPos(self.y[index])

        ## use signal proxy to turn original arguments into a tuple
        proxy = pg.SignalProxy(self.linescanPlot.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)  
        self.linescanPlot.scene().sigMouseMoved.connect(mouseMoved)
        return 

# =============================================================================
#     def getLineData(self):
#         #get current frame index
#         self.index = self.win.imageview.currentIndex 
#         #get current frame data
#         data = self.win.imageview.getProcessedImage()
# 
#         #using data.shape to detect if colour image ##TODO fix this to better deal with movies v single images, bw v colour etc
#         if data.shape[2] != 3:
#             data = data[self.index]
#             linescanData = self.win.rois[0].getArrayRegion(data,self.win.imageview.imageItem)
#             self.x = range(linescanData.shape[0])
#             self.y = linescanData
# 
#         else:
#             linescanData = self.win.rois[0].getArrayRegion(data,self.win.imageview.imageItem)
# 
#             #colour image line plot
#             channel = self.colourChannelDict[self.channelSelection]
#     
#             self.x = range(linescanData[:,0].shape[0])
#             self.y = linescanData[:,channel]
# =============================================================================


    def getROIdata(self):
        #get current frame index
        self.index = self.win.imageview.currentIndex 
        #get current frame data
        data = self.win.imageview.getProcessedImage()

        #using data.shape to detect if colour image ##TODO fix this to better deal with movies v single images, bw v colour etc
        if data.shape[2] != 3:
            data = data[self.index]
            
            if self.integrationSelection == 'mean':            
                linescanData = np.mean(self.roi.getArrayRegion(data,self.win.imageview.imageItem),axis=1)
            elif self.integrationSelection == 'min':            
                linescanData = np.min(self.roi.getArrayRegion(data,self.win.imageview.imageItem),axis=1)
            elif self.integrationSelection == 'max':            
                linescanData = np.max(self.roi.getArrayRegion(data,self.win.imageview.imageItem),axis=1)            
                        
            self.x = range(linescanData.shape[0])
            self.y = linescanData

        else:
            if self.integrationSelection == 'mean':            
                linescanData = np.mean(self.roi.getArrayRegion(data,self.win.imageview.imageItem),axis=1)
            elif self.integrationSelection == 'min':            
                linescanData = np.min(self.roi.getArrayRegion(data,self.win.imageview.imageItem),axis=1)
            elif self.integrationSelection == 'max':            
                linescanData = np.max(self.roi.getArrayRegion(data,self.win.imageview.imageItem),axis=1) 

            #colour image line plot
            channel = self.colourChannelDict[self.channelSelection]
    
            self.x = range(linescanData[:,0].shape[0])
            self.y = linescanData[:,channel]        


    def setStyle(self):
        #set pen type
        #penType = None                                                     ## For Scatter Plot
        #penType = pg.mkPen('y', width=3, style=Qt.DashLine)                ## Make a dashed yellow line 2px wide
        self.penType = pg.mkPen(0.5)                                        ## solid grey line 1px wide
        #penType = pg.mkPen(color=(200, 200, 255), style=Qt.DotLine)        ## Dotted pale-blue line
        return

    def setChannel(self, channel):
        self.channelSelection = channel
        self.update()
        return

    def setIntegration(self, intergration):
        self.integrationSelection = intergration
        self.update()
        return

    def setUpdateConnections(self):
        #connect roi change to update
        #self.win.rois[0].sigRegionChanged.connect(self.update)
        self.roi.sigRegionChanged.connect(self.update)        
        #connect frame index change to update
        self.win.sigTimeChanged.connect(self.update)
        return

    def disconnectROI(self):
        self.roi.sigRegionChanged.disconnect(self.update)
        self.win.sigTimeChanged.disconnect(self.update)

    def update(self):
        #self.getLineData()
        self.getROIdata()
        self.linescanPlot.plot(self.x, self.y, clear=True)
        return

    def exportLineData(self):
        return (self.x, self.y)

    def drawROI(self):
        # Custom ROI for selecting an image region
        handle1_x = self.win.rois[0].getLocalHandlePositions()[0][1].x()
        handle1_y = self.win.rois[0].getLocalHandlePositions()[0][1].y()
        handle2_x = self.win.rois[0].getLocalHandlePositions()[1][1].x()
        handle2_y = self.win.rois[0].getLocalHandlePositions()[1][1].y()               
        #imageHeight = self.win.imageDimensions()[1]
        #angle = math.degrees(math.atan2(handle2_y-handle1_y , handle2_x-handle1_x))
        
        def newY(handle1_y,handle2_y):
            diff = abs(handle1_y-handle2_y)
            if handle2_y - handle1_y >= 0:
                return handle1_y - diff
            else:
                return handle1_y + diff
            
        
        self.roi = pg.LineROI([handle1_x, handle1_y], [handle2_x, newY(handle1_y,handle2_y)], width = self.width, pen=(1,9))
        #roi.addScaleHandle([0.5, 1], [0.5, 0.5])
        #roi.addScaleHandle([0, 0.5], [0.5, 0.5])
        self.win.removeAllROIs()
        self.win.imageview.addItem(self.roi)
        #roi.setZValue(10)  # make sure ROI is drawn above image

    def close(self):
        try:
            self.disconnectROI()
            self.win.imageview.removeItem(self.roi)
            self.roi = None
            self.plotWin.close()
        except:
            pass

class OptionsGUI(QDialog):
    def __init__(self, parent = None):
        super(OptionsGUI, self).__init__(parent)
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 150
                
        #ComboBox
        self.integrationSelectorBoxLabel = QLabel("width integration") 
        self.integrationSelectorBox = QComboBox()
        self.integrationSelectorBox.addItems(["mean", "min", "max"])
        self.integrationSelectorBox.currentIndexChanged.connect(self.integrationSelectionChange)
        self.integrationSelection = self.integrationSelectorBox.currentText()        
        
        self.channelSelectorBoxLabel = QLabel("channel") 
        self.channelSelectorBox = QComboBox()
        self.channelSelectorBox.addItems(["R", "G", "B"])
        self.channelSelectorBox.currentIndexChanged.connect(self.channelSelectionChange)
        self.channelSelection = self.channelSelectorBox.currentText()
 
        #buttons
        self.closeButton = QPushButton('Close')
        self.closeButton.pressed.connect(self.closeOptions)
        
        #self.exportButton = QPushButton('Export')
        #self.exportButton.pressed.connect(self.export)        
        
        #grid layout
        layout = QGridLayout()
        layout.setSpacing(10)
        
        layout.addWidget(self.integrationSelectorBoxLabel, 4, 0)        
        layout.addWidget(self.integrationSelectorBox, 4, 1)
        layout.addWidget(self.channelSelectorBoxLabel, 5, 0)  
        layout.addWidget(self.channelSelectorBox, 5, 1)
        #layout.addWidget(self.exportButton, 7, 0)           
        layout.addWidget(self.closeButton, 8, 0)
                     
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("options GUI")

        self.show()

    def channelSelectionChange(self):
        self.channelSelection = self.channelSelectorBox.currentText()
        linescan.linescanPlot.setChannel(self.channelSelection)
        return

    def integrationSelectionChange(self):
        self.integrationSelection = self.integrationSelectorBox.currentText()
        linescan.linescanPlot.setIntegration(self.integrationSelection)
        return

    def closeOptions(self):
        #TODO
        linescan.optionsGUI.close()
        return

#    def export(self):
#        data = linescan.getData()       
#        return

class Linescan(BaseProcess_noPriorWindow):
    """
    Plot linescan in new window based on line ROI in current window
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        self.linescanPlot = None

    def __call__(self):
        '''
        
        '''
        self.linescanPlot.close()
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
        
        #self.exportFolder = FolderSelector('*.txt')
        
        self.items.append({'name': 'linescan', 'string': 'Linescan: ', 'object': self.linescanButton})         
        self.items.append({'name': 'options', 'string': 'Options: ', 'object': self.optionsButton})       
        super().gui()

    def linescan(self):
        #clear old instances
        if self.linescanPlot != None:
            self.linescanPlot.close()
        #create plotWindow instance
        self.linescanPlot = PlotWindow(g.win, 'R')   
        
        return

    def options(self):
        #create optionsGUI instance
        self.optionsGUI = OptionsGUI()      
        return        

    def getData(self):
        return self.linescanPlot.exportLineData()
 
linescan = Linescan()
