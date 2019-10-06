import numpy as np
from qtpy import QtWidgets, QtCore, QtGui
import flika
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui
import pyqtgraph as pg
import time
import os
from os import listdir
from os.path import expanduser, isfile, join
from distutils.version import StrictVersion
from copy import deepcopy
from numpy import moveaxis
from skimage.transform import rescale
from pyqtgraph.dockarea import *
from pyqtgraph import mkPen
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import copy
import pyqtgraph.opengl as gl
from OpenGL.GL import *
from qtpy.QtCore import Signal

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector, FileSelector

from .helperFunctions import *
from .pyqtGraph_classOverwrites import *
from .scalebar_classOverwrite import Scale_Bar_volumeView
from .histogramExtension import HistogramLUTWidget_Overlay
from .texturePlot import *
from .scatterPlot import *
from .overlayOptions import *

from pyqtgraph import HistogramLUTWidget

dataType = np.float16
from matplotlib import cm

#########################################################################################
#############                  slice viewer (3D Display)                #################
#########################################################################################
class SliceViewer(BaseProcess):

    def __init__(self, viewerInstance, A):
        super().__init__()

        self.viewer = viewerInstance
        self.overlayFlag = False

        self.shift_factor = self.viewer.dialogbox.shiftFactor
        self.interpolate = False
        self.theta = self.viewer.dialogbox.theta
        self.inputArrayOrder = self.viewer.getInputArrayOrder()
        self.displayArrayOrder = self.viewer.getDisplayArrayOrder()

        self.trim_last_frame = self.viewer.dialogbox.trim_last_frame

        print('shift factor: '+ str(self.shift_factor))
        print('theta: ' + str(self.theta))
        print('input Array Order: '+ str(self.inputArrayOrder))
        print('display Array Order: ' + str(self.displayArrayOrder))

        #3D plot params
        self.prob = 0.001
        self.threshold_3D = 300
        self.plotCube = False
        self.hideAxis = True

        if self.trim_last_frame:
            self.originalData = A[:, :-1, :, :]
        else:
            self.originalData = A

        self.originalDataShape = self.originalData.shape
        self.nVols = self.originalDataShape[1]
        self.nSteps = self.originalDataShape[0]

        self.originalData= perform_shear_transform(self.originalData, self.shift_factor, self.interpolate, self.originalData.dtype, self.theta, inputArrayOrder=self.inputArrayOrder,displayArrayOrder=self.displayArrayOrder)
        self.dataShape = self.originalData.shape

        self.height = self.dataShape[3]
        self.width = self.dataShape[2]

        self.currentVolume = 0

        self.data = self.originalData[:,0,:,:]

        ## Create window with 4 docks
        self.win = QtWidgets.QMainWindow()
        self.win.resize(1000,800)
        self.win.setWindowTitle('Lightsheet 3D display')

        #create Dock area
        self.area = DockArea()
        self.win.setCentralWidget(self.area)

        #define docks
        self.dock1 = Dock("Top View - Max Projection", size=(500,400))
        self.dock2 = Dock("X Slice", size=(500,400), closable=True)
        self.dock3 = Dock("Y Slice", size=(500,400), closable=True)
        self.dock4 = Dock("Free ROI", size=(500,400), closable=True)
        self.dock5 = Dock("Time Slider", size=(950,50))
        self.dock6 = Dock("Z Slice", size=(500,400), closable=True)
        self.dock7 = Dock("Quick Buttons",size=(50,50))

        #add docks to area
        self.area.addDock(self.dock1, 'left')              ## place d1 at left edge of dock area
        self.area.addDock(self.dock2, 'right')             ## place d2 at right edge of dock area
        self.area.addDock(self.dock3, 'bottom', self.dock1)   ## place d3 at bottom edge of d1
        self.area.addDock(self.dock4, 'bottom', self.dock2)   ## place d4 at bottom edge of d2
        self.area.addDock(self.dock5, 'bottom')            ## place d4 at bottom edge of d2
        self.area.addDock(self.dock6, 'below', self.dock4)    ## tab below d4
        self.area.addDock(self.dock7, 'right', self.dock5)  

        #initialise image widgets
        self.imv1 = pg.ImageView()
        self.imv2 = pg.ImageView()
        self.imv3 = pg.ImageView()
        self.imv4 = pg.ImageView()
        self.imv6 = pg.ImageView()

        self.imageWidgits = [self.imv1,self.imv2,self.imv3,self.imv4,self.imv6]

        #add image widgets to docks
        self.dock1.addWidget(self.imv1)
        self.dock2.addWidget(self.imv3)
        self.dock3.addWidget(self.imv2)
        self.dock4.addWidget(self.imv4)
        self.dock6.addWidget(self.imv6)

        self.imv1.scene.sigMouseMoved.connect(self.mouseMoved_1)
        self.imv2.scene.sigMouseMoved.connect(self.mouseMoved_2)
        self.imv3.scene.sigMouseMoved.connect(self.mouseMoved_3)
        self.imv4.scene.sigMouseMoved.connect(self.mouseMoved_4)
        self.imv6.scene.sigMouseMoved.connect(self.mouseMoved_6)

        #hide dock title-bars at start
        self.dock5.hideTitleBar()
        self.dock7.hideTitleBar()

        #add menu functions
        self.state = self.area.saveState()

        self.menubar = self.win.menuBar()

        self.fileMenu1 = self.menubar.addMenu('&Options')
        self.resetLayout = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Reset Layout')
        self.resetLayout.setShortcut('Ctrl+R')
        self.resetLayout.setStatusTip('Reset Layout')
        self.resetLayout.triggered.connect(self.reset_layout)
        self.fileMenu1.addAction(self.resetLayout)

        self.showTitles = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Show Titles')
        self.showTitles.setShortcut('Ctrl+G')
        self.showTitles.setStatusTip('Show Titles')
        self.showTitles.triggered.connect(self.show_titles)
        self.fileMenu1.addAction(self.showTitles)

        self.hideTitles = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Hide Titles')
        self.hideTitles.setShortcut('Ctrl+H')
        self.hideTitles.setStatusTip('Hide Titles')
        self.hideTitles.triggered.connect(self.hide_titles)
        self.fileMenu1.addAction(self.hideTitles)

        self.hideCursors = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Hide Cursors')
        #self.hideCursors.setShortcut('Ctrl+H')
        self.hideCursors.setStatusTip('Hide Titles')
        self.hideCursors.triggered.connect(self.hide_cursors)
        self.fileMenu1.addAction(self.hideCursors)

        self.showCursors = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Show Cursors')
        #self.showCursors.setShortcut('Ctrl+H')
        self.showCursors.setStatusTip('Hide Titles')
        self.showCursors.triggered.connect(self.show_cursors)
        self.fileMenu1.addAction(self.showCursors)

        self.fileMenu2 = self.menubar.addMenu('&3D Plot')
        self.plot = QtWidgets.QAction(QtGui.QIcon('open.png'), '3D Plot')
        self.plot.setShortcut('Ctrl+P')
        self.plot.setStatusTip('3D Plot')
        self.plot.triggered.connect(self.plot3D)
        self.fileMenu2.addAction(self.plot)

        self.plotOptions = QtWidgets.QAction(QtGui.QIcon('open.png'), '3D Plot Options')
        self.plotOptions.setStatusTip('3D Plot Options')
        self.plotOptions.triggered.connect(self.plot3D_options)
        self.fileMenu2.addAction(self.plotOptions)

        self.fileMenu3 = self.menubar.addMenu('&Texture Plot')
        self.texturePlot = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Texture Plot')
        #self.texturePlot.setShortcut('Ctrl+T')
        self.texturePlot.setStatusTip('Texture Plot')
        self.texturePlot.triggered.connect(self.plotTexture)
        self.fileMenu3.addAction(self.texturePlot)

        self.texturePlotControl = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Texture Plot Control')
        #self.texturePlotControl.setShortcut('Ctrl+M')
        self.texturePlotControl.setStatusTip('Texture Plot')
        self.texturePlotControl.triggered.connect(self.plotTexture_control)
        self.fileMenu3.addAction(self.texturePlotControl)

        self.fileMenu4 = self.menubar.addMenu('&Export')
        self.exportFlika = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Export to flika')
        #self.exportFlika.setShortcut('Ctrl+E')
        self.exportFlika.setStatusTip('Export to Flika')
        self.exportFlika.triggered.connect(self.exportDialog)
        self.fileMenu4.addAction(self.exportFlika)
        
        self.export = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Export Volume to Array')
        self.export.setShortcut('Ctrl+E')
        self.export.setStatusTip('Export to Array')
        self.export.triggered.connect(self.exportCurrentVolToArray)
        self.fileMenu4.addAction(self.export)                       
                
        self.fileMenu5 = self.menubar.addMenu('&Overlay')
        self.overlayArray = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Overlay (from Array)')
        self.overlayArray.setShortcut('Ctrl+O')
        self.overlayArray.setStatusTip('OverlayArray')
        self.overlayArray.triggered.connect(self.overlayArray_start)
        self.fileMenu5.addAction(self.overlayArray)
        
        self.overlayToggle = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Toggle Overlay')
        self.overlayToggle.setShortcut('Ctrl+T')
        self.overlayToggle.setStatusTip('OverlayOff')
        self.overlayToggle.triggered.connect(self.toggleOverlay)
        self.fileMenu5.addAction(self.overlayToggle)
        
        self.overlayArrayWin = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Overlay Options')
        #self.overlayArrayWin.setShortcut('Ctrl+L')
        self.overlayArrayWin.setStatusTip('OverlayOff')
        self.overlayArrayWin.triggered.connect(self.overlayOptions)
        self.fileMenu5.addAction(self.overlayArrayWin)
        
        self.overlayScale_win1 = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Scale Bar Options (Top)')
        #self.overlayScale.setShortcut('Ctrl+S')
        self.overlayScale_win1.setStatusTip('Overlay Scale Bar')
        self.overlayScale_win1.triggered.connect(self.overlayScaleOptions_win1)
        self.fileMenu5.addAction(self.overlayScale_win1)
        
        self.overlayScale_win2 = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Scale Bar Options (Y)')
        #self.overlayScale.setShortcut('Ctrl+S')
        self.overlayScale_win2.setStatusTip('Overlay Scale Bar')
        self.overlayScale_win2.triggered.connect(self.overlayScaleOptions_win2)
        self.fileMenu5.addAction(self.overlayScale_win2)        
        
        self.overlayScale_win3 = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Scale Bar Options (X)')
        #self.overlayScale.setShortcut('Ctrl+S')
        self.overlayScale_win3.setStatusTip('Overlay Scale Bar')
        self.overlayScale_win3.triggered.connect(self.overlayScaleOptions_win3)
        self.fileMenu5.addAction(self.overlayScale_win3) 

        self.overlayScale_win6 = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Scale Bar Options (Z)')
        #self.overlayScale.setShortcut('Ctrl+S')
        self.overlayScale_win6.setStatusTip('Overlay Scale Bar')
        self.overlayScale_win6.triggered.connect(self.overlayScaleOptions_win6)
        self.fileMenu5.addAction(self.overlayScale_win6)         
        
        self.fileMenu6 = self.menubar.addMenu('&Quit')
        self.quit = QtWidgets.QAction(QtGui.QIcon('open.png'), 'Quit')
        self.quit.setShortcut('Ctrl+Q')
        self.quit.setStatusTip('Quit')
        self.quit.triggered.connect(self.close)
        self.fileMenu6.addAction(self.quit)

        #add time slider
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(self.nVols)
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)

        self.dock5.addWidget(self.slider1)

        #add buttons to 'quick button' dock
        self.quickOverlayButton = QtWidgets.QPushButton("Overlay") 
        self.dock7.addWidget(self.quickOverlayButton)
        self.quickOverlayButton.clicked.connect(self.quickOverlay)

        #display window
        self.win.show()

        #define single line roi
        self.roi1 = pg.LineSegmentROI([[10, 64], [120,64]], pen='r')
        self.imv1.addItem(self.roi1)

        #define crosshair rois
        self.dash = mkPen('w', width=1,style=QtCore.Qt.DashLine)
        self.roi2 = pg.LineSegmentROI([[0, int(self.height/2)], [self.width, int(self.height/2)]], pen='y', maxBounds=QtCore.QRect(0,-int(self.height/2),0,self.height))
        self.roi2b = pg.LineSegmentROI([[0, int(self.height/2)], [self.width, int(self.height/2)]], pen=self.dash, maxBounds=QtCore.QRect(0,-int(self.height/2),0,self.height),movable=False)
        self.roi2c = pg.LineSegmentROI([[0, int(self.height/2)], [int(self.width/6), int(self.height/2)]], pen=self.dash, maxBounds=QtCore.QRect(0,-int(self.height/2),0,self.height),movable=False)       
        self.imv1.addItem(self.roi2)
        self.imv6.addItem(self.roi2b)
        self.imv3.addItem(self.roi2c)        

        self.roi3 = pg.LineSegmentROI([[int(self.width/2), 0], [int(self.width/2), self.height]], pen='y', maxBounds=QtCore.QRect(-int(self.width/2),0,self.width,0))
        self.roi3b = pg.LineSegmentROI([[int(self.width/2), 0], [int(self.width/2), self.height]], pen=self.dash, maxBounds=QtCore.QRect(-int(self.width/2),0,self.width,0),movable=False)
        self.roi3c = pg.LineSegmentROI([[int(self.width/2), 0], [int(self.width/2), self.height]], pen=self.dash, maxBounds=QtCore.QRect(-int(self.width/2),0,self.width,0),movable=False)
        self.imv1.addItem(self.roi3)
        self.imv6.addItem(self.roi3b)
        self.imv2.addItem(self.roi3c)

        #define crosshair center roi
        self.roiCenter = pg.CircleROI([int(self.width/2)-10,int(self.height/2)-10], [20, 20], pen=(4,9))
        self.roiCenter.setPen(None)
        self.imv1.addItem(self.roiCenter)

        #define height indicator rois
        self.dash2 = mkPen('r', width=1,style=QtCore.Qt.DashLine)
        self.roi4 = pg.LineSegmentROI([[0, 0], [self.width, 0]], pen=self.dash2, maxBounds=QtCore.QRect(0,-int(self.height/2),0,self.height),movable=False)
        self.imv2.addItem(self.roi4)
        self.roi5 = pg.LineSegmentROI([[0, 0], [0, self.height]], pen=self.dash2, maxBounds=QtCore.QRect(-int(self.width/2),0,self.width,0),movable=False)
        self.imv3.addItem(self.roi5)


        #hide default imageview buttons
        def hideButtons(imv):
            imv.ui.roiBtn.hide()
            imv.ui.menuBtn.hide()

        for imv in self.imageWidgits:
            hideButtons(imv)

        #disconnect roi handles
        def disconnectHandles(roi):
            handles = roi.getHandles()
            if len(handles) == 2:
                handles[0].disconnectROI(roi)
                handles[1].disconnectROI(roi)
                handles[0].currentPen = mkPen(None)
                handles[1].currentPen = mkPen(None)
                handles[0].pen = mkPen(None)
                handles[1].pen = mkPen(None)
            elif len(handles) == 1:
                #handles[0].disconnectROI(roi)
                #handles[0].currentPen = mkPen(None)
                #handles[0].pen = mkPen(None)
                roi.removeHandle(-1)

        self.MainROIList = [self.roi2,self.roi3,self.roi2b,self.roi2c,self.roi3b,self.roi3c,self.roi4,self.roi5,self.roiCenter]

        for roi in self.MainROIList:
            disconnectHandles(roi)
       

        #add max projection data to main window
        self.imv1.setImage(self.maxProjection(self.data)) #display topview (max of slices)

        #add sliceable data to side window
        self.imv6.setImage(self.data)

        #connect roi updates
        self.roi1.sigRegionChanged.connect(self.update)
        self.roi2.sigRegionChanged.connect(self.update_2)
        self.roi3.sigRegionChanged.connect(self.update_3)

        self.imv6.sigTimeChanged.connect(self.update_6)

        self.roi2.sigRegionChangeFinished.connect(self.update_center_fromLines)
        self.roi3.sigRegionChangeFinished.connect(self.update_center_fromLines)        
        self.roiCenter.sigRegionChanged.connect(self.update_center)

        #initial update to populate roi windows
        self.update()
        self.update_2()
        self.update_3()

        #autolevel roi windows (not main) at start
        for imv in self.imageWidgits[1:]:
            imv.autoLevels()

        self.slider1.valueChanged.connect(self.timeUpdate)

        #initialize time index for z-slice
        self.index6 = 0        
        #correct roi4 position
        self.update_6()
        
        self.OverlayLUT = 'inferno'
        self.OverlayMODE = QtGui.QPainter.CompositionMode_SourceOver
        self.OverlayOPACITY = 0.5
        self.useOverlayLUT = False
        
        self.gradientPreset = 'grey' #  'thermal','flame','yellowy','bipolar','spectrum','cyclic','greyclip','grey'
        self.usePreset = False
        self.gradientState_1 = None
        self.gradientState_2 = None
        self.gradientState_3 = None
        self.gradientState_4 = None
        self.gradientState_6 = None   
        self.useSharedState = False
        self.sharedState = None
        self.sharedLevels = None
               
        #init overlay levels 
        self.levels_1 = self.imv1.getHistogramWidget().getLevels()
        self.levels_2 = self.imv2.getHistogramWidget().getLevels()
        self.levels_3 = self.imv3.getHistogramWidget().getLevels()
        self.levels_4 = self.imv4.getHistogramWidget().getLevels()     
        self.levels_6 = self.imv6.getHistogramWidget().getLevels()          

        self.overlayFlag = False
        self.overlayArrayLoaded = False
        
        #create lists of objects
        self.dockList = [self.dock1, self.dock2, self.dock3, self.dock4, self.dock6]
        self.roiList = [self.roi2b, self.roi2c, self.roi3b, self.roi3c, self.roi4, self.roi5]

        #link TOP overlay histgramLUT to other windows
        self.imv1.getHistogramWidget().item.sigLevelsChanged.connect(self.setMainLevels)
        self.histogramsLinked = True




    #define update calls for each roi
    def update(self):
        levels = self.imv4.getHistogramWidget().getLevels()
        self.d1 = self.roi1.getArrayRegion(self.data, self.imv1.imageItem, axes=(1,2))
        self.imv4.setImage(self.d1, autoRange=False, autoLevels=False, levels=levels)
        
        if self.overlayFlag:
            self.runOverlayUpdate(1)
            self.runOverlayUpdate(4)            

    def update_2(self):
        levels = self.imv2.getHistogramWidget().getLevels()
        self.d2 = np.rot90(self.roi2.getArrayRegion(self.data, self.imv1.imageItem, axes=(1,2)), axes=(1,0))
        self.imv2.setImage(self.d2, autoRange=False, autoLevels=False, levels=levels)
        self.roi2b.setPos(self.roi2.pos(), finish=False)
        self.roi2c.setPos(self.roi2.pos(), finish=False)    
        
        if self.overlayFlag:
            self.runOverlayUpdate(2)

    def update_3(self):
        levels = self.imv3.getHistogramWidget().getLevels()
        self.d3 = self.roi3.getArrayRegion(self.data, self.imv1.imageItem, axes=(1,2))
        self.imv3.setImage(self.d3, autoRange=False, autoLevels=False, levels=levels)
        self.roi3b.setPos(self.roi3.pos(),finish=False)
        self.roi3c.setPos(self.roi3.pos(),finish=False) 
        
        if self.overlayFlag:
            self.runOverlayUpdate(3)

    def update_6(self):
        #self.index6 = self.imv6.currentIndex
        roi4_x, roi4_y = self.roi4.pos()
        roi5_x, roi5_y = self.roi5.pos()
        self.roi4.setPos((roi4_x, self.imv2.imageItem.height()-self.imv6.currentIndex)) #check this is starting at right end
        self.roi5.setPos((self.imv6.currentIndex, roi5_y))

        if self.overlayFlag:
            self.runOverlayUpdate(6)

    def update_center_fromLines(self):
        #move center roi after cursor lines moved
        self.roiCenter.setPos((self.roi3.pos()[0]+int(self.width/2)-10,self.roi2.pos()[1]+int(self.height/2)-10), finish=False)  
        self.roiCenter.setPos((self.roi3.pos()[0]+int(self.width/2)-10,self.roi2.pos()[1]+int(self.height/2)-10), finish=False)  
        
    def update_center(self):
        #move cursor lines as center roi is moved
        self.roi2.setPos((self.roi2.pos()[0],self.roiCenter.pos()[1]-int(self.height/2)+10), finish=False)
        self.roi3.setPos((self.roiCenter.pos()[0]-int(self.width/2)+10,self.roi3.pos()[1]), finish=False)

    def runOverlayUpdate(self, win):
            self.updateOverlayLevels()
            self.overlayOff_temp(win)
            self.overlayFlag = True
            self.overlayUpdate(win) 
            return
    

    #connect time slider
    def timeUpdate(self,value):
        self.currentVolume = value
        self.index6 = self.imv6.currentIndex
        levels1 = self.imv1.getHistogramWidget().getLevels()
        levels6 = self.imv6.getHistogramWidget().getLevels()
        self.data = self.originalData[:,value,:,:]
        self.imv1.setImage(self.maxProjection(self.data),autoRange=False, levels=levels1)
        self.imv6.setImage(self.data,autoRange=False, levels=levels6)
        self.imv6.setCurrentIndex(self.index6)
        self.update()
        self.update_2()
        self.update_3()
        self.update_6()
        return

    def reset_layout(self):
        self.area.restoreState(self.state)

    def hide_titles(self,_):
        for dock in self.dockList:
            dock.hideTitleBar()

    def show_titles(self):
        for dock in self.dockList:
            dock.showTitleBar()
        return

    def hide_cursors(self):
        for roi in self.roiList:
            roi.setPen(None)
        return

    def show_cursors(self):
        self.roi2b.setPen(self.dash)
        self.roi2c.setPen(self.dash)        
        self.roi3b.setPen(self.dash)
        self.roi3c.setPen(self.dash)        
        self.roi4.setPen(self.dash2)
        self.roi5.setPen(self.dash2)
        return

    def maxProjection(self,data):
        return np.max(data,axis=0)

    def plot3D(self):
        vol_downSample = copy.deepcopy(self.data)

        #downsample data
        mask = np.random.choice([False, True], vol_downSample.shape, p=[self.prob, 1-self.prob])
        vol_downSample[mask] = 0

        ##z,x,y = vol.nonzero()
        z,x,y =(vol_downSample > self.threshold_3D).nonzero()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, -z, zdir='z', c= 'red', s=1)

        vol_corners = getCorners(self.data)
        z_c,x_c,y_c = vol_corners.nonzero()

        ax.scatter(x_c, y_c, -z_c, zdir='z', c= 'green', s=5)

        x_min,x_max,y_min,y_max,z_min,z_max = getDimensions(self.data)
        maxDim = getMaxDimension(self.data)

        ax.set_xlim(0, maxDim)
        ax.set_ylim(0, maxDim)
        ax.set_zlim(0, -maxDim)

        ax.view_init(0,180)

        #hide axis
        if self.hideAxis:
            ax.axis('off')

        plt.draw()

        #plot array outline
        outline = [(x_min,y_min,z_min),(x_min,y_max,z_min),(x_max,y_min,z_min),(x_min,y_min,-z_max)]

        # plot array shape
        if self.plotCube:
            plot_cube(ax, outline)

        fig.show()

    def plot3D_options(self):
        self.plot3D_optionWin = plot3D_options(self.viewer, self.prob,self.threshold_3D)
        self.plot3D_optionWin.show()
        return

    def plotTexture(self):
        self.texturePlot = plotTexture(self.data)
        self.texturePlot.w.show()

    def plotTexture_control(self):
        X,Y,Z = self.texturePlot.getSliceValues()
        self.textureControl = textureDialog_win(self.viewer, self.texturePlot.getShape(), X, Y, Z)
        self.textureControl.show()
        return

    def close(self):
        self.roi1.sigRegionChanged.disconnect(self.update)
        self.roi2.sigRegionChanged.disconnect(self.update_2)
        self.roi3.sigRegionChanged.disconnect(self.update_3)
        
        for imv in self.imageWidgits:
            imv.close()
        
        self.exportDialogWin.close()
        self.win.close()
        self.win.destroy()
        return

    def closeEvent(self, event):
        event.accept()

    def setProb(self, prob):
        self.prob = prob
        return

    def setThreshold(self, thresh):
        self.threshold_3D = thresh
        return

    def setPlotCube(self, flag):
        self.plotCube = flag
        return

    def setPlotAxis(self, flag):
        self.hideAxis = flag
        return

    def exportDialog(self):
        self.exportDialogWin = exportDialog_win(self.viewer)
        self.exportDialogWin.show()
        return

    def exportCurrentVolToArray(self):
        arraySavePath = QtWidgets.QFileDialog.getSaveFileName(self.win,'Save File', os.path.expanduser("~/Desktop"), 'Numpy array (*.npy)')
        arraySavePath = str(arraySavePath[0])
        self.viewer.savePath = arraySavePath
        self.viewer.exportArray(vol=self.currentVolume)
        return

    def getXWin(self):
        return self.data.swapaxes(0,1)

    def getYWin(self):
        return self.data.swapaxes(2,0)

    def getZWin(self):
        return self.data.swapaxes(1,2)

    def mouseMoved(self, point, imv):
        '''mouseMoved(self,point)
        Event handler function for mouse movement.
        '''
        point=imv.getImageItem().mapFromScene(point)
        self.point = point
        self.x = point.x()
        self.y = point.y()
        image=imv.getImageItem().image
        if self.x < 0 or self.y < 0 or self.x >= image.shape[0] or self.y >= image.shape[1]:
            pass# if we are outside the image
        else:
            z=imv.currentIndex
            value=image[int(self.x),int(self.y)]
            g.m.statusBar().showMessage('x={}, y={}, z={}, value={}'.format(int(self.x),int(self.y),z,value))

    def mouseMoved_1(self,point):
        self.mouseMoved(point,self.imv1)
        return

    def mouseMoved_2(self,point):
        self.mouseMoved(point,self.imv2)
        return

    def mouseMoved_3(self,point):
        self.mouseMoved(point,self.imv3)
        return

    def mouseMoved_4(self,point):
        self.mouseMoved(point,self.imv4)
        return

    def mouseMoved_6(self,point):
        self.mouseMoved(point,self.imv6)
        return

    def overlayArray_start(self):
        #get array data
        A_path = open_file_gui(directory=os.path.expanduser("~/Desktop"),filetypes='*.npy')
        g.m.statusBar().showMessage("Importing Array: " + A_path)
        self.A_overlay = np.load(str(A_path))
        #perform transform
        self.A_overlay= perform_shear_transform(self.A_overlay, self.shift_factor, self.interpolate, self.originalData.dtype, self.theta, inputArrayOrder=self.inputArrayOrder,displayArrayOrder=self.displayArrayOrder)
        #set flags
        self.overlayFlag = True      
        self.overlayArrayLoaded = True
        #update overlay
        self.overlayUpdate(0)
        #link TOP overlay histgramLUT to other windows
        self.bgItem_imv1.hist_luttt.item.sigLevelsChanged.connect(self.setOverlayLevels)
        return


    def getLUT(self, lutNAME = "nipy_spectral"):
        colormap = cm.get_cmap(lutNAME)  # cm.get_cmap("CMRmap")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        return lut

    def overlay(self, overlayImage, imv, levels, gradientState):
        bgItem = pg.ImageItem()        
        bgItem.setImage(overlayImage, autoRange=False, autoLevels=False, levels=levels,opacity=self.OverlayOPACITY)
        bgItem.setCompositionMode(self.OverlayMODE)
        imv.view.addItem(bgItem)

        bgItem.hist_luttt = HistogramLUTWidget(fillHistogram = False)
        
        if gradientState == None or self.usePreset == True:
            bgItem.hist_luttt.item.gradient.loadPreset(self.gradientPreset)
        else:
            
            if self.useSharedState:
                bgItem.hist_luttt.item.gradient.restoreState(self.sharedState)
                levels = self.sharedLevels

            else:    
                bgItem.hist_luttt.item.gradient.restoreState(gradientState)
        
        bgItem.hist_luttt.setMinimumWidth(110)
        bgItem.hist_luttt.setImageItem(bgItem)

        #bgItem.hist_luttt.item.levelMode = 'rgba'
        
        #unlinked LUT as placeholder to stop image resizing on update
        state = bgItem.hist_luttt.item.gradient.saveState()
        bgItem.blank_luttt = HistogramLUTWidget_Overlay()
        bgItem.blank_luttt.item.gradient.restoreState(state)        
        
        imv.ui.gridLayout.addWidget(bgItem.blank_luttt, 0, 4, 1, 4)       
        imv.ui.gridLayout.addWidget(bgItem.hist_luttt, 0, 4, 1, 4)        
        bgItem.setLevels(levels)
        bgItem.hist_luttt.item.setLevels(levels[0],levels[1])
        
        if self.useOverlayLUT:
            bgItem.hist_luttt.setLUT(self.getLUT(lutNAME = self.OverlayLUT))
        
        return bgItem



    def overlayOff_temp(self, win):
        def overlay_hide_temp(bgItem,imv):
            bgItem.hist_luttt.hide()
            imv.ui.gridLayout.removeWidget(bgItem.hist_luttt)
            imv.view.removeItem(bgItem)
            return        
        
        if self.overlayFlag:
            if win == 1:
                overlay_hide_temp(self.bgItem_imv1,self.imv1)
            elif win == 2:
                overlay_hide_temp(self.bgItem_imv2,self.imv2) 
            elif win == 3:
                overlay_hide_temp(self.bgItem_imv3,self.imv3)
            elif win == 4:    
                overlay_hide_temp(self.bgItem_imv4,self.imv4) 
            elif win == 6:
                overlay_hide_temp(self.bgItem_imv6,self.imv6)
            self.overlayFlag = False            
        return



    def overlayOff(self):        
        def overlay_hide(bgItem,imv):
            bgItem.hist_luttt.hide()
            bgItem.blank_luttt.hide()
            imv.ui.gridLayout.removeWidget(bgItem.hist_luttt)
            imv.ui.gridLayout.removeWidget(bgItem.blank_luttt)
            imv.view.removeItem(bgItem)  
            #bgItem.hist_luttt = None 
            #bgItem.blank_luttt = None 
            #bgItem = None             
            return
        
        if self.overlayFlag:
            overlay_hide(self.bgItem_imv1,self.imv1)
            overlay_hide(self.bgItem_imv2,self.imv2) 
            overlay_hide(self.bgItem_imv3,self.imv3)
            overlay_hide(self.bgItem_imv4,self.imv4) 
            overlay_hide(self.bgItem_imv6,self.imv6)
            self.overlayFlag = False   
            
        self.resetImages()            
        return


    def toggleOverlay(self):
        if self.overlayArrayLoaded == False:
            print('load array first!')
            return
        
        if self.overlayFlag:
            self.overlayOff()        
        else:
            self.overlayFlag = True
            self.updateAllOverlayWins()
        return


    def setOverlayLevels(self):  
        if self.histogramsLinked == False:
            return
        levels = self.bgItem_imv1.getLevels()
        bgItemList = [self.bgItem_imv2,self.bgItem_imv3,self.bgItem_imv4,self.bgItem_imv6]
        for bgItem in bgItemList:
            bgItem.hist_luttt.item.setLevels(levels[0],levels[1])
        return

    def setMainLevels(self):
        if self.histogramsLinked == False:
            return
        levels = self.imv1.getLevels()
        imvList = [self.imv2,self.imv3,self.imv4,self.imv6]
        for imv in imvList:
            imv.getHistogramWidget().item.setLevels(levels[0],levels[1])
        return

    def updateOverlayLevels(self):
        self.levels_1 = self.bgItem_imv1.getLevels()
        self.levels_2 = self.bgItem_imv2.getLevels()   
        self.levels_3 = self.bgItem_imv3.getLevels() 
        self.levels_4 = self.bgItem_imv4.getLevels()    
        self.levels_6 = self.bgItem_imv6.getLevels()          
        self.gradientState_1 = self.bgItem_imv1.hist_luttt.item.gradient.saveState()
        self.gradientState_2 = self.bgItem_imv2.hist_luttt.item.gradient.saveState()
        self.gradientState_3 = self.bgItem_imv3.hist_luttt.item.gradient.saveState()
        self.gradientState_4 = self.bgItem_imv4.hist_luttt.item.gradient.saveState()
        self.gradientState_6 = self.bgItem_imv6.hist_luttt.item.gradient.saveState()
        #reconnect link TOP overlay histgramLUT to other windows
        self.bgItem_imv1.hist_luttt.item.sigLevelsChanged.connect(self.setOverlayLevels)

    def overlayUpdate(self,win):
        self.A_overlay_currentVol =self.A_overlay[:,0,:,:] #first volume

        #overlay images
        if win == 0:
            self.bgItem_imv1 = self.overlay(self.maxProjection(self.A_overlay_currentVol), self.imv1, self.levels_1, self.gradientState_1)
            self.bgItem_imv3 = self.overlay(self.roi3.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), self.imv3, self.levels_3,self.gradientState_3)
            self.bgItem_imv2 = self.overlay(np.rot90(self.roi2.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), axes=(1,0)), self.imv2,self.levels_2,self.gradientState_2)
            self.bgItem_imv4 = self.overlay(self.roi1.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), self.imv4, self.levels_4,self.gradientState_4)
            self.bgItem_imv6 = self.overlay(self.A_overlay_currentVol[self.index6], self.imv6, self.levels_6,self.gradientState_6)        
                
        elif win == 1:
            self.bgItem_imv1 = self.overlay(self.maxProjection(self.A_overlay_currentVol), self.imv1,self.levels_1,self.gradientState_1)
        elif win== 3:
            self.bgItem_imv3 = self.overlay(self.roi3.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), self.imv3,self.levels_3,self.gradientState_3)
        elif win == 2:  
            self.bgItem_imv2 = self.overlay(np.rot90(self.roi2.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), axes=(1,0)), self.imv2,self.levels_2,self.gradientState_2)
        elif win == 4:   
            self.bgItem_imv4 = self.overlay(self.roi1.getArrayRegion(self.A_overlay_currentVol, self.imv1.imageItem, axes=(1,2)), self.imv4, self.levels_4,self.gradientState_4)
        elif win == 6:    
            self.bgItem_imv6 = self.overlay(self.A_overlay_currentVol[self.imv6.currentIndex], self.imv6, self.levels_6,self.gradientState_6)
        return

    def updateAllOverlayWins(self):
        for winNumber in [1,2,3,4,6]:
            self.runOverlayUpdate(winNumber)

    def setOverlayLUT(self, lut):
        self.OverlayLUT = lut

    def overlayOptions(self):
        if self.overlayArrayLoaded == False:
            print('load array first!')
            return
        self.overlayOptionsWin = OverlayOptions(self.viewer)   
        self.overlayOptionsWin.show()
        return

    def setOverlayMode(self,keyText):
        modeDict = {"Source Overlay" : QtGui.QPainter.CompositionMode_SourceOver,
                    "Overlay" : QtGui.QPainter.CompositionMode_Overlay,
                    "Plus" : QtGui.QPainter.CompositionMode_Plus,
                    "Mutiply" : QtGui.QPainter.CompositionMode_Multiply}
        
        self.OverlayMODE = modeDict[keyText]

    def resetImages(self):
        #add max projection data to main window
        self.imv1.setImage(self.maxProjection(self.data))
        #add sliceable data to side window
        self.imv6.setImage(self.data)
        #update to populate roi windows
        self.update()
        self.update_2()
        self.update_3() 
        #self.update_4() 
        #correct roi4 position
        self.update_6()


    def overlayScaleOptions_win1(self):
        scale_bar_1=Scale_Bar_volumeView(self.imv1, self.imv1.image.shape[1], self.imv1.image.shape[0])
        scale_bar_1.gui()
        return

    def overlayScaleOptions_win2(self):
        scale_bar_2=Scale_Bar_volumeView(self.imv2, self.imv2.image.shape[1], self.imv2.image.shape[0])
        scale_bar_2.gui()
        return
    
    def overlayScaleOptions_win3(self):
        scale_bar_3=Scale_Bar_volumeView(self.imv3, self.imv3.image.shape[1], self.imv3.image.shape[0])
        scale_bar_3.gui()
        return

    def overlayScaleOptions_win4(self):
        scale_bar_4=Scale_Bar_volumeView(self.imv4, self.imv4.image.shape[1], self.imv4.image.shape[0])
        scale_bar_4.gui()
        return

    
    def overlayScaleOptions_win6(self):
        scale_bar_6=Scale_Bar_volumeView(self.imv6, self.imv6.image.shape[1], self.imv6.image.shape[0])
        scale_bar_6.gui()
        return

    def quickOverlay(self):
        if self.overlayArrayLoaded:
            print('Array already loaded')
            return
        else:                        
            self.A_overlay = self.viewer.getVolumeArray(self.currentVolume)
            #perform transform
            self.A_overlay= perform_shear_transform(self.A_overlay, self.shift_factor, self.interpolate, self.originalData.dtype, self.theta, inputArrayOrder=self.inputArrayOrder,displayArrayOrder=self.displayArrayOrder)
            #set flags
            self.overlayFlag = True      
            self.overlayArrayLoaded = True
            #update overlay
            self.overlayUpdate(0)
            #link TOP overlay histgramLUT to other windows
            self.bgItem_imv1.hist_luttt.item.sigLevelsChanged.connect(self.setOverlayLevels)
        return

    
class exportDialog_win(QtWidgets.QDialog):
    def __init__(self, viewerInstance, parent = None):
        super(exportDialog_win, self).__init__(parent)

        self.viewer = viewerInstance

        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 200

        #buttons
        self.button1 = QtWidgets.QPushButton("Export Z view")
        self.button2 = QtWidgets.QPushButton("Export X view")
        self.button3 = QtWidgets.QPushButton("Export Y view")

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.button1, 0, 0)
        layout.addWidget(self.button2, 1, 0)
        layout.addWidget(self.button3, 2, 0)

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Export Options")

        #connect buttons
        self.button1.clicked.connect(self.exportZ)
        self.button2.clicked.connect(self.exportX)
        self.button3.clicked.connect(self.exportY)
        return

    def exportZ(self):
        self.z_displayWindow = Window(self.viewer.viewer.getZWin(),'Z view')
        return

    def exportX(self):
        self.x_displayWindow = Window(self.viewer.viewer.getXWin(),'X view')
        return

    def exportY(self):
        self.y_displayWindow = Window(self.viewer.viewer.getYWin(),'Y view')
        return

