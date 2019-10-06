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

from pyqtgraph import HistogramLUTWidget

dataType = np.float16
from matplotlib import cm

### disable messages from PyQt ################
def handler(msg_type, msg_log_context, msg_string):
    pass

QtCore.qInstallMessageHandler(handler)
#####################################################


#########################################################################################
#############                  slice viewer (3D Display)                #################
#########################################################################################
class SliceViewer(BaseProcess):

    def __init__(self, A):
        super().__init__()

        self.overlayFlag = False

        self.shift_factor = camVolumeSlider.dialogbox.shiftFactor
        self.interpolate = False
        self.theta = camVolumeSlider.dialogbox.theta
        self.inputArrayOrder = camVolumeSlider.getInputArrayOrder()
        self.displayArrayOrder = camVolumeSlider.getDisplayArrayOrder()

        self.trim_last_frame = camVolumeSlider.dialogbox.trim_last_frame

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
        self.d1 = Dock("Top View - Max Projection", size=(500,400))
        self.d2 = Dock("X Slice", size=(500,400), closable=True)
        self.d3 = Dock("Y Slice", size=(500,400), closable=True)
        self.d4 = Dock("Free ROI", size=(500,400), closable=True)
        self.d5 = Dock("Time Slider", size=(1000,50))
        self.d6 = Dock("Z Slice", size=(500,400), closable=True)

        #add docks to area
        self.area.addDock(self.d1, 'left')              ## place d1 at left edge of dock area
        self.area.addDock(self.d2, 'right')             ## place d2 at right edge of dock area
        self.area.addDock(self.d3, 'bottom', self.d1)   ## place d3 at bottom edge of d1
        self.area.addDock(self.d4, 'bottom', self.d2)   ## place d4 at bottom edge of d2
        self.area.addDock(self.d5, 'bottom')            ## place d4 at bottom edge of d2
        self.area.addDock(self.d6, 'below', self.d4)    ## tab below d4

        #initialise image widgets
        self.imv1 = pg.ImageView()
        self.imv2 = pg.ImageView()
        self.imv3 = pg.ImageView()
        self.imv4 = pg.ImageView()
        self.imv6 = pg.ImageView()

        self.imageWidgits = [self.imv1,self.imv2,self.imv3,self.imv4,self.imv6]

        #add image widgets to docks
        self.d1.addWidget(self.imv1)
        self.d2.addWidget(self.imv3)
        self.d3.addWidget(self.imv2)
        self.d4.addWidget(self.imv4)
        self.d6.addWidget(self.imv6)

        self.imv1.scene.sigMouseMoved.connect(self.mouseMoved_1)
        self.imv2.scene.sigMouseMoved.connect(self.mouseMoved_2)
        self.imv3.scene.sigMouseMoved.connect(self.mouseMoved_3)
        self.imv4.scene.sigMouseMoved.connect(self.mouseMoved_4)
        self.imv6.scene.sigMouseMoved.connect(self.mouseMoved_6)

        #hide dock title-bars at start
        self.d5.hideTitleBar()

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

        self.d5.addWidget(self.slider1)

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
        self.dockList = [self.d1, self.d2, self.d3, self.d4, self.d6]
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
        #global state
        self.area.restoreState(self.state)

    def hide_titles(self,_):
        for dock in self.dockList:
            dock.hideTitleBar()

    def show_titles(self):
        for dock in self.dockList:
            dock.d1.showTitleBar()
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
        self.plot3D_optionWin = plot3D_options(self.prob,self.threshold_3D)
        self.plot3D_optionWin.show()
        return

    def plotTexture(self):
        self.texturePlot = plotTexture(self.data)
        self.texturePlot.w.show()

    def plotTexture_control(self):
        X,Y,Z = self.texturePlot.getSliceValues()
        self.textureControl = textureDialog_win(self.texturePlot.getShape(), X, Y, Z)
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
        self.exportDialogWin = exportDialog_win()
        self.exportDialogWin.show()
        return

    def exportCurrentVolToArray(self):
        arraySavePath = QtWidgets.QFileDialog.getSaveFileName(self.win,'Save File', os.path.expanduser("~/Desktop"), 'Numpy array (*.npy)')
        arraySavePath = str(arraySavePath[0])
        camVolumeSlider.savePath = arraySavePath
        camVolumeSlider.exportArray(vol=self.currentVolume)
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
        self.overlayOptionsWin = OverlayOptions()   
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

#########################################################################################
#############                  volumeViewer class                ########################
#########################################################################################
class CamVolumeSlider(BaseProcess):

    def __init__(self):
        super().__init__()
        self.nVols = 1

        #create dtype dict
        self.dTypeDict = {
                'float16': np.float16,
                'float32': np.float32,
                'float64': np.float64,
                'int8': np.uint8,
                'int16': np.uint16,
                'int32': np.uint32,
                'int64': np.uint64                                  }

        self.dataType = dataType

        #create array order dict
        self.arrayDict = {
                '[0, 1, 2, 3]': [0, 1, 2, 3],
                '[0, 1, 3, 2]': [0, 1, 3, 2],
                '[0, 2, 1, 3]': [0, 2, 1, 3],
                '[0, 2, 3, 1]': [0, 2, 3, 1],
                '[0, 3, 1, 2]': [0, 3, 1, 2],
                '[0, 3, 2, 1]': [0, 3, 2, 1],
                '[1, 0, 2, 3]': [1, 0, 2, 3],
                '[1, 0, 3, 2]': [1, 0, 3, 2],
                '[1, 2, 0, 3]': [1, 2, 0, 3],
                '[1, 2, 3, 0]': [1, 2, 3, 0],
                '[1, 3, 0, 2]': [1, 3, 0, 2],
                '[1, 3, 2, 0]': [1, 3, 2, 0],
                '[2, 0, 1, 3]': [2, 0, 1, 3],
                '[2, 0, 3, 1]': [2, 0, 3, 1],
                '[2, 1, 0, 3]': [2, 1, 0, 3],
                '[2, 1, 3, 0]': [2, 1, 3, 0],
                '[2, 3, 0, 1]': [2, 3, 0, 1],
                '[2, 3, 1, 0]': [2, 3, 1, 0],
                '[3, 0, 1, 2]': [3, 0, 1, 2],
                '[3, 0, 2, 1]': [3, 0, 2, 1],
                '[3, 1, 0, 2]': [3, 1, 0, 2],
                '[3, 1, 2, 0]': [3, 1, 2, 0],
                '[3, 2, 0, 1]': [3, 2, 0, 1],
                '[3, 2, 1, 0]': [3, 2, 1, 0]
                }

        self.inputArrayOrder = [0, 3, 1, 2]
        self.displayArrayOrder = [3, 0, 1, 2]

        #array savepath
        self.savePath = ''
        return

    def startVolumeSlider(self, A=[], keepWindow=False):
        if A == []:
            #copy selected window
            self.A =  np.array(deepcopy(g.win.image),dtype=self.dataType)
            if keepWindow == False:
                g.win.close()
            
        else:
            #load numpy array
            self.B = A
            self.nFrames, self.nVols, self.x, self.y = self.B.shape
            self.dialogbox = Form2()
            self.viewer = SliceViewer(self.B)            
            return
            
        self.B = []
        #get shape
        self.nFrames, self.x, self.y = self.A.shape
        self.framesPerVol = int(self.nFrames/self.nVols)
        #setup display window
        self.displayWindow = Window(self.A,'Volume Slider Window')
        #open gui
        self.dialogbox = Form2()
        self.dialogbox.show()        
        return

    def updateDisplay_volumeSizeChange(self):
        #remove final volume if it dosen't contain the full number of frames
        numberFramesToRemove = self.nFrames%self.getFramesPerVol()
        if numberFramesToRemove != 0:
            self.B = self.A[:-numberFramesToRemove,:,:]
        else:
            self.B = self.A

        self.nFrames, self.x, self.y = self.B.shape

        #reshape to 4D
        self.B = np.reshape(self.B, (self.getFramesPerVol(),self.getNVols(),self.x,self.y), order='F')
        self.displayWindow.imageview.setImage(self.B[0],autoLevels=False)
        return

    def updateDisplay_sliceNumberChange(self, index):
        displayIndex = self.displayWindow.imageview.currentIndex
        self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)
        self.displayWindow.imageview.setCurrentIndex(displayIndex)
        return

    def getNFrames(self):
        return self.nFrames

    def getNVols(self):
        return self.nVols

    def getFramesPerVol(self):
        return self.framesPerVol

    def updateVolsandFramesPerVol(self, nVols, framesPerVol):
        self.nVols = nVols
        self.framesPerVol = framesPerVol

    def closeEvent(self, event):
        event.accept()

    def getArrayShape(self):
        if self.B == []:
            return self.A.shape
        return self.B.shape

    def subtractBaseline(self):
        index = self.displayWindow.imageview.currentIndex
        baseline = self.dialogbox.getBaseline()
        if self.B == []:
            print('first set number of frames per volume')
            return
        else:
            self.B = self.B - baseline
            self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)

    def averageByVol(self):
        index = self.displayWindow.imageview.currentIndex
        #TODO
        return

    def ratioDFF0(self):
        index = self.displayWindow.imageview.currentIndex
        ratioStart, ratioEnd = self.dialogbox.getF0()
        if ratioStart >= ratioEnd:
            print('invalid F0 selection')
            return

        ratioVol = self.B[:,ratioStart:ratioEnd,:,]
        ratioVol = np.mean(ratioVol, axis=1,keepdims=True)

        self.B = np.divide(self.B, ratioVol, dtype=self.dataType)
        self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)
        return

    def exportToWindow(self):
        Window(np.reshape(self.B, (self.nFrames, self.x, self.y), order='F'))
        return

    def exportArray(self, vol='None'):
        if vol == 'None':
            np.save(self.savePath, self.B)
        else:
            f,v,x,y = self.B.shape
            np.save(self.savePath, self.B[:,vol,:,:].reshape((f,1,x,y)))
        return

    def getMaxPixel(self):
        if self.B == []:
            return np.max(self.A)
        else:
            return np.max(self.B)

    def setDType(self, newDataType):
        index = self.displayWindow.imageview.currentIndex
        if self.B == []:
            print('first set number of frames per volume')
            return
        else:
            self.dataType = self.dTypeDict[newDataType]
            self.B = self.B.astype(self.dataType)
            self.dialogbox.dataTypeText.setText(self.getDataType())
            self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)
        return

    def getDataType(self):
        return str(self.dataType).split(".")[-1].split("'")[0]

    def getArrayKeys(self):
        return list(self.arrayDict.keys())

    def getInputArrayOrder(self):
        return self.inputArrayOrder

    def getDisplayArrayOrder(self):
        return self.displayArrayOrder

    def setInputArrayOrder(self, value):
        self.inputArrayOrder = self.arrayDict[value]
        return

    def setDisplayArrayOrder(self, value):
        self.displayArrayOrder = self.arrayDict[value]
        return

    def multiplyByFactor(self, factor):
        index = self.displayWindow.imageview.currentIndex
        if self.B == []:
            print('first set number of frames per volume')
            return
        else:
            self.B = self.B * float(factor)
            print(self.B.shape)
            self.displayWindow.imageview.setImage(self.B[index],autoLevels=False)
        return

    def startViewer(self):
        self.viewer = SliceViewer(self.B)
        return

    def closeViewer(self):
        self.viewer.close()
        return

camVolumeSlider = CamVolumeSlider()





#########################################################################################
#############                  volumeViewer GUI setup            ########################
#########################################################################################
class Form2(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(Form2, self).__init__(parent)

        self.s = g.settings['volumeSlider']

        self.arraySavePath = camVolumeSlider.savePath
        self.arrayImportPath = "None"

        #window geometry
        self.left = 300
        self.top = 300
        self.width = 600
        self.height = 400

        self.slicesPerVolume = self.s['slicesPerVolume']
        self.baselineValue = self.s['baselineValue']
        self.f0Start = self.s['f0Start']
        self.f0End = self.s['f0End']
        self.multiplicationFactor = self.s['multiplicationFactor']
        self.currentDataType = self.s['currentDataType']
        self.newDataType = self.s['newDataType']
        self.inputArrayOrder = self.s['inputArrayOrder']
        self.displayArrayOrder = self.s['displayArrayOrder'] = 16
        self.theta = self.s['theta']
        self.shiftFactor = self.s['shiftFactor']
        self.trim_last_frame = self.s['trimLastFrame']

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Slice #")
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,camVolumeSlider.getNFrames())
        self.SpinBox1.setValue(0)

        self.spinLabel2 = QtWidgets.QLabel("# of slices per volume: ")
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,camVolumeSlider.getNFrames())
        if self.slicesPerVolume < camVolumeSlider.getNFrames():
            self.SpinBox2.setValue(self.slicesPerVolume)
        else:
            self.SpinBox2.setValue(1)

        self.spinLabel4 = QtWidgets.QLabel("baseline value: ")
        self.SpinBox4 = QtWidgets.QSpinBox()
        self.SpinBox4.setRange(0,camVolumeSlider.getMaxPixel())
        if self.baselineValue < camVolumeSlider.getMaxPixel():
            self.SpinBox4.setValue(self.baselineValue)
        else:
            self.SpinBox4.setValue(0)           

        self.spinLabel6 = QtWidgets.QLabel("F0 start volume: ")
        self.SpinBox6 = QtWidgets.QSpinBox()
        self.SpinBox6.setRange(0,camVolumeSlider.getNVols())
        if self.f0Start < camVolumeSlider.getNVols():
            self.SpinBox6.setValue(self.f0Start)
        else:
            self.SpinBox6.setValue(0)            

        self.spinLabel7 = QtWidgets.QLabel("F0 end volume: ")
        self.SpinBox7 = QtWidgets.QSpinBox()
        self.SpinBox7.setRange(0,camVolumeSlider.getNVols())
        if self.f0End < camVolumeSlider.getNVols():
            self.SpinBox7.setValue(self.f0End)
        else:
            self.SpinBox7.setValue(0)

        self.spinLabel8 = QtWidgets.QLabel("factor to multiply by: ")
        self.SpinBox8 = QtWidgets.QSpinBox()
        self.SpinBox8.setRange(0,10000)
        self.SpinBox8.setValue(self.multiplicationFactor)

        self.spinLabel9 = QtWidgets.QLabel("theta: ")
        self.SpinBox9 = QtWidgets.QSpinBox()
        self.SpinBox9.setRange(0,360)
        self.SpinBox9.setValue(self.theta)

        self.spinLabel10 = QtWidgets.QLabel("shift factor: ")
        self.SpinBox10 = QtWidgets.QSpinBox()
        self.SpinBox10.setRange(0,100)
        self.SpinBox10.setValue(self.shiftFactor)


        #sliders
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(camVolumeSlider.getNFrames())
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)

        #ComboBox
        self.dTypeSelectorBox = QtWidgets.QComboBox()
        self.dTypeSelectorBox.addItems(["float16", "float32", "float64","int8","int16","int32","int64"])
        self.inputArraySelectorBox = QtWidgets.QComboBox()
        self.inputArraySelectorBox.addItems(camVolumeSlider.getArrayKeys())
        self.inputArraySelectorBox.setCurrentIndex(4)
        self.inputArraySelectorBox.currentIndexChanged.connect(self.inputArraySelectionChange)

        self.displayArraySelectorBox = QtWidgets.QComboBox()
        self.displayArraySelectorBox.addItems(camVolumeSlider.getArrayKeys())
        self.displayArraySelectorBox.setCurrentIndex(18)
        self.displayArraySelectorBox.currentIndexChanged.connect(self.displayArraySelectionChange)

        #buttons
        self.button1 = QtWidgets.QPushButton("Autolevel")
        self.button2 = QtWidgets.QPushButton("Set Slices")
        #self.button3 = QtWidgets.QPushButton("Average Volumes")
        self.button4 = QtWidgets.QPushButton("subtract baseline")
        self.button5 = QtWidgets.QPushButton("run DF/F0")
        self.button6 = QtWidgets.QPushButton("export to Window")
        self.button7 = QtWidgets.QPushButton("set data Type")
        self.button8 = QtWidgets.QPushButton("multiply")
        self.button9 = QtWidgets.QPushButton("export to array")        

        self.button12 = QtWidgets.QPushButton("open 3D viewer")
        self.button13 = QtWidgets.QPushButton("close 3D viewer")

        #labels
        self.volumeLabel = QtWidgets.QLabel("# of volumes: ")
        self.volumeText = QtWidgets.QLabel("  ")

        self.shapeLabel = QtWidgets.QLabel("array shape: ")
        self.shapeText = QtWidgets.QLabel(str(camVolumeSlider.getArrayShape()))

        self.dataTypeLabel = QtWidgets.QLabel("current data type: ")
        self.dataTypeText = QtWidgets.QLabel(str(camVolumeSlider.getDataType()))
        self.dataTypeChangeLabel = QtWidgets.QLabel("new data type: ")

        self.inputArrayLabel = QtWidgets.QLabel("input array order: ")
        self.displayArrayLabel = QtWidgets.QLabel("display array order: ")

        self.arraySavePathLabel = QtWidgets.QLabel(str(self.arraySavePath))

        self.trim_last_frameLabel = QtWidgets.QLabel("Trim Last Frame: ")
        self.trim_last_frame_checkbox = CheckBox()
        self.trim_last_frame_checkbox.setChecked(self.trim_last_frame)
        self.trim_last_frame_checkbox.stateChanged.connect(self.trim_last_frameClicked)


        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)
        layout.addWidget(self.SpinBox1, 1, 1)

        layout.addWidget(self.slider1, 2, 0, 2, 5)

        layout.addWidget(self.spinLabel2, 4, 0)
        layout.addWidget(self.SpinBox2, 4, 1)
        layout.addWidget(self.button2, 4, 2)

        layout.addWidget(self.spinLabel4, 6, 0)
        layout.addWidget(self.SpinBox4, 6, 1)
        layout.addWidget(self.button4, 6, 2)

        layout.addWidget(self.spinLabel6, 7, 0)
        layout.addWidget(self.SpinBox6, 7, 1)
        layout.addWidget(self.spinLabel7, 7, 2)
        layout.addWidget(self.SpinBox7, 7, 3)
        layout.addWidget(self.button5, 7, 4)

        layout.addWidget(self.volumeLabel, 8, 0)
        layout.addWidget(self.volumeText, 8, 1)
        layout.addWidget(self.shapeLabel, 9, 0)
        layout.addWidget(self.shapeText, 9, 1)

        layout.addWidget(self.spinLabel8, 10, 0)
        layout.addWidget(self.SpinBox8, 10, 1)
        layout.addWidget(self.button8, 10, 2)

        layout.addWidget(self.dataTypeLabel, 11, 0)
        layout.addWidget(self.dataTypeText, 11, 1)
        layout.addWidget(self.dataTypeChangeLabel, 11, 2)
        layout.addWidget(self.dTypeSelectorBox, 11,3)
        layout.addWidget(self.button7, 11, 4)

        layout.addWidget(self.button6, 13, 0)
        layout.addWidget(self.button1, 13, 4)
        layout.addWidget(self.button6, 13, 0)
        layout.addWidget(self.button1, 13, 4)
        layout.addWidget(self.button9, 14, 0)
        layout.addWidget(self.arraySavePathLabel, 14, 1, 1, 4)      
        
        
        layout.addWidget(self.spinLabel9, 16, 0)
        layout.addWidget(self.SpinBox9, 16, 1)

        layout.addWidget(self.spinLabel10, 17, 0)
        layout.addWidget(self.SpinBox10, 17, 1)
        layout.addWidget(self.trim_last_frameLabel, 18, 0)
        layout.addWidget(self.trim_last_frame_checkbox, 18, 1)

        layout.addWidget(self.inputArrayLabel, 19, 0)
        layout.addWidget(self.inputArraySelectorBox, 19, 1)

        layout.addWidget(self.displayArrayLabel, 19, 2)
        layout.addWidget(self.displayArraySelectorBox, 19, 3)

        layout.addWidget(self.button12, 20, 0)
        layout.addWidget(self.button13, 20, 1)


        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Volume Slider GUI")

        #connect sliders & spinboxes
        self.slider1.valueChanged.connect(self.slider1ValueChange)
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        self.SpinBox9.valueChanged.connect(self.setTheta)
        self.SpinBox10.valueChanged.connect(self.setShiftFactor)

        #connect buttons
        self.button1.clicked.connect(self.autoLevel)
        self.button2.clicked.connect(self.updateVolumeValue)
        #self.button3.clicked.connect(self.averageByVol)
        self.button4.clicked.connect(self.subtractBaseline)
        self.button5.clicked.connect(self.ratioDFF0)
        self.button6.clicked.connect(self.exportToWindow)
        self.button7.clicked.connect(self.dTypeSelectionChange)
        self.button8.clicked.connect(self.multiplyByFactor)
        self.button9.clicked.connect(self.exportArray)            
        self.button12.clicked.connect(self.startViewer)
        self.button13.clicked.connect(self.closeViewer)

        return

     #volume changes with slider & spinbox
    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        return

    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        camVolumeSlider.updateDisplay_sliceNumberChange(value)
        return

    def autoLevel(self):
        camVolumeSlider.displayWindow.imageview.autoLevels()
        return

    def updateVolumeValue(self):
        self.slicesPerVolume = self.SpinBox2.value()
        noVols = int(camVolumeSlider.getNFrames()/self.slicesPerVolume)
        camVolumeSlider.updateVolsandFramesPerVol(noVols, self.slicesPerVolume)
        self.volumeText.setText(str(noVols))

        camVolumeSlider.updateDisplay_volumeSizeChange()
        self.shapeText.setText(str(camVolumeSlider.getArrayShape()))

        if (self.slicesPerVolume)%2 == 0:
            self.SpinBox1.setRange(0,self.slicesPerVolume-1) #if even, display the last volume
            self.slider1.setMaximum(self.slicesPerVolume-1)
        else:
            self.SpinBox1.setRange(0,self.slicesPerVolume-2) #else, don't display the last volume
            self.slider1.setMaximum(self.slicesPerVolume-2)

        self.updateVolSpinBoxes()
        return

    def updateVolSpinBoxes(self):
        #self.SpinBox3.setRange(0,camVolumeSlider.getNVols())

        self.SpinBox6.setRange(0,camVolumeSlider.getNVols())
        self.SpinBox7.setRange(0,camVolumeSlider.getNVols())
        return

    def getBaseline(self):
        return self.SpinBox4.value()

    def getF0(self):
        return self.SpinBox6.value(), self.SpinBox7.value()

    def subtractBaseline(self):
        camVolumeSlider.subtractBaseline()
        return

    def ratioDFF0(self):
        camVolumeSlider.ratioDFF0()
        return

    def exportToWindow(self):
        camVolumeSlider.savePath = self.arraySavePath
        camVolumeSlider.exportToWindow()
        return

    def dTypeSelectionChange(self):
        camVolumeSlider.setDType(self.dTypeSelectorBox.currentText())
        self.dataTypeText = QtWidgets.QLabel(str(camVolumeSlider.getDataType()))
        return

    def multiplyByFactor(self):
        camVolumeSlider.multiplyByFactor(self.SpinBox8.value())
        return

    def exportArray(self):
        self.arraySavePath = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', self.arraySavePath, 'Numpy array (*.npy)')
        self.arraySavePath = str(self.arraySavePath[0])
        camVolumeSlider.savePath = self.arraySavePath
        self.arraySavePathLabel.setText(self.arraySavePath)
        camVolumeSlider.exportArray()
        return

    def startViewer(self):
        self.saveSettings()
        camVolumeSlider.startViewer()
        return

    def closeViewer(self):
        camVolumeSlider.closeViewer()
        return

    def setTheta(self):
        self.theta = self.SpinBox9.value()

    def setShiftFactor(self):
        self.shiftFactor = self.SpinBox10.value()

    def trim_last_frameClicked(self):
        self.trim_last_frame = self.trim_last_frame_checkbox.isChecked()

    def inputArraySelectionChange(self, value):
        camVolumeSlider.setInputArrayOrder(self.inputArraySelectorBox.currentText())
        return

    def displayArraySelectionChange(self, value):
        camVolumeSlider.setDisplayArrayOrder(self.displayArraySelectorBox.currentText())
        return

    def saveSettings(self):
        self.s['theta'] = self.theta
        self.s['slicesPerVolume'] = self.slicesPerVolume
        self.s['baselineValue'] = self.baselineValue
        self.s['f0Start'] = self.f0Start
        self.s['f0End'] = self.f0End
        self.s['multiplicationFactor'] = self.multiplicationFactor
        self.s['currentDataType'] = self.currentDataType
        self.s['newDataType'] = self.newDataType
        self.s['shiftFactor'] = self.shiftFactor
        self.s['trimLastFrame'] = self.trim_last_frame        
        
        g.settings['volumeSlider'] = self.s
        
        return


    def close(self):
        self.saveSettings()
        camVolumeSlider.closeViewer()
        camVolumeSlider.displayWindow.close()
        camVolumeSlider.dialogbox.destroy()
        camVolumeSlider.end()
        self.closeAllWindows()
        return


#########################################################################################
#############            3D Matlibplot scatter plot            ##########################
#########################################################################################
class plot3D_options(QtWidgets.QDialog):
    def __init__(self, prob, threshold, parent = None):
        super(plot3D_options, self).__init__(parent)

        self.prob = prob
        self.threshold = threshold

        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 200

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Amount of downsampling (0-1)")
        self.SpinBox1 = QtWidgets.QDoubleSpinBox()
        self.SpinBox1.setDecimals(4)
        self.SpinBox1.setRange(0,1.0000)
        self.SpinBox1.setValue(self.prob)

        self.spinLabel2 = QtWidgets.QLabel("Threshold level")
        self.SpinBox2 = QtWidgets.QSpinBox()
        self.SpinBox2.setRange(0,10000)
        self.SpinBox2.setValue(self.threshold)

        #checkboxes
        self.checkBox1_label = QtWidgets.QLabel("Display array outline")
        self.checkBox1 = QtWidgets.QCheckBox()
        self.checkBox2_label = QtWidgets.QLabel("Display plot axis")
        self.checkBox2 = QtWidgets.QCheckBox()
        #buttons
        #self.button1 = QtWidgets.QPushButton("Close")

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)

        layout.addWidget(self.spinLabel1, 1, 0)
        layout.addWidget(self.SpinBox1, 1, 1)
        layout.addWidget(self.spinLabel2, 2, 0)
        layout.addWidget(self.SpinBox2, 2, 1)
        layout.addWidget(self.checkBox1_label, 3, 0)
        layout.addWidget(self.checkBox1, 3, 1)
        layout.addWidget(self.checkBox2_label, 4, 0)
        layout.addWidget(self.checkBox2, 4, 1)
        #layout.addWidget(self.button1, 5, 2)

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("3D Plot Options")

        #connect sliders & spinboxes
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        self.SpinBox2.valueChanged.connect(self.spinBox2ValueChange)

        #connect checkboxes
        self.checkBox1.stateChanged.connect(self.checkBox1ValueChange)
        self.checkBox2.stateChanged.connect(self.checkBox2ValueChange)

        #connect buttons
        #self.button1.clicked.connect(self.close)

        return

    def spinBox1ValueChange(self, value):
        camVolumeSlider.viewer.setProb(value)
        return

    def spinBox2ValueChange(self, value):
        camVolumeSlider.viewer.setThreshold(value)
        return

    def checkBox1ValueChange(self, value):
        camVolumeSlider.viewer.setPlotCube(value)
        return

    def checkBox2ValueChange(self, value):
        camVolumeSlider.viewer.setPlotAxis((not value))
        return


class exportDialog_win(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(exportDialog_win, self).__init__(parent)

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
        self.z_displayWindow = Window(camVolumeSlider.viewer.getZWin(),'Z view')
        return

    def exportX(self):
        self.x_displayWindow = Window(camVolumeSlider.viewer.getXWin(),'X view')
        return

    def exportY(self):
        self.y_displayWindow = Window(camVolumeSlider.viewer.getYWin(),'Y view')
        return

class OverlayOptions(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(OverlayOptions, self).__init__(parent)

        #window geometry
        self.left = 300
        self.top = 300
        self.width = 300
        self.height = 200

        self.opacity = 50
        self.lut = 'grey'
        self.overlay1 = None
        self.overlay2 = None
        self.overlay3 = None
        self.overlay4 = None        

        #buttons
        self.linkOverlayButton = QtWidgets.QPushButton("Set All Overlays") 
        self.linkOverlayButton.clicked.connect(self.setOverlays)

        self.transferButton1 = QtWidgets.QPushButton("Transfer Top Main") 
        self.transferButton1.clicked.connect(self.transfer1)        
        
        self.transferButton2 = QtWidgets.QPushButton("Transfer Top Overlay") 
        self.transferButton2.clicked.connect(self.transfer2)   


        #combo boxes
        self.cmSelectorBox = QtWidgets.QComboBox()
        #self.cmSelectorBox.addItems(["inferno", "magma", "plasma","viridis","Reds","Greens","Blues", "binary","bone","Greys",
        #                             "hot","Set1","RdBu","Accent","autumn","jet","hsv","Spectral"])
        self.cmSelectorBox.addItems(['grey','thermal','flame','yellowy','bipolar','spectrum','cyclic','greyclip'])
        self.cmSelectorBox.setCurrentIndex(0)
        self.cmSelectorBox.currentIndexChanged.connect(self.setColorMap)

        self.modeSelectorBox = QtWidgets.QComboBox()
        self.modeSelectorBox.addItems(["Source Overlay", "Overlay", "Plus", "Mutiply"])
        self.modeSelectorBox.setCurrentIndex(0)
        self.modeSelectorBox.currentIndexChanged.connect(self.setMode)

        #sliders
        self.sliderOpacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliderOpacity.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.sliderOpacity.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.sliderOpacity.setMinimum(0)
        self.sliderOpacity.setMaximum(100)
        self.sliderOpacity.setTickInterval(10)
        self.sliderOpacity.setSingleStep(1)
        self.sliderOpacity.setValue(self.opacity)
        self.sliderOpacity.valueChanged.connect(self.setOpacity)

        #checkboxes
        self.linkCheck = QtWidgets.QCheckBox()
        self.linkCheck.setChecked(True)
        self.linkCheck.stateChanged.connect(self.linkCheckValueChange)
        
        #labels
        self.labelCM = QtWidgets.QLabel("Color Map Presets")
        self.labelOverlayMode = QtWidgets.QLabel("Overlay Mode")
        self.labelOpacity = QtWidgets.QLabel("Opacity (%)")
        self.labelLinkCheck = QtWidgets.QLabel("Link Histogram Sliders")         
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.labelCM, 1, 0)
        layout.addWidget(self.cmSelectorBox, 1, 1)
        layout.addWidget(self.linkOverlayButton, 1, 2)  
        
        layout.addWidget(self.transferButton1, 3, 0)  
        layout.addWidget(self.transferButton2, 3, 1)          
        
        layout.addWidget(self.labelOverlayMode, 5, 0)
        layout.addWidget(self.modeSelectorBox, 5, 1)
        layout.addWidget(self.labelOpacity, 6, 0)
        layout.addWidget(self.sliderOpacity, 6, 1,1,4)  
        layout.addWidget(self.labelLinkCheck, 7, 0)    
        layout.addWidget(self.linkCheck, 7, 1)          

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Overlay Options")


        return

    def setColorMap(self):
        #camVolumeSlider.viewer.useOverlayLUT = True
        #camVolumeSlider.viewer.linkOverlays = True
        self.lut = self.cmSelectorBox.currentText()
        #camVolumeSlider.viewer.setOverlayLUT(lut)
        #camVolumeSlider.viewer.gradientPreset = lut
        #camVolumeSlider.viewer.updateAllOverlayWins()
        return

    def setOverlays(self):     
        camVolumeSlider.viewer.gradientPreset = self.lut
        camVolumeSlider.viewer.usePreset = True
        camVolumeSlider.viewer.updateAllOverlayWins()
        camVolumeSlider.viewer.usePreset = False
        return

    def transfer1(self):
        lut = camVolumeSlider.viewer.imv1.getHistogramWidget().item.gradient.saveState()
        levels = camVolumeSlider.viewer.imv1.getHistogramWidget().item.getLevels()
        #set lut
        camVolumeSlider.viewer.imv2.getHistogramWidget().item.gradient.restoreState(lut)
        camVolumeSlider.viewer.imv3.getHistogramWidget().item.gradient.restoreState(lut)
        camVolumeSlider.viewer.imv4.getHistogramWidget().item.gradient.restoreState(lut)
        camVolumeSlider.viewer.imv6.getHistogramWidget().item.gradient.restoreState(lut)    
        #set levels
        camVolumeSlider.viewer.imv2.getHistogramWidget().item.setLevels(levels[0],levels[1])
        camVolumeSlider.viewer.imv3.getHistogramWidget().item.setLevels(levels[0],levels[1])
        camVolumeSlider.viewer.imv4.getHistogramWidget().item.setLevels(levels[0],levels[1])
        camVolumeSlider.viewer.imv6.getHistogramWidget().item.setLevels(levels[0],levels[1])           
        return

    def transfer2(self):
        camVolumeSlider.viewer.sharedState = camVolumeSlider.viewer.bgItem_imv1.hist_luttt.item.gradient.saveState()
        camVolumeSlider.viewer.sharedLevels = camVolumeSlider.viewer.bgItem_imv1.hist_luttt.item.getLevels()
        camVolumeSlider.viewer.useSharedState = True
        camVolumeSlider.viewer.updateAllOverlayWins()
        camVolumeSlider.viewer.useSharedState = False
        return

    
    def setMode(self):
        camVolumeSlider.viewer.setOverlayMode(self.modeSelectorBox.currentText())
        camVolumeSlider.viewer.updateAllOverlayWins()
        return
    
    def setOpacity(self,value):
        self.opacity = value
        camVolumeSlider.viewer.OverlayOPACITY = (value/100)
        camVolumeSlider.viewer.updateAllOverlayWins()
        return

    def linkCheckValueChange(self, value):
        camVolumeSlider.viewer.histogramsLinked = value
        return
    
#########################################################################################
#############            3D Texture plot            #####################################
#########################################################################################
class plotTexture(QtWidgets.QDialog):
    def __init__(self, data, parent = None):
        super(plotTexture, self).__init__(parent)

        self.data = data

        self.w = gl.GLViewWidget()
        self.w.opts['distance'] = 200
        self.w.setWindowTitle('3D slice - texture plot')

        self.shape = self.data.shape

        #set intensity
        self.levels = (0, 1000)

        ## slice out three center planes, convert to RGBA for OpenGL texture
        self.slice1 = int(self.shape[0]/2)
        self.slice2 = int(self.shape[1]/2)
        self.slice3 = int(self.shape[2]/2)

        ## Create three image items from textures, add to view
        self.updateYZ()
        self.updateXZ()
        self.updateXY()
        self.addBorder(self.shape[0],self.shape[1],self.shape[2])


    def updateYZ(self):
        try:
            self.w.removeItem(self.v1)
        except:
            pass
        tex1 = pg.makeRGBA(self.data[self.slice1], levels=self.levels)[0]       # yz plane
        self.v1 = gl.GLImageItem(tex1)
        self.v1.translate(-self.slice2, -self.slice3, 0)
        self.v1.rotate(90, 0,0,1)
        self.v1.rotate(-90, 0,1,0)
        self.w.addItem(self.v1)
        return

    def updateXZ(self):
        try:
            self.w.removeItem(self.v2)
        except:
            pass
        tex2 = pg.makeRGBA(self.data[:,self.slice2], levels=self.levels)[0]     # xz plane
        self.v2 = gl.GLImageItem(tex2)
        self.v2.translate(-self.slice1, -self.slice3, 0)
        self.v2.rotate(-90, 1,0,0)
        self.w.addItem(self.v2)
        return

    def updateXY(self):
        try:
            self.w.removeItem(self.v3)
        except:
            pass
        tex3 = pg.makeRGBA(self.data[:,:,self.slice3], levels=self.levels)[0]   # xy plane
        self.v3 = gl.GLImageItem(tex3)
        self.v3.translate(-self.slice1, -self.slice2, 0)
        self.w.addItem(self.v3)
        return

    def addBorder(self,x,y,z):
        self.ax = GLBorderItem()
        self.ax.setSize(x=x,y=y,z=z)
        self.w.addItem(self.ax)

    def getSliceValues(self):
        return [self.slice1,self.slice2,self.slice3]

    def setSliceValues(self,slice1,slice2,slice3):
        self.slice1 = slice1
        self.slice2 = slice2
        self.slice3 = slice3
        return

    def getShape(self):
        return self.shape

class textureDialog_win(QtWidgets.QDialog):
    def __init__(self, shape, X,Y,Z, parent = None):
        super(textureDialog_win, self).__init__(parent)

        #window geometry
        self.left = 300
        self.top = 300
        self.width = 500
        self.height = 200

        #sliders
        self.sliderX = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliderX.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.sliderX.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.sliderX.setMinimum(0)
        self.sliderX.setMaximum(shape[0])
        self.sliderX.setTickInterval(1)
        self.sliderX.setSingleStep(1)
        self.sliderX.setValue(X)

        self.sliderY = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliderY.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.sliderY.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.sliderY.setMinimum(0)
        self.sliderY.setMaximum(shape[1])
        self.sliderY.setTickInterval(1)
        self.sliderY.setSingleStep(1)
        self.sliderY.setValue(Y)

        self.sliderZ = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliderZ.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.sliderZ.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.sliderZ.setMinimum(0)
        self.sliderZ.setMaximum(shape[2])
        self.sliderZ.setTickInterval(1)
        self.sliderZ.setSingleStep(1)
        self.sliderZ.setValue(Z)

        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(5)
        layout.addWidget(self.sliderX, 0, 0)
        layout.addWidget(self.sliderY, 1, 0)
        layout.addWidget(self.sliderZ, 2, 0)

        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #add window title
        self.setWindowTitle("Texture Control")

        #connect sliders
        self.sliderX.valueChanged.connect(self.sliderXValueChange)
        self.sliderY.valueChanged.connect(self.sliderYValueChange)
        self.sliderZ.valueChanged.connect(self.sliderZValueChange)
        return

    def sliderXValueChange(self, value):
        camVolumeSlider.viewer.texturePlot.slice1 = value
        camVolumeSlider.viewer.texturePlot.updateYZ()
        return

    def sliderYValueChange(self, value):
        camVolumeSlider.viewer.texturePlot.slice2 = value
        camVolumeSlider.viewer.texturePlot.updateXZ()
        return

    def sliderZValueChange(self, value):
        camVolumeSlider.viewer.texturePlot.slice3 = value
        camVolumeSlider.viewer.texturePlot.updateXY()
        return


#########################################################################################
#############          FLIKA Base Menu             #####################################
#########################################################################################
class VolumeSliderBase(BaseProcess_noPriorWindow):
    """
    Start Volume Slider from differnt sources

        |Select source (current window or saved numpy array)
        
    Returns volumeSlider GUI

    """
    
    def __init__(self):
        if g.settings['volumeSlider'] is None or 'displayArrayOrder' not in g.settings['volumeSlider']:
            s = dict() 
            s['inputChoice'] = 'Current Window'              
            s['keepOriginalWindow'] = False   
            s['slicesPerVolume'] =    1
            s['baselineValue'] = 0
            s['f0Start'] = 0
            s['f0End'] = 0
            s['multiplicationFactor'] = 100
            s['currentDataType'] = 0
            s['newDataType'] = 0
            s['theta'] = 45
            s['shiftFactor'] = 1
            s['trimLastFrame'] = False
            s['inputArrayOrder'] = 4
            s['displayArrayOrder'] = 16
                            
            g.settings['volumeSlider'] = s
                
        BaseProcess_noPriorWindow.__init__(self)
        
    def __call__(self, inputChoice,keepOriginalWindow,keepSourceWindow=False):
        g.settings['volumeSlider']['inputChoice'] = inputChoice
        g.settings['volumeSlider']['keepOriginalWindow'] = keepOriginalWindow

        g.m.statusBar().showMessage("Starting Volume Slider...")
        
        if inputChoice == 'Current Window':
            camVolumeSlider.startVolumeSlider(keepWindow=keepOriginalWindow)
            
        elif inputChoice == 'Numpy Array':
            A_path = open_file_gui(directory=os.path.expanduser("~/Desktop"),filetypes='*.npy')
            g.m.statusBar().showMessage("Importing Array: " + A_path)
            A = np.load(str(A_path))
            camVolumeSlider.startVolumeSlider(A=A,keepWindow=keepOriginalWindow)
            
        return

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)

    def gui(self):
        self.gui_reset()
                       
        #combobox
        inputChoice = ComboBox()
        inputChoice.addItem('Current Window')
        inputChoice.addItem('Numpy Array')
        
        #checkbox
        self.keepOriginalWindow = CheckBox()
        self.keepOriginalWindow.setValue(False)          
        
        #populate GUI
        self.items.append({'name': 'inputChoice', 'string': 'Choose Input Data:', 'object': inputChoice}) 
        self.items.append({'name': 'keepOriginalWindow','string':'Keep Original Window','object': self.keepOriginalWindow})                                     
        super().gui()
        
        
volumeSliderBase = VolumeSliderBase()