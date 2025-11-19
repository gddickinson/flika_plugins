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
from time import time
import os
from os import listdir
from os.path import expanduser, isfile, join
from distutils.version import StrictVersion
import re

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow


class TiffSelector(QtWidgets.QWidget):
    """
    This widget is a button with a label.  Once you click the button, the widget waits for you to select a folder.  Once you do, it sets self.folder and it sets the label.
    """
    valueChanged=QtCore.Signal()
    def __init__(self,filetypes='*.*'):
        QtWidgets.QWidget.__init__(self)
        self.button=QtWidgets.QPushButton('Select File')
        self.label=QtWidgets.QLabel('None')
        self.window=None
        self.layout=QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.button.clicked.connect(self.buttonclicked)
        self.filetypes = filetypes
        self.filePath = ''
        
    def buttonclicked(self):
        prompt = 'testing TiffSelector'
        self.filePath, _ = QtWidgets.QFileDialog.getOpenFileName(g.m, "Select file", expanduser("~"), self.filetypes,)
        
        self.label.setText('...'+os.path.split(self.filePath)[-1][-20:])
        self.valueChanged.emit()

    def value(self):
        return self.filePath

    def setValue(self, filePath):
        self.filePath = str(filePath)
        self.label.setText('...' + os.path.split(self.filePath)[-1][-20:])    
    
    
    
class TiffPageLoader(BaseProcess_noPriorWindow):
    """
    Loads tiff files by page.
    """
    def __init__(self):
        if g.settings['tiff_page_loader'] is None or 'setSteps' not in g.settings['tiff_page_loader']:
            s = dict()
            s['nSteps'] = 1
            s['setSteps'] = True
            g.settings['tiff_page_loader'] = s
            
        #test saved filePath is valid
        
        super().__init__()
        return             
    
    def __call__(self, filePath, nSteps, setSteps):
        g.settings['on_the_fly']['filePath'] = filePath
        g.settings['on_the_fly']['nSteps']=nSteps
        g.settings['on_the_fly']['setSteps']=setSteps
        g.m.statusBar().showMessage("Scanning file ...")
        
        self.mainGUI = MainGUI()
        self.mainGUI.show()

    def closeEvent(self, event):
        BaseProcess_noPriorWindow.closeEvent(self, event)
        
    def gui(self):
        s=g.settings['tiff_page_loader']
        self.gui_reset()
        self.nSteps = pg.SpinBox(int=True, step=1)
        self.nSteps.setMinimum(1)
        self.nSteps.setValue(s['nSteps'])
        self.filePath = TiffSelector('*.tiff, *.tif')
        
        self.setSteps = CheckBox()
        self.setSteps.setValue(s['setSteps'])

        self.items.append({'name':'filePath','string':'Tiff file path','object': self.filePath})
        self.items.append({'name': 'nSteps', 'string': 'Number of steps per volume', 'object': self.nSteps})
        self.items.append({'name': 'setSteps', 'string': 'Automatically determine step size from Tiff metadata', 'object': self.setSteps})

        super().gui()
 
tiffPageLoader = TiffPageLoader()  


class MainGUI(QtWidgets.QDialog):
    def __init__(self, parent = None):
        super(MainGUI, self).__init__(parent)
        g.m.statusBar().showMessage("")
        #get paramters
        s=g.settings['on_the_fly']
        self.fileName = s['filePath']
        #self.fileName = "C:\\Users\\George\\Desktop\\testRun\\george_2color\\vol_0\\vol_0.tif"
        
        #load tiff metadata
        self.shape, self.dtype, self.axes, self.nPages = self.getTiffInfo()
        
        #set step size
        self.setSteps = s['setSteps']
        self.nSteps = s['nSteps']
        if self.setSteps == True: #user selection
            if self.axes == 'TZCYX':
            #multi-sweep (micromanager)
                self.nSteps = self.shape[0]
            if self.axes == 'ZCYX':
            #single-sweep (micromanager)
                self.nSteps = self.shape[0]
        
        #window geometry
        self.left = 300
        self.top = 300
        self.width = 600
        self.height = 250

        #spinboxes
        self.spinLabel1 = QtWidgets.QLabel("Slice #") 
        self.SpinBox1 = QtWidgets.QSpinBox()
        self.SpinBox1.setRange(0,self.nSteps)
        self.SpinBox1.setValue(0)

       
        #sliders
        self.sliderLabel1 = QtWidgets.QLabel("Slice #")
        self.slider1 = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider1.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.slider1.setTickPosition(QtWidgets.QSlider.TicksBothSides)
        self.slider1.setMinimum(0)
        self.slider1.setMaximum(self.nSteps)
        self.slider1.setTickInterval(1)
        self.slider1.setSingleStep(1)
        
        #buttons
        self.button1 = QtWidgets.QPushButton("Autolevel") 
        self.button2 = QtWidgets.QPushButton("Set Slice")         
        self.button3 = QtWidgets.QPushButton("Overlay")   
        self.button4 = QtWidgets.QPushButton("Load Tiff")  

        #labels
        self.filePathLabel = QtWidgets.QLabel("File: ")
        self.filePathText = QtWidgets.QLabel(str(self.fileName).split('/')[-1])         
        self.shapeLabel = QtWidgets.QLabel("Shape: ")
        self.shapeText = QtWidgets.QLabel(str(self.shape))       
        self.dtypeLabel = QtWidgets.QLabel("Data Type: ")
        self.dtypeText = QtWidgets.QLabel(str(self.dtype))     
        self.axesLabel = QtWidgets.QLabel("Axes: ")
        self.axesText = QtWidgets.QLabel(str(self.axes))       
        self.nPagesLabel = QtWidgets.QLabel("# of Pages: ")
        self.nPagesText = QtWidgets.QLabel(str(self.nPages))      
        self.nVolLabel = QtWidgets.QLabel("# of Volumes: ")
        self.nVolText = QtWidgets.QLabel(str(int(self.nPages/self.nSteps)))        
        
        #grid layout
        layout = QtWidgets.QGridLayout()
        layout.setSpacing(10)

        layout.addWidget(self.spinLabel1, 1, 0)        
        layout.addWidget(self.SpinBox1, 1, 1)
        layout.addWidget(self.button4, 1, 4) 
        layout.addWidget(self.slider1, 2, 0, 2, 5)
        layout.addWidget(self.button2, 4, 0)        
        layout.addWidget(self.button3, 4, 3)      
        layout.addWidget(self.button1, 4, 4) 
        layout.addWidget(self.shapeLabel, 5, 0)         
        layout.addWidget(self.shapeText, 5, 1)  
        layout.addWidget(self.dtypeLabel, 6, 0)         
        layout.addWidget(self.dtypeText, 6, 1)         
        layout.addWidget(self.axesLabel, 7, 0)         
        layout.addWidget(self.axesText, 7, 1)  
        layout.addWidget(self.nPagesLabel, 8, 0)         
        layout.addWidget(self.nPagesText, 8, 1) 
        layout.addWidget(self.nVolLabel, 9, 0)         
        layout.addWidget(self.nVolText, 9, 1)  
        layout.addWidget(self.filePathLabel, 10, 0)         
        layout.addWidget(self.filePathText, 10, 1)        
        
 
        self.setLayout(layout)
        self.setGeometry(self.left, self.top, self.width, self.height)
        
        #add window title
        self.setWindowTitle("Volume Slider GUI")
        
        #connect sliders & spinboxes
        self.slider1.valueChanged.connect(self.slider1ValueChange)
        self.SpinBox1.valueChanged.connect(self.spinBox1ValueChange)
        
        #connect buttons
        #self.button1.clicked.connect(self.autoLevel)
        #self.button2.clicked.connect(self.updateVolumeValue)
        #self.button3.clicked.connect(self.overlay)
        self.button4.clicked.connect(self.loadTiff_Display)
        
        
        return
 
     #volume changes with slider & spinbox
    def slider1ValueChange(self, value):
        self.SpinBox1.setValue(value)
        return
            
    def spinBox1ValueChange(self, value):
        self.slider1.setValue(value)
        #camVolumeSlider2.updateDisplay_sliceNumberChange(value)
        return       

    def getTiffInfo(self):
        with tifffile.TiffFile(self.fileName) as tif:
            info = tif.info()

        shape, dtype, axes, numberPages = re.search('Series (.*) pages,', info).group(1).replace(" ","").split(':')[-1].split(',')
        shape = shape.split('x')
        shape = list(map(int, shape))
        numberPages = int(numberPages)
        return shape, dtype, axes, numberPages


    def loadTiff_Display(self):
        t = time() 
        g.m.statusBar().showMessage("Loading movie...")
        
        if self.axes == 'IYX':
            #single channel (stk?)
            startPage = 0
            endPage = self.nPages
            stepSize = 1

            volume = 0

            series = list(range(startPage+volume, endPage+volume, stepSize))

            #for stk? file (axes == 'IYX')
            Page_series = tifffile.imread(self.fileName,key=series)
            print(Page_series.shape)
            self.displayWindow_0 = Window(Page_series, 'Loaded from pages - channel 0')
        
        
        
        elif self.axes == 'TZCYX':
            #multi-sweep (micromanager)
            volNum = shape[1]

            #2-channel - get channels serperately 
            #for micromanager file (axes == 'TZCYX')
            with tifffile.TiffFile(self.fileName) as tif:
                initiated_0 = False
                initiated_1 = False
                for page in tif.pages:
                    #print('ChannelIndex',page.tags.micromanager_metadata.value.ChannelIndex)
                    #print('ImageNumber',page.tags.micromanager_metadata.value.ImageNumber)
                    g.m.statusBar().showMessage("Loading movie...(Image #:{})".format(page.tags.micromanager_metadata.value.ImageNumber))
                    if page.tags.micromanager_metadata.value.ChannelIndex == 0 and int(page.tags.micromanager_metadata.value.ImageNumber)%volNum == 0:
                        image_0 = page.asarray()
                        if initiated_0 == False:
                            stack_0 = image_0
                            initiated_0 = True
                        else:
                            stack_0 = np.dstack((stack_0, image_0))

                    if page.tags.micromanager_metadata.value.ChannelIndex == 1 and int(page.tags.micromanager_metadata.value.ImageNumber)%volNum == 0:
                        image_1 = page.asarray()
                        if initiated_1 == False:
                            stack_1 = image_1
                            initiated_1 = True
                        else:
                            stack_1 = np.dstack((stack_1, image_1))



            print(stack_0.shape)
            print(stack_1.shape)
            stackAxes = ['y','x','t']
            target_axes = ['t', 'y', 'x']	
            perm = get_permutation_tuple(stackAxes, target_axes)
            stack_0 = np.transpose(stack_0, perm)
            stack_1 = np.transpose(stack_1, perm)

            stack_0.shape
            stack_1.shape
            
            self.displayWindow_0 = Window(stack_0,'Loaded Page series - channel 0')
            self.displayWindow_1 = Window(stack_1,'Loaded Page series - channel 1')
        
        elif self.axes == 'ZCYX':
        #single-sweep (micromanager)
        #2-channel 
        
            with tifffile.TiffFile(self.fileName) as tif:
                axes = [tifffile.AXES_LABELS[ax] for ax in tif.series[0].axes]
                print(axes)
                initiated_0 = False
                initiated_1 = False
                for page in tif.pages:
                    ##print('ChannelIndex',page.tags.micromanager_metadata.value.ChannelIndex)
                    ##print('ImageNumber',page.tags.micromanager_metadata.value.ImageNumber)
                    g.m.statusBar().showMessage("Loading movie...(Image #:{})".format(page.tags.micromanager_metadata.value.ImageNumber))
                    if page.tags.micromanager_metadata.value.ChannelIndex == 0:
                        image_0 = page.asarray()
                        if initiated_0 == False:
                            stack_0 = image_0
                            initiated_0 = True
                        else:
                            stack_0 = np.dstack((stack_0, image_0))

                    if page.tags.micromanager_metadata.value.ChannelIndex == 1:
                        image_1 = page.asarray()
                        if initiated_1 == False:
                            stack_1 = image_1
                            initiated_1 = True
                        else:
                            stack_1 = np.dstack((stack_1, image_1))

            tif.close()                  
            print(stack_0.shape)
            print(stack_1.shape)
            stackAxes = ['y','x','t']
            target_axes = ['t', 'y', 'x']	
            perm = get_permutation_tuple(stackAxes, target_axes)
            stack_0 = np.transpose(stack_0, perm)
            stack_1 = np.transpose(stack_1, perm)

            stack_0.shape
            stack_1.shape
            self.displayWindow_0 = Window(stack_0,'Loaded from pages - channel 0')
            self.displayWindow_1 = Window(stack_1,'Loaded from pages - channel 1')
 
        elif self.axes == 'TZYX':
        #single-sweep (micromanager)
        #2-channel 
        
            with tifffile.TiffFile(self.fileName) as tif:
                axes = [tifffile.AXES_LABELS[ax] for ax in tif.series[0].axes]
                print(axes)
                initiated_0 = False
                initiated_1 = False
                print('ok')
                for page in tif.pages:
                    ##print('ChannelIndex',page.tags.micromanager_metadata.value.ChannelIndex)
                    print('ImageNumber',page.tags.micromanager_metadata.value.ImageNumber)
                    g.m.statusBar().showMessage("Loading movie...(Image #:{})".format(page.tags.micromanager_metadata.value.ImageNumber))
                    if page.tags.micromanager_metadata.value.ChannelIndex == 0:
                        image_0 = page.asarray()
                        if initiated_0 == False:
                            stack_0 = image_0
                            initiated_0 = True
                        else:
                            stack_0 = np.dstack((stack_0, image_0))

                    if page.tags.micromanager_metadata.value.ChannelIndex == 1:
                        image_1 = page.asarray()
                        if initiated_1 == False:
                            stack_1 = image_1
                            initiated_1 = True
                        else:
                            stack_1 = np.dstack((stack_1, image_1))

            tif.close()                  
            print(stack_0.shape)
            print(stack_1.shape)
            stackAxes = ['y','x','t']
            target_axes = ['t', 'y', 'x']	
            perm = get_permutation_tuple(stackAxes, target_axes)
            stack_0 = np.transpose(stack_0, perm)
            stack_1 = np.transpose(stack_1, perm)

            stack_0.shape
            stack_1.shape
            self.displayWindow_0 = Window(stack_0,'Loaded from pages - channel 0')
            self.displayWindow_1 = Window(stack_1,'Loaded from pages - channel 1') 
        
        else:
            g.m.statusBar().showMessage("No Movie Generated")
            return
        
        g.m.statusBar().showMessage("Successfully generated movie ({} s)".format(round(time() - t)))
        return
        
        
        
    def autoLevel(self):
        #camVolumeSlider2.displayWindow.imageview.autoLevels()
        #if camVolumeSlider2.nChannels == 2:
        #    camVolumeSlider2.displayWindow_2.imageview.autoLevels()
        return 
    
        
    def overlay(self):
        #camVolumeSlider2.overlay()
        return

mainGUI = MainGUI()