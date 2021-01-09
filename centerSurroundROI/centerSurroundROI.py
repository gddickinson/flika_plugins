from qtpy.QtCore import *
from qtpy.QtGui import *
from qtpy.QtWidgets import *
import numpy as np
from scipy.ndimage.interpolation import shift
from flika.window import Window
import flika.global_vars as g
import pyqtgraph as pg
from time import time
from distutils.version import StrictVersion
import flika
from flika import global_vars as g
from flika.window import Window
from flika.utils.io import tifffile
from flika.process.file_ import get_permutation_tuple
from flika.utils.misc import open_file_gui


flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector

#from flika.process.file_ import open_file
from .roi_GD import makeROI

class CenterSurroundROI(BaseProcess_noPriorWindow):
    """
    Creates 2 linked ROI
        Center ROI creates trace
        Outer ROI used for scaling or background subtraction
    """
    def __init__(self):
        self.center_ROI = None
        self.surround_ROI = None
        self.subtractPlotWidget = None
        self.subtractPlot = None
        self.initiated = False
                        
        if g.settings['centerSurroundROI'] is None or 'centerSize' not in g.settings['centerSurroundROI']:
            s = dict()
            s['surroundWidth'] = 5
            s['centerWidth'] = 10
            s['centerHeight'] = 10  
            s['centerSize'] = 10                        
            g.settings['centerSurroundROI'] = s
        
        super().__init__()

    def __call__(self, surroundWidth, centerWidth, centerHeight, centerSize):
        '''
        
        '''
        g.settings['centerSurroundROI']['surroundWidth']=surroundWidth
        g.settings['centerSurroundROI']['centerWidth']=centerWidth
        g.settings['centerSurroundROI']['centerHeight']=centerHeight
        g.settings['centerSurroundROI']['centerSize']=centerSize
        
        if self.initiated: 
            self.closeAll()
        return

    def closeEvent(self, event):
        if self.initiated: 
            self.closeAll()
        BaseProcess_noPriorWindow.closeEvent(self, event)

    def closeAll(self):
        self.center_ROI.sigRegionChangeFinished.disconnect(self.getSubtract)
        self.surround_ROI.sigRegionChangeFinished.disconnect(self.getSubtract)
        self.subtractPlotWidget.close()
        self.center_ROI.traceWindow.close()
        self.center_ROI.delete()
        self.surround_ROI.delete() 
        self.initiated = False


    def displaySurround(self):
        print('clicked')                    

    def startROItrace(self):
        #check start button state
        if self.startButton.isEnabled == False:
            return
        
        self.win = self.getValue('active_window')
        #get window shape
        height = self.win.my
        width = self.win.mx
        
        #create rois
        self.center_ROI = makeROI('center',[[height/2, width/2], [self.centerHeight, self.centerWidth]], color=QColor(255, 0, 0, 127), window=self.win)
        self.surround_ROI = makeROI('surround',[[(height/2)-5, (width/2)-5], [self.surroundWidth, self.surroundWidth]], color=QColor(0, 0, 255, 127), window=self.win)
        
        #link rois
        self.center_ROI.addSurround(self.surround_ROI)
        self.surround_ROI.surroundWidth = self.surroundWidth
        self.center_ROI.surroundWidth = self.surroundWidth
        self.center_ROI.updateSurround(finish=False)
        
        #plot roi average trace
        self.plotCenter = self.center_ROI.plot()  
        self.plotSurround = self.surround_ROI.plot()  
        
        #get trace data
        self.traceCenter = self.center_ROI.getTrace() 
        self.traceSurround = self.surround_ROI.getTrace()   

        #link roi changes to subtract update    
        self.center_ROI.sigRegionChangeFinished.connect(self.getSubtract)
        self.surround_ROI.sigRegionChangeFinished.connect(self.getSubtract)
        
        #start subtract plot
        self.subtractPlotWidget = pg.PlotWidget(name='Subtract',title='Subtract')
        self.subtractPlot = self.subtractPlotWidget.plot(title="Subtract")
        self.subtractPlotWidget.show()
        self.getSubtract()
        
        #only allow one start action
        self.startButton.setEnabled(False)
        self.initiated = True

    def gui(self):
        #get saved settings        
        s=g.settings['centerSurroundROI']        
        self.surroundWidth = s['surroundWidth']
        self.centerWidth = s['centerWidth']
        self.centerHeight = s['centerHeight']
        self.centerSize = self.centerWidth #setting size to width at start
        #setup GUI        
        self.gui_reset()        
        self.active_window = WindowSelector()

        self.displaySurroundButton = QPushButton('Display Surround ROI')
        self.displaySurroundButton.pressed.connect(self.displaySurround)
        
        self.startButton = QPushButton('Start')
        self.startButton.pressed.connect(self.startROItrace)        

        self.width = SliderLabel(0)
        self.width.setRange(1,20) 
        self.width.setValue(self.surroundWidth)
        self.width.valueChanged.connect(self.updateWidth)
        self.width.slider.sliderReleased.connect(self.getSubtract)
        
        self.widthCenter = SliderLabel(0)
        self.widthCenter.setRange(1,500) 
        self.widthCenter.setValue(self.centerWidth)
        self.widthCenter.valueChanged.connect(self.updateCenterWidth)
        self.widthCenter.slider.sliderReleased.connect(self.getSubtract)

        self.heightCenter = SliderLabel(0)
        self.heightCenter.setRange(1,500) 
        self.heightCenter.setValue(self.centerHeight)
        self.heightCenter.valueChanged.connect(self.updateCenterHeight) 
        self.heightCenter.slider.sliderReleased.connect(self.getSubtract)

        self.sizeCenter = SliderLabel(0)
        self.sizeCenter.setRange(1,500) 
        self.sizeCenter.setValue(self.centerSize)
        self.sizeCenter.valueChanged.connect(self.updateCenterSize)    
        self.sizeCenter.slider.sliderReleased.connect(self.getSubtract)
        
        self.scaleImages = CheckBox()

        self.items.append({'name': 'active_window', 'string': 'Select Window', 'object': self.active_window})
        self.items.append({'name': 'surroundWidth', 'string': 'Set Surround Width', 'object': self.width})  
        self.items.append({'name': 'centerWidth', 'string': 'Set Center Width', 'object': self.widthCenter})  
        self.items.append({'name': 'centerHeight', 'string': 'Set Center Height', 'object': self.heightCenter})
        self.items.append({'name': 'centerSize', 'string': 'Set Center Size (as square)', 'object': self.sizeCenter})         
        #self.items.append({'name': 'scaleImages', 'string': 'Scale trace', 'object': self.scaleImages})
        #self.items.append({'name': 'displaySurround_button', 'string': '          ', 'object': self.displaySurroundButton})
        self.items.append({'name': 'start_button', 'string': '          ', 'object': self.startButton})        
        
        super().gui()

    def updateWidth(self):
        self.surroundWidth = self.width.value()
        self.surround_ROI.updateWidth(self.surroundWidth)
        
    def updateCenterWidth(self):
        self.centerWidth = self.widthCenter.value()
        self.center_ROI.updateWidth(self.centerWidth)
        
    def updateCenterHeight(self):
        self.centerHeight = self.heightCenter.value()
        self.center_ROI.updateHeight(self.centerHeight) 

    def updateCenterSize(self):
        self.centerSize = self.sizeCenter.value()
        self.center_ROI.updateSize(self.centerSize)  
        self.widthCenter.setValue(self.centerSize)
        self.heightCenter.setValue(self.centerSize)        

    def getSubtract(self):
        subtract = np.array(np.subtract(self.center_ROI.getTrace(),self.surround_ROI.getTrace()))
        self.subtractPlot.setData(y=subtract,x=np.arange(len(subtract)))
        

centerSurroundROI = CenterSurroundROI()