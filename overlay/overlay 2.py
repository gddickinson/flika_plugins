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


import numba
pg.setConfigOption('useNumba', True)

flika_version = flika.__version__
if StrictVersion(flika_version) < StrictVersion('0.2.23'):
    from flika.process.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector
else:
    from flika.utils.BaseProcess import BaseProcess, SliderLabel, CheckBox, ComboBox, BaseProcess_noPriorWindow, WindowSelector


from pyqtgraph import HistogramLUTWidget

def gammaCorrect(img, gamma):
    gammaCorrection = 1/gamma
    maxIntensity = np.max(img)
    return np.array(maxIntensity*(img / maxIntensity) ** gammaCorrection)


class Overlay(BaseProcess_noPriorWindow):
    """
    Overlay two image stacks. 
    
    If Scale Images is ticked the dimmer image is rescaled to
    match the range of the brighter image.
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        self.current_red = None
        self.current_green = None
        self.greenIntensity =0
        self.redIntensity =0
        self.greenMinValue=0
        self.redMinValue=0
        
        self.gammaWindow = None

    def __call__(self):
        '''
        
        '''
        pass
        return

    def closeEvent(self, event):
        self.unlink_frames(self.current_red, self.current_green)
        BaseProcess_noPriorWindow.closeEvent(self, event)


    def unlink_frames(self, *windows):
        for window in windows:
            if window != None:
                try:
                    window.sigTimeChanged.disconnect(self.indexChanged)
                except:
                    pass

                    
    # def merge(self):
    #     red = self.getValue('red_window').image
    #     green = self.getValue('green_window').image
    #     checked = self.getValue('scaleImages')
        
    #     self.merged = np.zeros((red.shape[0], red.shape[1], red.shape[2], 3))
 
    #     if checked == True:
    #         if np.max(red) > np.max(green):
    #             green = green * int((np.max(red)/np.max(green)))
    #         else:
    #             red = red * int((np.max(green)/np.max(red)))
            
 
    #     self.merged[:,:,:,0] = red
    #     self.merged[:,:,:,1] = green
       
        
    #     #print(np.max(red),np.max(green))
    #     self.displayWindow_Merged = Window(self.merged,'Merged')
    #     return

    # def updateGreen(self):
    #     newA = self.getValue('green_window').image * self.greenIntensity
    #     newA[newA<self.greenMinValue] = 0
    #     self.merged[:,:,:,1] = newA        
    #     self.displayWindow_Merged.imageview.updateImage()

                    
    # def updateRed(self):       
    #     newA = self.getValue('red_window').image * self.redIntensity
    #     newA[newA<self.redMinValue] = 0
    #     self.merged[:,:,:,0] = newA
    #     self.displayWindow_Merged.imageview.updateImage()
     

    # def updateGreenMin(self, value):        
    #     self.greenMinValue = value
    #     self.updateGreen()
        
    # def updateRedMin(self, value):        
    #     self.redMinValue = value
    #     self.updateRed()

    # def updateGreenIntensity(self, value):
    #     self.greenIntensity = value
    #     self.updateGreen()

    # def updateRedIntensity(self, value):
    #     self.redIntensity = value
    #     self.updateRed()

    def overlay(self):
        red = self.getValue('red_window').image
        green = self.getValue('green_window').image[0] #just 1st image of stack
        
        self.displayWindow_Overlay = Window(red,'Overlay')
        
        self.OverlayLUT = 'inferno'
        self.OverlayMODE = QPainter.CompositionMode_SourceOver
        self.OverlayOPACITY = 0.5
        self.useOverlayLUT = False
        
        self.gradientPreset = 'grey' #  'thermal','flame','yellowy','bipolar','spectrum','cyclic','greyclip','grey'
        self.usePreset = False
        self.gradientState = None  
        self.useSharedState = False
        self.sharedState = None
        self.sharedLevels = None
               
        #init overlay levels 
        levels = self.getValue('green_window').imageview.getHistogramWidget().getLevels()
         

        self.overlayFlag = False
        self.overlayArrayLoaded = False

        self.bgItem = pg.ImageItem()
        if self.gammaCorrect.isChecked():
            green = gammaCorrect(green, self.gamma.value())
        self.bgItem.setImage(green, autoRange=False, autoLevels=False, levels=levels, opacity=self.OverlayOPACITY)
        self.bgItem.setCompositionMode(self.OverlayMODE)
        self.displayWindow_Overlay.imageview.view.addItem(self.bgItem)

        self.bgItem.hist_luttt = HistogramLUTWidget(fillHistogram = False)


        self.bgItem.hist_luttt.setMinimumWidth(110)
        self.bgItem.hist_luttt.setImageItem(self.bgItem)


        self.displayWindow_Overlay.imageview.ui.gridLayout.addWidget(self.bgItem.hist_luttt, 0, 4, 1, 4)  


# =============================================================================
#         #link TOP overlay histgramLUT to other windows
#         self.getValue('green_window').imageview.getHistogramWidget().item.sigLevelsChanged.connect(self.setOverlayLevels)
#         self.getValue('green_window').imageview.getHistogramWidget().item.sigLookupTableChanged.connect(self.setOverlayLUTs)                
#         self.getValue('red_window').imageview.getHistogramWidget().item.sigLevelsChanged.connect(self.setMainLevels)
#         self.getValue('red_window').imageview.getHistogramWidget().item.sigLookupTableChanged.connect(self.setMainLUTs)        
#         
#         self.histogramsLinked = True
# 
#     def setOverlayLevels(self):
#         if self.histogramsLinked == False:
#             return
#         levels = self.getValue('green_window').imageview.getHistogramWidget()
#         self.bgItem.hist_luttt.item.setLevels(levels[0],levels[1])
#         return
# 
#     def setOverlayLUTs(self):
#         if self.histogramsLinked == False:
#             return
#         lut = self.getValue('green_window').imageview.getHistogramWidget().item.gradient.saveState()
#         self.bgItem.hist_luttt.item.gradient.restoreState(lut)
#         return  
# 
#     def setMainLevels(self):
#         if self.histogramsLinked == False:
#             return
#         levels = self.getValue('red_window').imageview.getHistogramWidget()
#         self.displayWindow_Overlay.imageview.getHistogramWidget().item.setLevels(levels[0],levels[1])
#         return
# 
#     def setMainLUTs(self):
#         if self.histogramsLinked == False:
#             return
#         lut = self.getValue('red_window').imageview.getHistogramWidget().item.gradient.saveState()
#         self.imageview.getHistogramWidget().item.gradient.restoreState(lut)
#         return          
# =============================================================================

    def previewWindow(self):
        if self.gammaWindow is not None and not self.gammaWindow.closed:
            self.gammaWindow.close() 
        else:
            self.gammaImg = self.getValue('green_window').image[0]
            self.gammaWindow = Window(self.gammaImg, 'Gamma Preview')

    def updateGamma(self, value):
        if self.gammaWindow is not None and not self.gammaWindow.closed:
            levels = self.gammaWindow.imageview.getHistogramWidget().getLevels()
            gammaCorrrectedImg = gammaCorrect(self.gammaImg, value)
            self.gammaWindow.imageview.setImage(gammaCorrrectedImg, autoLevels=False, levels=levels)


    def gui(self):
        self.gui_reset()
        self.red_window = WindowSelector()
        self.green_window = WindowSelector()
        self.overlayButton = QPushButton('Overlay')
        self.overlayButton.pressed.connect(self.overlay)
        #self.mergeButton = QPushButton('Merge')
        #self.mergeButton.pressed.connect(self.merge)        
        
        #self.scaleImages = CheckBox()
        
        #self.greenScale = SliderLabel(0)
        #self.greenScale.setRange(0,10)
        #self.greenScale.setValue(1) 
        
        #self.redScale = SliderLabel(0)
        #self.redScale.setRange(0,10)
        #self.redScale.setValue(1)   
        
        #self.greenMin = SliderLabel(0)
        #self.greenMin.setRange(0,1000)
        #self.greenMin.setValue(0) 
        
        #self.redMin = SliderLabel(0)
        #self.redMin.setRange(0,1000)
        #self.redMin.setValue(0)         
            
                
        #self.greenScale.valueChanged.connect(self.updateGreenIntensity)
        #self.redScale.valueChanged.connect(self.updateRedIntensity)        
        #self.greenMin.valueChanged.connect(self.updateGreenMin)
        #self.redMin.valueChanged.connect(self.updateRedMin)  
        
        
        #Gamma correct
        self.gammaCorrect = CheckBox()
        self.previewGamma = CheckBox()
        self.gamma = SliderLabel(1)
        self.gamma.setRange(0.0,20.0)
        self.gamma.setValue(0.0) 
        
        self.previewGamma.stateChanged.connect(self.previewWindow)
        self.gamma.valueChanged.connect(self.updateGamma)

        self.items.append({'name': 'red_window', 'string': 'CH1 (r)', 'object': self.red_window})
        self.items.append({'name': 'green_window', 'string': 'CH2 (g)', 'object': self.green_window})
        #self.items.append({'name': 'scaleImages', 'string': 'Scale?', 'object': self.scaleImages})
        #self.items.append({'name': 'merge_button', 'string': '', 'object': self.mergeButton})
        self.items.append({'name': 'overlay_button', 'string': '', 'object': self.overlayButton})  
        
        self.items.append({'name': 'gammaCorrect', 'string': 'Gamma Corrrect', 'object': self.gammaCorrect})          
        self.items.append({'name': 'gamma', 'string': 'Gamma', 'object': self.gamma}) 
        self.items.append({'name': 'gammaPreview', 'string': 'preview Gamma', 'object': self.previewGamma})          
        
        #self.items.append({'name': 'red_scaler', 'string': 'red', 'object': self.redScale})
        #self.items.append({'name': 'green_scaler', 'string': 'green', 'object': self.greenScale})       
        
        #self.items.append({'name': 'red_min', 'string': 'red min', 'object': self.redMin})
        #self.items.append({'name': 'green_min', 'string': 'green min', 'object': self.greenMin})  
        

        super().gui()
overlay = Overlay()