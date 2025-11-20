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
    Overlay multiple image stacks (up to three channels). 
    
    If Link Frames is ticked, all channels will scroll through time together.
    
    Gamma correction can be applied to each channel independently.
    """
    def __init__(self):
        BaseProcess_noPriorWindow.__init__(self)
        self.current_red = None
        self.current_green = None
        self.current_blue = None
        
        self.gammaWindow = None
        self.displayWindow_Overlay = None
        self.red_stack = None
        self.green_stack = None
        self.blue_stack = None
        self.current_frame = 0
        
        # Image items for each channel
        self.redItem = None
        self.greenItem = None
        self.blueItem = None
        
        # Channel visibility flags
        self.red_visible = True
        self.green_visible = True
        self.blue_visible = True

    def __call__(self):
        '''
        Create overlay when called
        '''
        self.overlay()
        return

    def closeEvent(self, event):
        self.unlink_frames(self.current_red, self.current_green, self.current_blue)
        BaseProcess_noPriorWindow.closeEvent(self, event)

    def unlink_frames(self, *windows):
        for window in windows:
            if window != None:
                try:
                    window.sigTimeChanged.disconnect(self.indexChanged)
                except:
                    pass
                    
    def indexChanged(self, index):
        """Handle frame change in any window"""
        self.current_frame = index
        self.update_overlay()
                    
    def overlay(self):
        """Create overlay of selected windows"""
        red_window = self.getValue('red_window')
        green_window = self.getValue('green_window')
        blue_window = self.getValue('blue_window')
        
        # Store reference to windows
        self.current_red = red_window
        self.current_green = green_window
        self.current_blue = blue_window
        
        # Store the image stacks
        self.red_stack = red_window.image if red_window is not None else None
        self.green_stack = green_window.image if green_window is not None else None
        self.blue_stack = blue_window.image if blue_window is not None else None
        
        # Determine which channel to use as base for display
        base_image = None
        if self.red_stack is not None:
            base_image = self.red_stack
        elif self.green_stack is not None:
            base_image = self.green_stack
        elif self.blue_stack is not None:
            base_image = self.blue_stack
        else:
            print("Error: No valid windows selected")
            return
            
        # Create display window
        self.displayWindow_Overlay = Window(base_image, 'RGB Overlay')
        
        # Setup composition mode and opacity
        self.OverlayMODE = QPainter.CompositionMode_SourceOver
        self.OverlayOPACITY = 0.5
        
        # Set flags
        self.overlayFlag = False
        self.overlayArrayLoaded = False
        
        # Clear view before adding items
        self.displayWindow_Overlay.imageview.view.clear()
        
        # Setup base image (red channel)
        self.redItem = pg.ImageItem()
        if self.red_stack is not None:
            first_red = self.red_stack
            if len(self.red_stack.shape) > 2:  # If it's a stack
                first_red = self.red_stack[0]
                
            # Apply gamma if needed
            if self.redGammaCorrect.isChecked():
                first_red = gammaCorrect(first_red, self.redGamma.value())
                
            # Set color to red channel
            red_lut = self._create_monochrome_lut([255, 0, 0])
            self.redItem.setLookupTable(red_lut)
            
            # Set image data
            red_levels = red_window.imageview.getHistogramWidget().getLevels() if red_window is not None else None
            self.redItem.setImage(first_red, autoRange=False, autoLevels=False if red_levels else True, 
                                  levels=red_levels, opacity=self.OverlayOPACITY)
            self.redItem.setCompositionMode(self.OverlayMODE)
            self.displayWindow_Overlay.imageview.view.addItem(self.redItem)
            
            # Setup histogram widget for red channel
            self.redItem.hist_luttt = HistogramLUTWidget(fillHistogram=False)
            self.redItem.hist_luttt.setMinimumWidth(110)
            self.redItem.hist_luttt.setImageItem(self.redItem)
            self.displayWindow_Overlay.imageview.ui.gridLayout.addWidget(self.redItem.hist_luttt, 0, 4, 1, 3)
        
        # Setup green overlay
        self.greenItem = pg.ImageItem()
        if self.green_stack is not None:
            first_green = self.green_stack
            if len(self.green_stack.shape) > 2:  # If it's a stack
                first_green = self.green_stack[0]
                
            # Apply gamma if needed
            if self.greenGammaCorrect.isChecked():
                first_green = gammaCorrect(first_green, self.greenGamma.value())
                
            # Set color to green channel
            green_lut = self._create_monochrome_lut([0, 255, 0])
            self.greenItem.setLookupTable(green_lut)
            
            # Set image data
            green_levels = green_window.imageview.getHistogramWidget().getLevels() if green_window is not None else None
            self.greenItem.setImage(first_green, autoRange=False, autoLevels=False if green_levels else True, 
                                    levels=green_levels, opacity=self.OverlayOPACITY)
            self.greenItem.setCompositionMode(self.OverlayMODE)
            self.displayWindow_Overlay.imageview.view.addItem(self.greenItem)
            
            # Setup histogram widget for green channel
            self.greenItem.hist_luttt = HistogramLUTWidget(fillHistogram=False)
            self.greenItem.hist_luttt.setMinimumWidth(110)
            self.greenItem.hist_luttt.setImageItem(self.greenItem)
            self.displayWindow_Overlay.imageview.ui.gridLayout.addWidget(self.greenItem.hist_luttt, 0, 7, 1, 3)
            
        # Setup blue overlay
        self.blueItem = pg.ImageItem()
        if self.blue_stack is not None:
            first_blue = self.blue_stack
            if len(self.blue_stack.shape) > 2:  # If it's a stack
                first_blue = self.blue_stack[0]
                
            # Apply gamma if needed
            if self.blueGammaCorrect.isChecked():
                first_blue = gammaCorrect(first_blue, self.blueGamma.value())
                
            # Set color to blue channel
            blue_lut = self._create_monochrome_lut([0, 0, 255])
            self.blueItem.setLookupTable(blue_lut)
            
            # Set image data
            blue_levels = blue_window.imageview.getHistogramWidget().getLevels() if blue_window is not None else None
            self.blueItem.setImage(first_blue, autoRange=False, autoLevels=False if blue_levels else True, 
                                   levels=blue_levels, opacity=self.OverlayOPACITY)
            self.blueItem.setCompositionMode(self.OverlayMODE)
            self.displayWindow_Overlay.imageview.view.addItem(self.blueItem)
            
            # Setup histogram widget for blue channel
            self.blueItem.hist_luttt = HistogramLUTWidget(fillHistogram=False)
            self.blueItem.hist_luttt.setMinimumWidth(110)
            self.blueItem.hist_luttt.setImageItem(self.blueItem)
            self.displayWindow_Overlay.imageview.ui.gridLayout.addWidget(self.blueItem.hist_luttt, 0, 10, 1, 3)
        
        # Connect time signals if link frames is checked
        if self.linkFrames.isChecked():
            # Connect the time change signal from main window to our handler
            self.displayWindow_Overlay.sigTimeChanged.connect(self.indexChanged)
            
            # Connect signals from source windows
            if red_window is not None and hasattr(red_window, 'sigTimeChanged'):
                red_window.sigTimeChanged.connect(self.indexChanged)
                
            if green_window is not None and hasattr(green_window, 'sigTimeChanged'):
                green_window.sigTimeChanged.connect(self.indexChanged)
                
            if blue_window is not None and hasattr(blue_window, 'sigTimeChanged'):
                blue_window.sigTimeChanged.connect(self.indexChanged)
                
        # Add visibility checkboxes below histograms
        self._add_channel_visibility_controls()

    def _add_channel_visibility_controls(self):
        """Add checkboxes for toggling channel visibility"""
        if self.displayWindow_Overlay is None:
            return
            
        # Create a widget to hold controls
        control_widget = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        control_widget.setLayout(layout)
        
        # Red channel checkbox
        if self.red_stack is not None:
            red_cb = QCheckBox("Red")
            red_cb.setChecked(True)
            red_cb.stateChanged.connect(lambda state: self._toggle_channel_visibility('red', state))
            layout.addWidget(red_cb)
            
        # Green channel checkbox
        if self.green_stack is not None:
            green_cb = QCheckBox("Green")
            green_cb.setChecked(True)
            green_cb.stateChanged.connect(lambda state: self._toggle_channel_visibility('green', state))
            layout.addWidget(green_cb)
            
        # Blue channel checkbox
        if self.blue_stack is not None:
            blue_cb = QCheckBox("Blue")
            blue_cb.setChecked(True)
            blue_cb.stateChanged.connect(lambda state: self._toggle_channel_visibility('blue', state))
            layout.addWidget(blue_cb)
            
        # Add control widget below histograms
        self.displayWindow_Overlay.imageview.ui.gridLayout.addWidget(control_widget, 1, 4, 1, 9)

    def _toggle_channel_visibility(self, channel, state):
        """Toggle visibility of a channel"""
        if channel == 'red' and self.redItem is not None:
            self.red_visible = bool(state)
            self.redItem.setVisible(self.red_visible)
        elif channel == 'green' and self.greenItem is not None:
            self.green_visible = bool(state)
            self.greenItem.setVisible(self.green_visible)
        elif channel == 'blue' and self.blueItem is not None:
            self.blue_visible = bool(state)
            self.blueItem.setVisible(self.blue_visible)

    def _create_monochrome_lut(self, rgb_color):
        """Create a monochrome lookup table for given RGB color"""
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(256):
            lut[i] = [int(rgb_color[0] * i / 255), int(rgb_color[1] * i / 255), int(rgb_color[2] * i / 255)]
        return lut

    def update_overlay(self):
        """Update the overlay with the current frame"""
        if self.displayWindow_Overlay is None or self.displayWindow_Overlay.closed:
            return
            
        # Update red channel
        if self.red_stack is not None and self.redItem is not None:
            if len(self.red_stack.shape) > 2:  # If it's a stack
                # Make sure the current frame index is valid
                if self.current_frame < self.red_stack.shape[0]:
                    # Get the current frame
                    current_red = self.red_stack[self.current_frame]
                    
                    # Apply gamma correction if needed
                    if self.redGammaCorrect.isChecked():
                        current_red = gammaCorrect(current_red, self.redGamma.value())
                        
                    # Update the image
                    levels = self.redItem.hist_luttt.getLevels()
                    self.redItem.setImage(current_red, autoLevels=False, levels=levels)
            
        # Update green channel
        if self.green_stack is not None and self.greenItem is not None:
            if len(self.green_stack.shape) > 2:  # If it's a stack
                # Make sure the current frame index is valid
                if self.current_frame < self.green_stack.shape[0]:
                    # Get the current frame
                    current_green = self.green_stack[self.current_frame]
                    
                    # Apply gamma correction if needed
                    if self.greenGammaCorrect.isChecked():
                        current_green = gammaCorrect(current_green, self.greenGamma.value())
                        
                    # Update the image
                    levels = self.greenItem.hist_luttt.getLevels()
                    self.greenItem.setImage(current_green, autoLevels=False, levels=levels)
                    
        # Update blue channel
        if self.blue_stack is not None and self.blueItem is not None:
            if len(self.blue_stack.shape) > 2:  # If it's a stack
                # Make sure the current frame index is valid
                if self.current_frame < self.blue_stack.shape[0]:
                    # Get the current frame
                    current_blue = self.blue_stack[self.current_frame]
                    
                    # Apply gamma correction if needed
                    if self.blueGammaCorrect.isChecked():
                        current_blue = gammaCorrect(current_blue, self.blueGamma.value())
                        
                    # Update the image
                    levels = self.blueItem.hist_luttt.getLevels()
                    self.blueItem.setImage(current_blue, autoLevels=False, levels=levels)
                    
        # Update the main display to the current frame
        self.displayWindow_Overlay.imageview.setCurrentIndex(self.current_frame)

    def preview_gamma(self, channel):
        """Preview gamma correction for a specific channel"""
        if self.gammaWindow is not None and not self.gammaWindow.closed:
            self.gammaWindow.close() 
            
        # Determine which channel to preview
        preview_img = None
        gamma_value = 1.0
        
        if channel == 'red' and self.getValue('red_window') is not None:
            window = self.getValue('red_window')
            gamma_value = self.redGamma.value()
        elif channel == 'green' and self.getValue('green_window') is not None:
            window = self.getValue('green_window')
            gamma_value = self.greenGamma.value()
        elif channel == 'blue' and self.getValue('blue_window') is not None:
            window = self.getValue('blue_window')
            gamma_value = self.blueGamma.value()
        else:
            return
            
        # Get first frame for preview
        if window.mt > 1:
            preview_img = window.image[0]
        else:
            preview_img = window.image
            
        # Apply gamma and create preview window
        gamma_corrected = gammaCorrect(preview_img, gamma_value)
        self.gammaWindow = Window(gamma_corrected, f'Gamma Preview ({channel})')

    def update_gamma(self, channel, value):
        """Update gamma correction for a specific channel"""
        # Update preview window if open
        if self.gammaWindow is not None and not self.gammaWindow.closed:
            # Check which channel is being previewed
            if (channel == 'red' and self.gammaWindow.name == 'Gamma Preview (red)' or
                channel == 'green' and self.gammaWindow.name == 'Gamma Preview (green)' or
                channel == 'blue' and self.gammaWindow.name == 'Gamma Preview (blue)'):
                
                # Get appropriate window
                if channel == 'red':
                    window = self.getValue('red_window')
                elif channel == 'green':
                    window = self.getValue('green_window')
                else:
                    window = self.getValue('blue_window')
                
                # Get image and apply gamma
                if window.mt > 1:
                    img = window.image[0]
                else:
                    img = window.image
                
                levels = self.gammaWindow.imageview.getHistogramWidget().getLevels()
                gamma_corrected = gammaCorrect(img, value)
                self.gammaWindow.imageview.setImage(gamma_corrected, autoLevels=False, levels=levels)
        
        # Update the overlay
        self.update_overlay()

    def gui(self):
        self.gui_reset()
        
        # Window selectors
        self.red_window = WindowSelector()
        self.green_window = WindowSelector()
        self.blue_window = WindowSelector()
        
        # Link frames checkbox
        self.linkFrames = CheckBox()
        self.linkFrames.setChecked(True)
        
        # Gamma correction for each channel
        self.redGammaCorrect = CheckBox()
        self.redGamma = SliderLabel(1)
        self.redGamma.setRange(0.1, 20.0)
        self.redGamma.setValue(1.0)
        self.redPreviewGamma = QPushButton('Preview')
        
        self.greenGammaCorrect = CheckBox()
        self.greenGamma = SliderLabel(1)
        self.greenGamma.setRange(0.1, 20.0)
        self.greenGamma.setValue(1.0) 
        self.greenPreviewGamma = QPushButton('Preview')
        
        self.blueGammaCorrect = CheckBox()
        self.blueGamma = SliderLabel(1)
        self.blueGamma.setRange(0.1, 20.0)
        self.blueGamma.setValue(1.0)
        self.bluePreviewGamma = QPushButton('Preview')
        
        # Connect signals
        self.redPreviewGamma.clicked.connect(lambda: self.preview_gamma('red'))
        self.redGamma.valueChanged.connect(lambda value: self.update_gamma('red', value))
        
        self.greenPreviewGamma.clicked.connect(lambda: self.preview_gamma('green'))
        self.greenGamma.valueChanged.connect(lambda value: self.update_gamma('green', value))
        
        self.bluePreviewGamma.clicked.connect(lambda: self.preview_gamma('blue'))
        self.blueGamma.valueChanged.connect(lambda value: self.update_gamma('blue', value))

        # Add items to GUI
        self.items.append({'name': 'red_window', 'string': 'CH1 (r)', 'object': self.red_window})
        self.items.append({'name': 'green_window', 'string': 'CH2 (g)', 'object': self.green_window})
        self.items.append({'name': 'blue_window', 'string': 'CH3 (b)', 'object': self.blue_window})
        self.items.append({'name': 'linkFrames', 'string': 'Link Frames', 'object': self.linkFrames})
        
        self.items.append({'name': 'redGammaCorrect', 'string': 'CH1 Gamma', 'object': self.redGammaCorrect})          
        self.items.append({'name': 'redGamma', 'string': 'Value', 'object': self.redGamma}) 
        self.items.append({'name': 'redPreviewGamma', 'string': '', 'object': self.redPreviewGamma})
        
        self.items.append({'name': 'greenGammaCorrect', 'string': 'CH2 Gamma', 'object': self.greenGammaCorrect})          
        self.items.append({'name': 'greenGamma', 'string': 'Value', 'object': self.greenGamma}) 
        self.items.append({'name': 'greenPreviewGamma', 'string': '', 'object': self.greenPreviewGamma})
        
        self.items.append({'name': 'blueGammaCorrect', 'string': 'CH3 Gamma', 'object': self.blueGammaCorrect})          
        self.items.append({'name': 'blueGamma', 'string': 'Value', 'object': self.blueGamma}) 
        self.items.append({'name': 'bluePreviewGamma', 'string': '', 'object': self.bluePreviewGamma})

        super().gui()
        
    def execute(self):
        """Override execute to perform overlay when OK is clicked"""
        self.overlay()
        return

overlay = Overlay()