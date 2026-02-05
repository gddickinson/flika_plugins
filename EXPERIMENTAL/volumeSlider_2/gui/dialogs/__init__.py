# gui/dialogs/__init__.py
#!/usr/bin/env python3
"""
GUI Dialogs Module
==================

Dialog components for the Volume Slider plugin.
"""

try:
    from .test_data_dialog import TestDataDialog
    __all__ = ['TestDataDialog']
except ImportError:
    __all__ = []

# volumeSlider_Start.py (create this file in your main directory)
#!/usr/bin/env python3
"""
FLIKA Volume Slider Plugin Entry Point
=====================================

This is the main entry point that FLIKA looks for to load the plugin.
"""

import logging
from flika import global_vars as g
from flika.utils.BaseProcess import BaseProcess_noPriorWindow
from qtpy.QtWidgets import QLabel

# Import the main plugin
try:
    from .main import VolumeSliderPlugin
    HAS_MAIN_PLUGIN = True
except ImportError as e:
    print(f"Warning: Could not import main plugin: {e}")
    HAS_MAIN_PLUGIN = False


class volumeSliderBase(BaseProcess_noPriorWindow):
    """
    FLIKA plugin base class for Volume Slider.
    
    This is what FLIKA instantiates when loading the plugin.
    """
    
    def __init__(self):
        super().__init__()
        self.plugin_window = None
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, **kwargs):
        """Launch the Volume Slider plugin."""
        try:
            if not HAS_MAIN_PLUGIN:
                g.m.statusBar().showMessage("Volume Slider plugin not properly installed")
                return
            
            # Create or show the plugin window
            if self.plugin_window is None or not self.plugin_window.isVisible():
                self.plugin_window = VolumeSliderPlugin()
            
            self.plugin_window.show()
            self.plugin_window.raise_()
            self.plugin_window.activateWindow()
            
            g.m.statusBar().showMessage("Volume Slider launched successfully")
            
        except Exception as e:
            error_msg = f"Error launching Volume Slider: {str(e)}"
            self.logger.error(error_msg)
            g.m.statusBar().showMessage(error_msg)
    
    def gui(self):
        """Create GUI for FLIKA's process dialog."""
        self.items = []
        self.items.append({
            'name': 'info', 
            'string': 'Launch Volume Slider Professional',
            'object': QLabel("Click OK to launch the Volume Slider Professional interface")
        })
        super().gui()
    
    def closeEvent(self, event):
        """Handle plugin window closure."""
        if self.plugin_window:
            self.plugin_window.close()
        super().closeEvent(event)


# Create the global instance that FLIKA expects
volumeSliderBase = volumeSliderBase()

# Also make it available with the old naming convention
volume_slider_plugin = volumeSliderBase