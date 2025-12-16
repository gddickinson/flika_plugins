#!/usr/bin/env python3
"""
FLIKA Volume Slider Plugin Entry Point
=====================================

This is the main entry point that FLIKA looks for to load the plugin.
Based on your info.xml, FLIKA expects to find volumeSliderBase here.
"""

import logging
import traceback
from flika import global_vars as g
from flika.utils.BaseProcess import BaseProcess_noPriorWindow
from qtpy.QtWidgets import QLabel, QMessageBox

# Import the main plugin
try:
    from .main import VolumeSliderPlugin
    HAS_MAIN_PLUGIN = True
except ImportError as e:
    print(f"Warning: Could not import main plugin: {e}")
    HAS_MAIN_PLUGIN = False
    VolumeSliderPlugin = None


class volumeSliderBase(BaseProcess_noPriorWindow):
    """
    FLIKA plugin base class for Volume Slider.
    
    This is what FLIKA instantiates when loading the plugin.
    The class name 'volumeSliderBase' matches what's expected in info.xml
    """
    
    def __init__(self):
        super().__init__()
        self.plugin_window = None
        self.logger = logging.getLogger(__name__)
        
        # Log plugin loading
        self.logger.info("Volume Slider Plugin base initialized")
    
    def __call__(self, **kwargs):
        """Launch the Volume Slider plugin."""
        try:
            if not HAS_MAIN_PLUGIN:
                error_msg = ("Volume Slider plugin components not properly loaded. "
                           "Check console for import errors.")
                g.m.statusBar().showMessage(error_msg)
                QMessageBox.warning(None, "Plugin Error", error_msg)
                return
            
            # Create or show the plugin window
            if self.plugin_window is None or not self.plugin_window.isVisible():
                self.plugin_window = VolumeSliderPlugin()
                self.logger.info("Created new Volume Slider window")
            
            self.plugin_window.show()
            self.plugin_window.raise_()
            self.plugin_window.activateWindow()
            
            g.m.statusBar().showMessage("Volume Slider launched successfully")
            self.logger.info("Volume Slider window shown")
            
        except Exception as e:
            error_msg = f"Error launching Volume Slider: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Show error to user
            g.m.statusBar().showMessage(error_msg)
            QMessageBox.critical(None, "Volume Slider Error", 
                               f"Failed to launch Volume Slider:\n\n{str(e)}\n\n"
                               "Check the console for detailed error information.")
    
    def gui(self):
        """Create GUI for FLIKA's process dialog."""
        try:
            self.items = []
            self.items.append({
                'name': 'info', 
                'string': 'Launch Volume Slider Professional',
                'object': QLabel("Click OK to launch the Volume Slider Professional interface\n"
                               "for 3D/4D lightsheet microscopy analysis.")
            })
            super().gui()
            
        except Exception as e:
            self.logger.error(f"Error creating GUI: {str(e)}")
    
    def closeEvent(self, event):
        """Handle plugin window closure."""
        try:
            if self.plugin_window:
                self.plugin_window.close()
                self.plugin_window = None
            super().closeEvent(event)
            
        except Exception as e:
            self.logger.error(f"Error during close: {str(e)}")


# Create the global instance that FLIKA expects
# This must match the function name in your info.xml menu_layout
volumeSliderBase = volumeSliderBase()


# Optional: Also provide a direct function interface
def launch_volume_slider():
    """Direct function to launch Volume Slider."""
    volumeSliderBase()


# Test function for development
def test_plugin():
    """Test function to check if plugin can be loaded."""
    try:
        print("Testing Volume Slider Plugin...")
        print(f"Main plugin available: {HAS_MAIN_PLUGIN}")
        
        if HAS_MAIN_PLUGIN:
            print("✓ Main plugin can be imported")
            # Try creating an instance
            test_window = VolumeSliderPlugin()
            print("✓ Plugin window can be created")
            test_window.close()
            print("✓ Plugin window can be closed")
        else:
            print("✗ Main plugin cannot be imported")
        
        print("Plugin test completed")
        return HAS_MAIN_PLUGIN
        
    except Exception as e:
        print(f"✗ Plugin test failed: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    # If running this file directly, run the test
    test_plugin()