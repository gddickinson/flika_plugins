"""
OpenCV Camera Plugin - Example Usage
Live camera capture and display using OpenCV.
"""
from flika import *

# Start flika
start_flika()

# Launch OpenCV camera GUI
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('openCV')

from plugins.openCV import openCVcam
openCVcam.gui()

# The GUI provides:
# - Camera selection (by device index)
# - Live preview window
# - Filter options (e.g., edge detection, blur)
# - Frame capture controls
# - Recording to image stack
#
# Requirements:
# - OpenCV (cv2) must be installed
# - A connected camera device
#
# Note: If no camera is available, the plugin will show an error.
