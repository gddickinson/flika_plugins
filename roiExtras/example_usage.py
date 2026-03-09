"""
ROI Extras Plugin - Example Usage
Real-time histogram of pixel values within the current ROI.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a test image with distinct intensity populations
test_data = np.zeros((64, 64), dtype=np.float32)
test_data[:32, :] = np.random.normal(100, 10, (32, 64))   # dim region
test_data[32:, :] = np.random.normal(200, 15, (32, 64))   # bright region

win = Window(test_data, name='Two Populations')

# Draw an ROI on the image before starting
# (Use flika's ROI tools to draw a rectangle)

# Launch ROI Extras
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('roiExtras')

from plugins.roiExtras import roiExtras
roiExtras.gui()

# The GUI shows:
# - Live histogram of pixel values inside the ROI
# - Auto-scaling X axis option
# - Updates in real time as you move or resize the ROI
# - ROI-masked image preview
