"""
Center-Surround ROI Plugin - Example Usage
Creates an annular ROI for measuring local background-subtracted signals.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a test image with a bright spot on varying background
test_data = np.random.poisson(50, (100, 128, 128)).astype(np.float32)
# Add a bright signal source
yy, xx = np.ogrid[-64:64, -64:64]
spot = 200 * np.exp(-(xx**2 + yy**2) / (2 * 5**2))
for t in range(100):
    test_data[t] += spot * (1 + 0.5 * np.sin(2 * np.pi * t / 20))

win = Window(test_data, name='Spot with Background')

# Launch center-surround ROI GUI
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('centerSurroundROI')

from plugins.centerSurroundROI import centerSurroundROI
centerSurroundROI.gui()

# The GUI allows you to:
# - Select the active window
# - Set width, height, and size of the annular ROI
# - Click Start to begin measuring center - surround signal
