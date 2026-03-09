"""
Annotator Plugin - Example Usage
Adds frame index annotations to image stacks.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a test image stack (50 frames, 128x128)
test_data = np.random.poisson(100, (50, 128, 128)).astype(np.float32)
w = Window(test_data, name='Test Stack')

# Launch annotator GUI
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('annotator')

from plugins.annotator import annotator
annotator.gui()

# The GUI allows you to:
# - Set font size
# - Choose font colour (white/black/red/green/blue)
# - Choose text position (Top Left, Top Right, Bottom Left, Bottom Right)
# - Click "Add Index" to burn frame numbers into the image
