"""
Flash Remover Plugin - Example Usage
Removes bright flash artifacts from image stacks.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create test data with a flash artifact
n_frames, h, w = 100, 64, 64
test_data = np.random.poisson(100, (n_frames, h, w)).astype(np.float32)

# Add a bright flash at frames 30-35
test_data[30:35] += 500

win = Window(test_data, name='Stack with Flash')

# Launch flash remover GUI
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('flash_remover')

from plugins.flash_remover import flash_remover
flash_remover.gui()

# The GUI allows you to:
# - Select the window to process
# - Set flash range (start and end frames)
# - Choose removal method (interpolate, moving average, etc.)
# - Set moving average window size
# - Apply the flash removal
