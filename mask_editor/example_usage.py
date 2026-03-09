"""
Mask Editor Plugin - Example Usage
Interactive binary mask painting tool.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a test image to draw masks on
test_data = np.random.poisson(100, (64, 64)).astype(np.float32)
# Add some structures that you might want to mask
yy, xx = np.ogrid[-32:32, -32:32]
test_data += 200 * np.exp(-(xx**2 + yy**2) / (2 * 8**2))

win = Window(test_data, name='Image for Masking')

# Launch mask editor
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('mask_editor')

# Access the mask editor through Plugins > Mask Editor
#
# Controls:
# - Left-click to draw (paint mask regions)
# - Shift + click to erase
# - Adjust brush size with the slider
# - D key: draw mode
# - E key: erase mode
# - Ctrl+Z: undo
# - Threshold: generate mask from intensity threshold
# - Invert: invert the mask
# - Fill/Clear: fill or clear entire mask
# - Save/Load: save mask to file
