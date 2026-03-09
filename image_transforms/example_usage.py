"""
Image Transforms Plugin - Example Usage
Provides rotation and flip operations for image stacks.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create an asymmetric test image so transforms are visible
test_data = np.zeros((10, 64, 64), dtype=np.float32)
test_data[:, 10:30, 10:20] = 200   # tall rectangle in upper-left
test_data[:, 50:55, 40:60] = 150   # wide rectangle in lower-right

win = Window(test_data, name='Asymmetric Test')

# Launch image transforms
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('image_transforms')

# Rotate 90 degrees
from plugins.image_transforms import rotate_90
rotate_90.gui()
# Select direction: Clockwise or Counter-clockwise

# Custom angle rotation
from plugins.image_transforms import rotate_custom
rotate_custom.gui()
# Set angle, reshape option, interpolation order

# Flip image
from plugins.image_transforms import flip_image
flip_image.gui()
# Select direction: Vertical or Horizontal
