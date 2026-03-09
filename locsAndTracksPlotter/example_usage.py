"""
Locs and Tracks Plotter Plugin - Example Usage
Visualizes localization and tracking data overlaid on images.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a test image stack
test_data = np.random.poisson(50, (100, 128, 128)).astype(np.float32)
win = Window(test_data, name='Test Image')

# Launch the plotter
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('locsAndTracksPlotter')

# The plugin provides a docking panel with:
# - Localization file loading (CSV from ThunderSTORM, etc.)
# - Track file loading
# - Color coding options (by track ID, time, intensity)
# - Filtering controls (by frame range, track length, etc.)
# - Overlay rendering on the active image window
#
# To use interactively:
# 1. Open an image stack in flika
# 2. Go to Plugins > Locs and Tracks Plotter
# 3. Load your localization CSV file
# 4. Adjust display settings
# 5. Overlay updates in real-time as you scrub through frames
