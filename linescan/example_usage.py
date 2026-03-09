"""
Linescan Plugin - Example Usage
Analyzes linescan (xt) imaging data from confocal microscopes.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create synthetic linescan data (x-position vs time)
n_lines = 500   # time points
n_pixels = 128  # spatial pixels
test_data = np.random.poisson(30, (n_lines, n_pixels)).astype(np.float32)

# Add a calcium spark event
for t in range(50, 80):
    x_center = 64
    sigma = 5
    amp = 200 * np.exp(-(t - 50) / 10.0)
    xx = np.arange(n_pixels) - x_center
    test_data[t] += amp * np.exp(-xx**2 / (2 * sigma**2))

win = Window(test_data, name='Linescan Data')

# Launch linescan analyzer
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('linescan')

from plugins.linescan import linescan
linescan.gui()

# The GUI provides:
# - Start button to begin analysis
# - Options menu for configuration
# - Automated spark/event detection in xt data
# - Measurement of event properties (amplitude, width, duration)
