"""
Kymograph Analyzer Plugin - Example Usage
Generates and analyzes kymographs from line ROIs on image stacks.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a test stack with a moving bright spot
n_frames, h, w = 100, 64, 64
test_data = np.random.poisson(20, (n_frames, h, w)).astype(np.float32)

# Moving spot along x at y=32
for t in range(n_frames):
    x_pos = int(10 + 0.4 * t)
    if x_pos < w - 5:
        yy, xx = np.ogrid[-32:h-32, -x_pos:w-x_pos]
        test_data[t] += 200 * np.exp(-(xx**2 + yy**2) / (2 * 3**2))

win = Window(test_data, name='Moving Spot')

# Draw a line ROI across the path of motion before launching
# (In flika: use the line ROI tool to draw across the spot trajectory)

# Launch kymograph analyzer
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('kymograph_analyzer')

from plugins.kymograph_analyzer import kymograph_analyzer
kymograph_analyzer.gui()

# The kymograph shows space along the ROI (x-axis) vs time (y-axis)
# Moving features appear as diagonal lines in the kymograph
