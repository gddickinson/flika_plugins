"""
Light Sheet Analyzer (GD Edit) Plugin - Example Usage
Reconstructs 3D volumes from light sheet microscopy data.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create simulated light sheet data
# Light sheet data is a sequence of frames at different z-positions
n_steps = 20   # z-planes per volume
n_volumes = 5  # number of 3D volumes
h, w = 64, 64
n_frames = n_steps * n_volumes

test_data = np.random.poisson(50, (n_frames, h, w)).astype(np.float32)

# Add a 3D structure (sphere) that appears at different z-planes
for vol in range(n_volumes):
    for z in range(n_steps):
        frame = vol * n_steps + z
        z_pos = z - n_steps // 2
        radius_at_z = max(0, 10**2 - z_pos**2)
        if radius_at_z > 0:
            yy, xx = np.ogrid[-32:32, -32:32]
            mask = (xx**2 + yy**2) < radius_at_z
            test_data[frame][mask] += 200

win = Window(test_data, name='Light Sheet Raw')

# Launch light sheet analyzer
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('light_sheet_analyzer_GDedit')

from plugins.light_sheet_analyzer_GDedit import light_sheet_analyzer
light_sheet_analyzer.gui()

# The GUI allows you to:
# - Set nSteps (z-planes per volume)
# - Set shift factor for skewed light sheet correction
# - Set theta angle for oblique acquisition
# - Toggle triangle scan mode
# - Toggle interpolation
# - Reconstruct the 3D volume
