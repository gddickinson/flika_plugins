"""
Detect Puffs (GD Edit) Plugin - Example Usage
Detects and analyzes calcium puff events in imaging data.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a synthetic calcium imaging stack with puff-like events
n_frames, h, w = 200, 64, 64
test_data = np.random.poisson(100, (n_frames, h, w)).astype(np.float32)

# Add synthetic puffs at known locations
def add_puff(data, t0, y0, x0, amp=300, sigma=3, duration=10):
    yy, xx = np.ogrid[-y0:h-y0, -x0:w-x0]
    spatial = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    for dt in range(duration):
        if t0 + dt < n_frames:
            temporal = amp * np.exp(-dt / 3.0)
            data[t0 + dt] += temporal * spatial

add_puff(test_data, 20, 30, 25)
add_puff(test_data, 50, 45, 40)
add_puff(test_data, 100, 15, 50)

win = Window(test_data, name='Calcium Puffs')

# Launch the puff detector
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('detect_puffs_GDedit')

# The plugin has a complex GUI accessed through:
# Plugins > Detect Puffs
# It performs:
# 1. Background subtraction (scaled average subtract)
# 2. Gaussian filtering
# 3. Threshold-based event detection
# 4. Cluster analysis of detected events
# 5. Measurement of puff amplitude, duration, and spatial extent
