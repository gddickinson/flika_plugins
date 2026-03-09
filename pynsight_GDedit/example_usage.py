"""
Pynsight (GD Edit) Plugin - Example Usage
Super-resolution reconstruction from single-molecule localization data.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create synthetic single-molecule blinking data
n_frames, h, w = 500, 64, 64
test_data = np.random.poisson(10, (n_frames, h, w)).astype(np.float32)

# Add sparse blinking emitters at known positions
rng = np.random.default_rng(42)
n_emitters = 50
positions = rng.uniform(5, 59, (n_emitters, 2))  # y, x positions

for t in range(n_frames):
    # Each emitter has a 5% chance of being "on" per frame
    for y, x in positions:
        if rng.random() < 0.05:
            yi, xi = int(y), int(x)
            # Add a PSF-like spot
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if 0 <= yi+dy < h and 0 <= xi+dx < w:
                        test_data[t, yi+dy, xi+dx] += 200 * np.exp(
                            -(dy**2 + dx**2) / 2.0)

win = Window(test_data, name='SMLM Data')

# Launch pynsight
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('pynsight_GDedit')

# Access through Plugins > Pynsight
# The plugin performs:
# 1. Single-molecule localization (fitting PSFs)
# 2. Drift correction
# 3. Super-resolution image reconstruction
# 4. Cluster analysis of localizations
