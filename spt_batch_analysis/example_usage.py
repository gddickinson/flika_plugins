"""
SPT Batch Analysis Plugin - Example Usage
Batch processing for single-particle tracking data.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a test image with moving particles
n_frames, h, w = 200, 128, 128
test_data = np.random.poisson(20, (n_frames, h, w)).astype(np.float32)

# Add several diffusing particles
rng = np.random.default_rng(42)
n_particles = 10
positions = rng.uniform(20, 108, (n_particles, 2))  # initial y, x

for t in range(n_frames):
    for i in range(n_particles):
        positions[i] += rng.normal(0, 0.5, 2)  # Brownian step
        positions[i] = np.clip(positions[i], 5, 123)
        yi, xi = int(positions[i, 0]), int(positions[i, 1])
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                if 0 <= yi+dy < h and 0 <= xi+dx < w:
                    test_data[t, yi+dy, xi+dx] += 300 * np.exp(
                        -(dy**2 + dx**2) / 2.0)

win = Window(test_data, name='Particles')

# Launch SPT batch analysis
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('spt_batch_analysis')

# The plugin provides:
# - Batch detection across multiple files
# - ThunderSTORM-based localization
# - Track linking with configurable parameters
# - Export of tracks and statistics
# - Multiple docks for parameter configuration
#
# Typical workflow:
# 1. Load or select image files
# 2. Configure detection parameters (threshold, PSF size)
# 3. Configure linking parameters (max distance, gap frames)
# 4. Run batch analysis
# 5. Export results
