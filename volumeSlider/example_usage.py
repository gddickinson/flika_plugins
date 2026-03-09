# -*- coding: utf-8 -*-
"""
Volume Slider Plugin - Example Usage
======================================
Demonstrates how to view 4D volumetric data using the Volume Slider plugin.
"""

from flika import *
start_flika()

import numpy as np
from flika.window import Window

# Create synthetic light-sheet data:
# 10 volumes x 20 slices x 128 x 128 = 200 total frames
n_volumes = 10
n_slices = 20
height, width = 128, 128
n_frames = n_volumes * n_slices

stack = np.random.normal(100, 10, (n_frames, height, width)).astype(np.float32)

# Add a bright sphere that changes intensity over volumes
for vol in range(n_volumes):
    intensity = 50 + 30 * np.sin(vol * 0.5)
    for z in range(n_slices):
        frame = vol * n_slices + z
        # Sphere centered at (64, 64), radius depends on Z slice
        r = max(1, 15 - abs(z - n_slices // 2))
        yy, xx = np.ogrid[:height, :width]
        mask = (yy - 64)**2 + (xx - 64)**2 < r**2
        stack[frame][mask] += intensity

w = Window(stack, name='Light Sheet Example')

# Launch the Volume Slider plugin
# Set slices_per_volume = 20 to match our synthetic data
from volumeSlider import volumeSlider_Start
volumeSlider_Start.volumeSlider.gui()

# In the Volume Slider panel:
# 1. Set "Slices per volume" to 20
# 2. Click "Reshape" to convert to 4D
# 3. Use the slice slider to navigate Z planes
# 4. Try dF/F0 with baseline volumes 0-2
