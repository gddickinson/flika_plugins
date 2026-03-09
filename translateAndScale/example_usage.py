# -*- coding: utf-8 -*-
"""
Translate and Scale Plugin - Example Usage
============================================
Demonstrates how to align point localization data to micropattern templates.

This plugin is used to register single-molecule tracking data (e.g., from
STORM/PALM) to known micropattern geometries (disc, square, crossbow, Y, H).
"""

from flika import *
start_flika()

import numpy as np
from flika.window import Window

# Create a synthetic image with a circular pattern
height, width = 256, 256
image = np.zeros((height, width), dtype=np.float32)
yy, xx = np.ogrid[:height, :width]
# Disc pattern centered at (128, 128) with radius 80
mask = (yy - 128)**2 + (xx - 128)**2 < 80**2
image[mask] = 1.0
# Add some noise
image += np.random.normal(0, 0.1, image.shape).astype(np.float32)

w = Window(image, name='Micropattern Image')

# Launch the Translate and Scale plugin
from translateAndScale import translateAndScale
translateAndScale.gui()

# In the plugin dialog:
# 1. Select template type (disc, square, crossbow, Y, H)
# 2. Click "Auto Detect" to find the pattern center and rotation
# 3. Fine-tune alignment using the draggable/rotatable ROI
# 4. Load a CSV file with point localizations to transform
# 5. Export the aligned coordinates

# For programmatic CSV alignment, prepare a DataFrame with 'x' and 'y' columns
# containing localization coordinates in pixels, then use the plugin's
# transform functions to align them to the template.
