# -*- coding: utf-8 -*-
"""
Video Exporter Plugin - Example Usage
=======================================
Demonstrates how to export an image stack as MP4 video.

Requirements:
    - ffmpeg must be installed and on the system PATH
"""

from flika import *
start_flika()

import numpy as np
from flika.window import Window

# Create a synthetic time-series stack (200 frames, 256x256)
stack = np.random.normal(100, 10, (200, 256, 256)).astype(np.float32)
# Add a moving bright spot
for t in range(200):
    y = 128 + int(60 * np.sin(t * 0.05))
    x = 128 + int(60 * np.cos(t * 0.05))
    stack[t, max(0,y-5):y+5, max(0,x-5):x+5] += 100

w = Window(stack, name='Video Export Demo')

# Launch the video exporter plugin GUI
from videoExporter import videoExporter
videoExporter.gui()

# The Video Exporter panel will open.
# Steps:
# 1. Draw an ROI on the image for a zoom view (optional)
# 2. Configure timestamp and scale bar settings
# 3. Set the output frame range
# 4. Click 'Export' and choose a save location
