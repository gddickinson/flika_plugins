# -*- coding: utf-8 -*-
"""
Timestamp Plugin - Example Usage
=================================
Demonstrates how to add a timestamp overlay to a flika window.
"""

from flika import *
start_flika()

import numpy as np
from flika.process.file_ import open_file

# Create a synthetic time-series stack (100 frames, 128x128)
stack = np.random.normal(100, 10, (100, 128, 128)).astype(np.float32)
# Add a moving bright spot
for t in range(100):
    y, x = 64 + int(20 * np.sin(t * 0.1)), 64 + int(20 * np.cos(t * 0.1))
    stack[t, max(0,y-3):y+3, max(0,x-3):x+3] += 50

from flika.window import Window
w = Window(stack, name='Time Series Example')

# Set the frame rate (important for accurate timestamps)
import flika.global_vars as g
g.settings['internal_data_rate'] = 30.0  # 30 fps

# Launch the timestamp plugin
from timestamp import timestamp
timestamp.gui()

# The timestamp will now update as you scroll through frames
# Use Left/Right arrow keys or the slider to navigate
