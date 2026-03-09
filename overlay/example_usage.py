"""
Overlay Plugin - Example Usage
Creates RGB channel overlays from separate flika windows.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create three single-channel images
h, w = 128, 128
n_frames = 20

# Red channel: nucleus (central blob)
red = np.random.poisson(20, (n_frames, h, w)).astype(np.float32)
yy, xx = np.ogrid[-64:64, -64:64]
red += 200 * np.exp(-(xx**2 + yy**2) / (2 * 15**2))

# Green channel: cytoplasm (ring)
green = np.random.poisson(20, (n_frames, h, w)).astype(np.float32)
r = np.sqrt(xx**2 + yy**2)
green += 150 * np.exp(-((r - 30)**2) / (2 * 8**2))

# Blue channel: membrane (outer ring)
blue = np.random.poisson(20, (n_frames, h, w)).astype(np.float32)
blue += 100 * np.exp(-((r - 45)**2) / (2 * 3**2))

w_red = Window(red, name='Red Channel')
w_green = Window(green, name='Green Channel')
w_blue = Window(blue, name='Blue Channel')

# Launch overlay GUI
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('overlay')

from plugins.overlay import overlay
overlay.gui()

# The GUI allows you to:
# - Assign each window to R, G, or B channel
# - Adjust per-channel gamma correction
# - Toggle channel visibility
# - View the composite RGB overlay in real time
