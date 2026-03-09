"""
Simulate Puff Plugin - Example Usage
Generates synthetic calcium puff events for testing.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a blank image stack to add puffs to
test_data = np.random.poisson(50, (200, 64, 64)).astype(np.float32)
win = Window(test_data, name='Blank Stack')

# Launch puff simulator
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('simulatePuff')

# Access through Plugins > Simulate Puff
# Simulation modes:
#
# 1. Add Single: click on image to place a puff at that location
# 2. Random Train: generates puffs at one site with Poisson timing
# 3. Multi-site: distributes puff sites randomly within an ROI
# 4. Full Simulation: creates an entire synthetic movie with known ground truth
#
# Parameters:
# - Amplitude: peak intensity of the synthetic puff
# - Sigma: width of the 2D Gaussian (pixels)
# - Duration: number of frames for the puff event
# - Rate: average puffs per second (for random modes)
# - N sites: number of puff sites (multi-site mode)
