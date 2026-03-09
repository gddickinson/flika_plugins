"""
Scaled Average Subtract Plugin - Example Usage
Removes global calcium response to isolate local events.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create synthetic calcium data with global wave + local puffs
n_frames, h, w = 200, 64, 64
test_data = np.random.poisson(50, (n_frames, h, w)).astype(np.float32)

# Add a global calcium transient (affects all pixels)
global_signal = np.zeros(n_frames)
global_signal[50:150] = 200 * np.exp(-np.arange(100) / 30.0)
test_data += global_signal[:, np.newaxis, np.newaxis]

# Add local puffs that we want to isolate
yy, xx = np.ogrid[-32:32, -32:32]
for t0, y0, x0 in [(60, 20, 30), (80, 45, 15), (110, 35, 50)]:
    spot = 300 * np.exp(-((xx - (x0-32))**2 + (yy - (y0-32))**2) / (2 * 4**2))
    for dt in range(10):
        if t0 + dt < n_frames:
            test_data[t0 + dt] += spot * np.exp(-dt / 3.0)

win = Window(test_data, name='Global + Local Calcium')

# Launch scaled average subtract
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('scaledAverageSubtract')

from plugins.scaledAverageSubtract import scaledAverageSubtract
scaledAverageSubtract.gui()

# The GUI allows you to:
# - Select the window to process
# - Set window size (rolling average for peak detection)
# - Set average size (frames around peak for template)
# - Click OK to generate the subtracted stack
# The result isolates local puffs by removing the global trend
