"""
Overlay Multiple Recordings Plugin - Example Usage
Overlays and compares multiple recording sessions.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create multiple test recordings that simulate repeated experiments
for i in range(3):
    data = np.random.poisson(50, (100, 64, 64)).astype(np.float32)
    # Add a signal with slightly different timing in each
    for t in range(20 + i*5, 40 + i*5):
        data[t] += 100
    Window(data, name=f'Recording {i+1}')

# Launch overlay multiple recordings
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('overlayMultipleRecordings')

# The plugin provides:
# - Folder selection for batch loading recordings
# - Alignment and overlay of multiple time series
# - Heatmap visualization of combined data
# - Parameter controls for display settings
#
# To use with files:
# 1. Organize recordings in a folder
# 2. Go to Plugins > Overlay Multiple Recordings
# 3. Select the folder containing recordings
# 4. Adjust alignment and display parameters
# 5. View overlaid heatmap and traces
