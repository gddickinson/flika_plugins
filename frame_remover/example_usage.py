"""
Frame Remover Plugin - Example Usage
Removes specific frames or periodic frame ranges from image stacks.
"""
from flika import *
import numpy as np

# Start flika
start_flika()

# Create a test stack with 200 frames
test_data = np.random.poisson(100, (200, 64, 64)).astype(np.float32)
# Add frame numbers as intensity so removal is visible
for i in range(200):
    test_data[i] += i

win = Window(test_data, name='200 Frame Stack')

# Launch frame remover GUI
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('frame_remover')

from plugins.frame_remover import frame_remover
frame_remover.gui()

# The GUI allows you to:
# - Select the window to process
# - Set start frame and end frame
# - Set length (number of frames to remove)
# - Set interval (for periodic removal)
# - Click "Remove Frames" to apply
