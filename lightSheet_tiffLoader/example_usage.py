"""
Light Sheet TIFF Loader Plugin - Example Usage
Loads multi-file TIFF stacks from light sheet microscopes.
"""
from flika import *

# Start flika
start_flika()

# Launch the TIFF loader
from flika.app.plugin_manager import PluginManager
pm = PluginManager()
pm.load_plugin('lightSheet_tiffLoader')

from plugins.lightSheet_tiffLoader import load_tiff
load_tiff.gui()

# The GUI opens a file dialog where you select a TIFF file.
# The loader handles multi-file sequences common to light sheet
# microscope output, assembling them into a single stack.
#
# Supported formats:
# - Single multi-page TIFF files
# - Numbered TIFF sequences from light sheet systems
# - Large files that exceed standard TIFF limits
