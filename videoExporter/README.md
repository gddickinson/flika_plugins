# Video Exporter

Exports flika image stacks as MP4 videos with optional ROI zoom views, timestamps, and scale bars.

## Features

- **MP4 video export** from any flika image stack via ffmpeg
- **ROI zoom view** -- magnified view of a selected region with independent controls
- **Timestamp overlay** -- configurable elapsed-time display
- **Scale bar overlay** -- calibrated scale bar with customizable size and position
- **Frame range control** -- export a subset of frames
- **Screen recorder** -- standalone utility for screen capture

## Requirements

- [flika](https://github.com/flika-org/flika) >= 0.2.23
- **ffmpeg** -- must be installed and on the system PATH
- **opencv-python** (optional, for screen recorder feature)

## Installation

```bash
# Install ffmpeg (macOS)
brew install ffmpeg

# Install ffmpeg (Linux)
sudo apt install ffmpeg
```

## Usage

1. Open an image stack in flika
2. Go to **Plugins > Video Exporter**
3. Optionally draw an ROI to define a zoom region
4. Configure timestamp, scale bar, and frame range
5. Click **Export** to save the video as MP4

## Programmatic Usage

```python
from flika import *
start_flika()

from flika.process.file_ import open_file
w = open_file('my_stack.tif')

from videoExporter import videoExporter
videoExporter.gui()
```
