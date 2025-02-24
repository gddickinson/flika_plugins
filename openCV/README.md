# OpenCV FLIKA Plugin

A FLIKA plugin that provides real-time camera control, image processing, and video recording capabilities using OpenCV. This plugin enables live camera viewing with various filters, image capture, and video recording functionality.

## Features

### Camera Control
- Live camera preview
- Image capture
- Video recording with FPS calculation
- Toggle between color and black & white modes
- Adjustable camera settings

### Image Processing Filters
- Canny Edge Detection
- 2D Convolution Filters:
  - Average
  - Smooth
  - Gaussian
  - Median
  - Bilateral
- Image Inversion
- Adaptive Threshold
- Laplacian Edge Detection
- Background Subtraction

### File Operations
- Support for image and video files
- Image rotation and flipping
- Zoom capabilities
- File format conversion

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- OpenCV (cv2)
- Python dependencies:
  - numpy
  - PyQt5
  - pyqtgraph

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/opencv_flika.git
```

2. Restart FLIKA to load the plugin

## Usage

### Camera Interface
1. Launch FLIKA
2. Navigate to the OpenCV plugin
3. Main controls:
   - "Click to Take Picture": Captures current frame
   - "Black & White": Toggles between color and grayscale
   - "Start Recording": Begins video recording
   - "Quit Live Camera": Stops camera feed

### Filter Selection
Use the dropdown menu to select from available filters:
- No Filter
- Canny Filter
- 2D Convolution filters
- Invert
- Adaptive Threshold
- Laplacian Edge
- Background Subtract

### File Operations
The plugin includes a separate interface for file operations:
- Open image/video files
- Convert between formats
- Rotate images (90° clockwise/counterclockwise)
- Flip images (horizontal/vertical)
- Zoom in/out

## Implementation Details

### Camera Module
- Uses OpenCV's VideoCapture for camera access
- Real-time frame processing
- FPS calculation for recordings
- Image format conversion for FLIKA compatibility

### Image Processing
- Filter implementation using OpenCV functions
- Real-time filter application
- Support for both color and grayscale processing
- Background subtraction with history tracking

### File Handling
- Support for various image formats
- Video file processing
- Image transformation tools
- Integration with FLIKA's Window system

## Version History

Current Version: Compatible with OpenCV version displayed at startup

## Author

Contact the FLIKA team for support and contributions

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Camera performance depends on system hardware
- Some filters may impact processing speed
- Background subtraction requires additional computational resources
- File operations support common image and video formats
- Real-time processing performance varies with selected filters