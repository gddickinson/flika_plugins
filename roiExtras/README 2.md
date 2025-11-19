# ROIExtras - FLIKA Plugin

A FLIKA plugin that extends the functionality of Region of Interest (ROI) analysis. This plugin provides additional tools for ROI visualization, histogram analysis, and real-time statistics for microscopy data.

## Features

### ROI Enhancement
- Real-time histogram generation
- Value distribution analysis
- Interactive ROI statistics
- Support for multiple ROI types:
  - Rectangle
  - Line
  - Freehand
  - Rect-line
  - Surround

### Visualization Tools
- Live histogram updates with ROI movement
- Automatic axis scaling
- Pixel value distribution display
- Pixel count tracking

### Analysis Features
- Real-time value distribution
- Automatic bin calculation
- Statistical summary display
- Interactive plot updates

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- Python dependencies:
  - numpy
  - scipy
  - PyQt5
  - pyqtgraph

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/roiExtras.git
```

2. Restart FLIKA to load the plugin

## Usage

### Basic Operation
1. Launch FLIKA
2. Load your image data
3. Navigate to `Plugins > ROIExtras`
4. Select:
   - Active Window
   - Autoscale options
   - Display parameters

### Histogram Analysis
1. Draw an ROI on your image
2. Click 'Start' to begin analysis
3. Histogram will update in real-time as you:
   - Move the ROI
   - Resize the ROI
   - Change frames (for stacks)

### Display Options
- Autoscale X-axis: Automatically adjust histogram x-axis range
- Autoscale Y-axis: Automatically adjust histogram y-axis range
- Custom axis ranges available

## Implementation Details

### ROI Integration
- Seamless integration with FLIKA's ROI system
- Support for all standard ROI types
- Custom ROI event handling
- Multiple ROI support

### Performance Optimization
- Efficient histogram calculation
- Real-time update capability
- Memory-efficient data handling
- Frame-by-frame analysis for stacks

### Display Windows
- Main interface
- Histogram window
- ROI preview window
- Statistics display

## Technical Features

### Histogram Generation
- Dynamic bin calculation
- Real-time value distribution
- Pixel count tracking
- Statistical summary

### ROI Analysis
- Mask generation
- Array region extraction
- Multi-dimensional support
- Stack analysis capability

## Version History

Current Version: 2020.12.04

## Author

George Dickinson

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Histogram updates are linked to ROI movements
- Performance may vary with ROI size and image dimensions
- Multiple ROIs can be analyzed simultaneously
- For stack data, analysis updates with frame changes
- Best performance with smaller ROIs due to real-time updates