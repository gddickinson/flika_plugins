# OnTheFly - Light Sheet Analysis FLIKA Plugin

A FLIKA plugin for real-time analysis and visualization of light sheet microscopy data. This plugin enables continuous monitoring of acquisition, on-the-fly reconstruction, and multi-view visualization of volumetric data.

## Features

### Real-time Analysis
- Continuous monitoring of data acquisition folder
- On-the-fly reconstruction of volumetric data
- Support for single and dual-channel imaging
- Real-time display of acquired data
- Customizable update rates for display and analysis

### Volume Reconstruction
- Configurable shear transformation for light sheet data
- Support for triangle scanning mode
- Adjustable shift factor and theta angle
- Optional interpolation for enhanced resolution
- Z-scan compatibility

### Multi-view Visualization
- Top view, X-view, and Y-view projections
- Interactive volume slider
- Real-time volume updates
- Customizable display parameters
- Batch processing capabilities

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- Python dependencies:
  - numpy
  - scipy
  - PyQt5
  - pyqtgraph
  - skimage
  - tifffile

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/onTheFly.git
```

2. Restart FLIKA to load the plugin

## Usage

### Main Interface
1. Launch FLIKA
2. Navigate to `Plugins > OnTheFly > OnTheFly`
3. Configure parameters:
   - Recording and export folder locations
   - Batch size
   - Update rates
   - Number of steps per volume
   - Display slice
   - Shift factor and theta angle
   - Scanning mode options

### Volume Viewer
- Click 'Start Viewer' to launch volume visualization
- Use slider to navigate through volumes
- Switch between different view projections (Top, X, Y)
- Update visualization in real-time
- Auto-leveling available for contrast adjustment

### Parameters

#### Basic Settings
- `Batch Size`: Number of volumes to process in each batch
- `Update Rate`: Refresh rate for display (seconds)
- `Number of Steps Per Volume`: Z-steps in each volume
- `Display Slice`: Which slice to show in main display

#### Advanced Settings
- `Shift Factor`: Factor for shear transformation
- `Theta`: Angle of light sheet
- `Triangle Scan`: Enable for triangle scanning mode
- `Interpolate`: Enable for interpolated reconstruction
- `Z Scan`: Enable for Z scanning mode
- `Number of Channels`: Support for 1 or 2 channels

## Implementation Notes

The plugin consists of three main components:
1. `onTheFly.py`: Main interface and real-time processing
2. `volumeViewConverter.py`: Volume reconstruction and transformation
3. `volumeSlider_display.py`: Volume visualization and navigation

### Data Flow
1. Monitor acquisition folder for new files
2. Load and process new volumes on arrival
3. Perform shear transformation
4. Update displays and volume viewer
5. Export processed data to specified folder

## Version History

Current Version: 2020.05.23

## Author

George Dickinson

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- The plugin assumes a specific file naming convention for volumes
- Large datasets may require significant system memory
- Performance depends on update rate settings and system capabilities
- Real-time processing may be affected by file system access speeds