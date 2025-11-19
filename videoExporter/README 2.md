# VideoExporter - FLIKA Plugin

A FLIKA plugin for exporting image stacks as videos with customizable timestamps, scale bars, and ROI zoom capabilities. This plugin enables high-quality video creation from microscopy data with professional annotations and multiple viewing options.

## Features

### Video Export
- MP4 format output
- Customizable frame rate
- Multiple view export options
- FFmpeg integration
- Batch processing support

### ROI Zoom
- Interactive ROI selection
- Synchronized zoom view
- Real-time updates
- Independent scale control
- Split-view export

### Annotations
- Customizable timestamps
  - Size control
  - Format options (ms/s)
  - Position adjustment
  - Time correction
- Scale bars
  - Multiple units (nm/µm)
  - Position options
  - Size control
  - Color selection

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- FFmpeg
- Python dependencies:
  - PyQt5
  - pyqtgraph
  - numpy
  - tqdm

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/videoExporter.git
```

2. Restart FLIKA to load the plugin

## Usage

### Basic Operation
1. Launch FLIKA
2. Load your image stack
3. Navigate to `Plugins > VideoExporter`
4. Configure:
   - Pixel size
   - Frame length
   - Export options

### ROI Video Export
1. Draw ROI on main window
2. Enable ROI Video Exporter
3. Configure zoom view:
   - Scale bar
   - Timestamp
   - Frame range
4. Click Export to save

### Parameters

#### Basic Settings
- Pixel Size: Physical size in microns
- Frame Length: Time in milliseconds
- Frame Rate: Export speed

#### Timestamp Options
- Size: Font size
- Unit: ms or seconds
- Show/Hide
- Time Correction
- Position

#### Scale Bar
- Width: Physical size
- Units: nm or µm
- Font Size
- Color
- Position
- Background

## Implementation Details

### Video Generation
- Multi-thread processing
- FFmpeg optimization
- Memory management
- Progress tracking

### Display Integration
- PyQtGraph for visualization
- Real-time preview
- Synchronized views
- Dynamic updates

### Export Process
1. Temporary file creation
2. Frame-by-frame export
3. FFmpeg encoding
4. Multi-view compilation

## Technical Features

### ROI Handling
- Real-time updates
- Size preservation
- Position tracking
- Zoom synchronization

### Time Management
- Frame-accurate timing
- Time correction support
- Multiple time formats
- Flexible display options

## Version History

Current Version: 2023.01.19

## Author

George Dickinson (george.dickinson@gmail.com)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- FFmpeg must be installed and accessible
- Large stacks may require significant processing time
- Multiple views increase export time
- Temporary files are automatically cleaned up
- Export path must have write permissions