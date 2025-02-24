# Overlay - FLIKA Plugin

A FLIKA plugin for overlaying and manipulating multiple image stacks with advanced control over visualization parameters. Perfect for multi-channel microscopy data visualization.

## Features

### Image Overlay
- Overlay two image stacks with adjustable opacity
- Independent histogram control for each channel
- Custom composition modes for overlay display
- Support for different color maps (LUTs)

### Gamma Correction
- Real-time gamma correction
- Live preview window
- Adjustable gamma values (0-20)
- Preview functionality for instant feedback

### Display Controls
- Independent histogram controls for each channel
- Custom LUT (Look-Up Table) selection
- Multiple gradient presets:
  - Thermal
  - Flame
  - Yellowy
  - Bipolar
  - Spectrum
  - Cyclic
  - Greyclip
  - Grey

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- Python dependencies:
  - numpy
  - scipy
  - PyQt
  - pyqtgraph
  - numba (optional, for performance)

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/overlay.git
```

2. Restart FLIKA to load the plugin

## Usage

### Basic Operation
1. Launch FLIKA
2. Load your image stacks
3. Navigate to `Plugins > Overlay`
4. Select input windows:
   - CH1 (r): First channel (base image)
   - CH2 (g): Second channel (overlay)
5. Click 'Overlay' to create the composite view

### Gamma Correction
1. Enable 'Gamma Correct' checkbox
2. Adjust gamma value using slider (0-20)
3. Use 'Preview Gamma' to see changes in real-time
4. Apply changes to final overlay

### Advanced Settings
- Independent histogram control for both channels
- Custom LUT selection for enhanced visualization
- Composition mode selection for overlay blending
- Opacity adjustment for overlay channel

## Implementation Notes

### Performance
- Uses numba acceleration when available
- Efficient image manipulation using numpy operations
- Real-time preview updates

### Data Handling
- Support for multi-dimensional image stacks
- Proper memory management for large datasets
- Efficient histogram computation

## Version History

Current Version: Compatible with FLIKA version 0.2.23 and above

## Author

Contact the FLIKA team for support and contributions

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Performance may vary with image size and system capabilities
- Gamma correction preview maintains original data integrity
- Some features require specific FLIKA version compatibility
- For best results, ensure input images are properly normalized