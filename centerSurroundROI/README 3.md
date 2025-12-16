# CenterSurroundROI FLIKA Plugin

A plugin for [FLIKA](https://github.com/flika-org/flika) that creates linked center and surround ROIs (Regions of Interest) for analyzing image data. This plugin is particularly useful for analyzing localized signals with background/surrounding region subtraction.

## Features

- Create paired center and surround ROIs
- Automatic surround region updates when center ROI is moved
- Customizable ROI dimensions
- Real-time trace visualization
- Background subtraction using surround region
- Interactive ROI manipulation
- Trace export capabilities

## Requirements

- [FLIKA](https://github.com/flika-org/flika) version 0.2.23 or higher
- Python 3.x
- PyQt5
- NumPy
- SciPy
- PyQtGraph

## Installation

1. Ensure you have FLIKA installed
2. Copy this plugin to your FLIKA plugins directory:
```bash
~/.FLIKA/plugins/
```

## Usage

1. Launch FLIKA
2. Go to Plugins → centerSurroundROI
3. Select your image window in the dropdown
4. Configure ROI settings:
   - **Set Surround Width**: Width of the surrounding region
   - **Set Center Width**: Width of the center ROI
   - **Set Center Height**: Height of the center ROI
   - **Set Center Size**: Set square dimensions for center ROI
5. Click "Start" to create ROIs

### ROI Controls

- Click and drag ROIs to move them
- Surround region automatically updates with center ROI movement
- Use sliders to adjust dimensions
- Real-time trace updates in plot window

### Trace Analysis

The plugin provides three trace windows:
- Center ROI trace
- Surround ROI trace
- Subtracted trace (Center - Surround)

## Settings

The plugin saves your preferences in FLIKA's settings:
- Surround width
- Center width
- Center height
- Center size

These settings will persist between sessions.

## Tips

1. **ROI Placement**:
   - Position the center ROI over your region of interest
   - The surround region will automatically maintain proper spacing

2. **Size Adjustments**:
   - Use the width slider to adjust surround region size
   - Adjust center ROI dimensions independently or use square sizing

3. **Trace Analysis**:
   - Monitor live updates as you adjust ROI positions
   - Use subtracted trace for background-corrected signals

## Known Issues

- ROIs must be manually deleted before starting new analysis
- Multiple ROI pairs in the same window may cause performance issues

## Future Improvements

- [ ] Add ROI deletion button
- [ ] Support for multiple ROI pairs
- [ ] Additional trace analysis tools
- [ ] Export options for analyzed data
- [ ] Batch processing capabilities

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- George Dickinson

## Version History

- 2021.01.08: Initial release

## Acknowledgments

- Built using the FLIKA plugin template
- Thanks to the FLIKA development team
