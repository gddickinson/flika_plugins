# FLIKA File Loader Plugin

An extended file loader plugin for [FLIKA](https://github.com/flika-org/flika) that adds support for additional microscopy file formats, including Nikon ND2 files and other Bio-Formats compatible formats.

## Overview

The File Loader plugin extends FLIKA's file loading capabilities by:
- Adding support for ND2 files
- Implementing Bio-Formats compatibility
- Handling multi-dimensional microscopy data
- Supporting various image formats

## Features

- Load Nikon ND2 files
- Support for Bio-Formats compatible formats
- Handle multi-dimensional data (time series, z-stacks, channels)
- Automatic dimension ordering
- Metadata extraction
- Memory-efficient loading

## Requirements

- [FLIKA](https://github.com/flika-org/flika) version 0.2.23 or higher
- Python 3.x
- Java Runtime Environment (JRE)
- Java Development Kit (JDK)
- Dependencies:
  - javabridge
  - python-bioformats
  - numpy
  - scipy
  - PyQt5

## Installation

1. Install Java dependencies:
   - Install Java Runtime Environment (JRE)
   - Install Java Development Kit (JDK)

2. Install Python dependencies:
```bash
pip install javabridge python-bioformats numpy scipy
```

3. Copy this plugin to your FLIKA plugins directory:
```bash
~/.FLIKA/plugins/
```

## Usage

### Loading Files

1. Launch FLIKA
2. Go to File → Load File
3. Select your file
4. The plugin will automatically detect the file type and use the appropriate loader

### Supported File Types

- `.nd2` - Nikon ND2 files
- `.tif`, `.stk`, `.tiff`, `.ome` - TIFF formats
- `.jpg`, `.png` - Standard image formats
- Other Bio-Formats compatible formats

### ND2 File Loading

The plugin handles ND2 files by:
1. Detecting multi-dimensional data
2. Properly organizing dimensions (time, channels, z-stacks)
3. Loading metadata
4. Creating appropriate FLIKA windows

## Bio-Formats Integration

The plugin uses Bio-Formats to:
- Read complex microscopy formats
- Extract metadata
- Handle multi-dimensional data
- Parse image series
- Support various pixel types

### Dimension Ordering

By default, dimensions are ordered as:
- T: Time
- Z: Z-stack
- Y: Y-dimension
- X: X-dimension
- C: Channels

## Memory Management 

- Java Virtual Machine (JVM) management
- Efficient loading of large datasets
- Automatic memory cleanup
- Configurable heap size

## Functions

### Main Functions

- `load_file_gui()`: Main file loading interface
- `read_image_series()`: Read multi-dimensional image data
- `parse_xml_metadata()`: Extract metadata from files
- `start()`: Initialize Java Virtual Machine
- `done()`: Cleanup Java Virtual Machine

### Data Handling

- Automatic file type detection
- Multi-dimensional array organization
- Metadata parsing and storage
- Memory-efficient data loading

## Error Handling

The plugin includes error handling for:
- Unsupported file formats
- Memory limitations
- JVM issues
- File reading errors
- Dimension mismatches

## Known Issues

- JVM must be properly configured
- Large files may require increased memory allocation
- Some complex multi-dimensional formats may have limitations
- JVM cannot be restarted once killed

## Future Improvements

- [ ] Additional file format support
- [ ] Improved memory management
- [ ] Better metadata handling
- [ ] Progress bar for large files
- [ ] Preview capability
- [ ] Batch processing support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- George Dickinson

## Version History

- 2020.06.20: Initial release

## Acknowledgments

- Bio-Formats team for their excellent library
- FLIKA development team
- Based on work by Lee Kamentsky

## Citation

If you use this software in your research, please cite:
- FLIKA
- Bio-Formats
[Citation information to be added]

## Technical Support

For issues related to:
- JVM configuration
- File loading problems
- Memory issues
- Format compatibility

Please create an issue in the repository or contact the development team.
