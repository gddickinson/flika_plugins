# FLIKA TIFF/CZI Loader Plugin

A specialized plugin for FLIKA that handles loading and processing of TIFF and CZI (Carl Zeiss Image) files, with particular support for lightsheet microscopy data containing multiple channels and volumes.

## Features

- Support for multiple file formats:
  - TIFF (.tif, .tiff, .stk)
  - Carl Zeiss Image files (.czi)
- Handles complex microscopy data structures:
  - Multi-channel imaging
  - Multi-volume data
  - Time series
  - Z-stacks
- Automatic axis detection and reorganization
- Channel separation for multi-channel data

## Installation

1. Make sure you have FLIKA installed
2. Clone this repository into your FLIKA plugins directory:
   ```bash
   cd ~/.FLIKA/plugins
   git clone https://github.com/yourusername/lightSheet_tiffLoader.git
   ```

## Dependencies

- FLIKA
- NumPy
- tifffile
- czifile (included)
- Qt (via qtpy)

## Usage

1. Launch FLIKA
2. Go to the plugins menu
3. Select "Load Tiff" from the tiffLoader submenu
4. Choose your TIFF or CZI file in the file dialog

The plugin will automatically detect the file type and data structure, then load it appropriately.

### Supported Data Structures

The plugin can handle the following data arrangements:

1. Single channel, multi-volume:
   - Axes: time, depth, height, width
   - Output: Reshaped array combining time and depth dimensions

2. Single channel, single-volume:
   - Axes: series/time, height, width
   - Output: Standard 3D array

3. Multi-channel, multi-volume:
   - Axes: time, depth, channel, height, width
   - Output: Separate windows for each channel

4. Multi-channel, single volume:
   - Axes: depth/time, channel, height, width
   - Output: Separate windows for each channel

### Axis Handling

For CZI files, the plugin automatically converts Zeiss axis labels to TIFF format:
- T → time
- C → channel
- Y → height
- X → width
- Z → depth

## API Reference

### Load_tiff Class

Main class handling file loading and processing.

#### Methods

`gui()`
- Launches file selection dialog
- Handles initial file loading process

`openTiff(filename)`
- Parameters:
  - filename (str): Path to the TIFF or CZI file
- Processes file based on extension and data structure
- Automatically detects and handles different axis arrangements
- Creates appropriate FLIKA Window objects for visualization

## File Support Details

### TIFF Files
- Supports standard TIFF formats
- Handles multi-page TIFFs
- Compatible with microscopy-specific TIFF variations

### CZI Files
- Full support for Zeiss microscopy file format
- Automatic handling of metadata
- Proper axis detection and transformation
- Memory-efficient loading of large datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

George Dickinson

## Acknowledgments

- FLIKA development team
- Based on the FLIKA plugin template
