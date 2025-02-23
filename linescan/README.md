# FLIKA Linescan Plugin

A FLIKA plugin for creating and analyzing intensity profiles along line ROIs in image data. Supports both grayscale and RGB images with real-time updates and interactive visualization.

## Features

- Interactive line ROI placement and manipulation
- Real-time intensity profile updates
- Support for both grayscale and RGB images
- Multiple integration methods:
  - Mean intensity
  - Minimum intensity
  - Maximum intensity
- Color channel selection for RGB images
- Interactive plot with cursor tracking
- Dynamic updates when changing frames in video data

## Installation

1. Ensure you have FLIKA installed
2. Clone this repository into your FLIKA plugins directory:
   ```bash
   cd ~/.FLIKA/plugins
   git clone https://github.com/yourusername/linescan.git
   ```

## Dependencies

- FLIKA
- NumPy
- PyQt (via qtpy)
- PyQtGraph

## Usage

### Basic Operation

1. Launch FLIKA and load your image
2. Draw a line ROI on your image
3. Open the Linescan plugin
4. Click "Start" to create the intensity profile plot
5. Click "Show Options Menu" to configure plot settings

### Options Menu

The options menu provides several controls:

- Width Integration Method:
  - Mean: Average intensity along the width of the line
  - Min: Minimum intensity along the width
  - Max: Maximum intensity along the width
- Channel Selection (for RGB images):
  - R: Red channel
  - G: Green channel
  - B: Blue channel

### Interactive Features

- The line ROI can be manipulated in real-time:
  - Click and drag endpoints to change position
  - The intensity plot updates automatically
- Plot window features:
  - Cursor tracking displays exact x,y coordinates
  - Crosshair follows cursor position
  - Interactive zoom and pan

## Classes

### PlotWindow
Main class for handling the intensity profile visualization

Methods:
- `initializePlot()`: Creates and configures the plot window
- `getROIdata()`: Extracts intensity data along the ROI
- `setStyle()`: Configures plot appearance
- `update()`: Updates plot with new data
- `exportLineData()`: Returns (x,y) plot data

### OptionsGUI
Handles the configuration interface

Methods:
- `channelSelectionChange()`: Updates selected color channel
- `integrationSelectionChange()`: Updates integration method
- `closeOptions()`: Closes the options window

### Linescan
Main plugin class

Methods:
- `gui()`: Creates the plugin interface
- `linescan()`: Initializes new plot window
- `options()`: Opens options menu
- `getData()`: Returns current plot data

## Data Handling

The plugin processes different image types:

### Grayscale Images
- Direct intensity measurement along line ROI
- Integration methods apply across ROI width

### RGB Images
- Separate handling for each color channel
- Channel selection via options menu
- Integration methods apply per channel

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

George Dickinson

## Acknowledgments

- FLIKA development team
- Based on the FLIKA plugin template
