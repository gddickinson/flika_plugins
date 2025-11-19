# OverlayMultipleRecordings - FLIKA Plugin

A FLIKA plugin designed to work with the translateAndScale plugin for analyzing and visualizing multiple localization microscopy recordings. This plugin enables the overlay of multiple localization files that have been centered and scaled, with capabilities for generating heat maps based on point counts or other data values.

## Features

### Data Visualization
- Overlay multiple localization files on a single image
- Interactive point display with selection capabilities
- Real-time visualization updates
- Support for both tracked and untracked localizations

### Heat Map Generation
- Generate heat maps from:
  - Point counts
  - Any numeric column in the data
- Multiple statistical methods:
  - Mean
  - Median
  - Maximum
  - Minimum
- Optional binning by experiment
- Adjustable bin sizes

### Data Manipulation
- Interactive controls for data positioning:
  - Translation (Up, Down, Left, Right)
  - Rotation (Clockwise, Counter-clockwise)
  - Adjustable step sizes
  - Multiplier for large movements
- Center data on image option
- Save transformed positions

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- Python dependencies:
  - numpy
  - pandas
  - scipy
  - scikit-image
  - PyQt5
  - pyqtgraph
  - matplotlib
  - tqdm
  - numba (optional, for performance)

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/overlayMultipleRecordings.git
```

2. Restart FLIKA to load the plugin

## Usage

### Basic Operation
1. Launch FLIKA
2. Load your image stack
3. Navigate to `Plugins > OverlayMultipleRecordings`
4. Select:
   - Image Window
   - Data Folder (containing transformed localization files)
   - Center Data option
   - Number of bins for heat maps

### Control Panel
The control panel provides:
- Movement controls
- Rotation controls
- Step size adjustment
- File selection
- Heat map value selection
- Statistical method selection

### Heat Map Generation
1. Select data column for heat map (optional)
2. Choose statistical method
3. Set binning options
4. Click 'Make heatmap'

### Data Export
- Click 'Save New Positions' to save transformed coordinates
- Exports to CSV in the original data folder with '_newPos_transform.csv' suffix

## Input Data Format
The plugin expects CSV files with the following columns:
- x_transformed, y_transformed: Localization coordinates
- track_number (optional): For tracked particles
- Additional columns can be used for heat map generation

## Implementation Details

### Performance Optimization
- Uses numba acceleration when available
- Efficient data structures using pandas DataFrames
- Optimized point rendering with pyqtgraph

### Data Processing
- Automatic center calculation
- Statistical binning for heat maps
- Support for large datasets

## Version History

Current Version: 2024.01.29

## Author

George Dickinson (george.dickinson@gmail.com)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Large datasets may impact performance
- Heat maps can only be generated from numeric data columns
- Save transformations before closing to preserve changes
- Designed to work with output from translateAndScale plugin