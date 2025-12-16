# Locs and Tracks Plotter

An interactive FLIKA plugin for visualizing and analyzing PIEZO1 protein dynamics. This plugin provides comprehensive tools for displaying particle localizations, tracking movement patterns, and analyzing protein behavior.

## Features

### Visualization
- Point plotting with customizable appearance
- Track visualization with color coding
- Interactive track selection
- Background subtraction
- Intensity scaling
- Multiple plot types:
  - Single track analysis
  - All tracks overview
  - Flower plots
  - ROI-specific analysis

### Analysis Tools
- Track statistics calculation
- Diffusion analysis
- Nearest neighbor detection
- Intensity measurements
- Movement pattern analysis

## Installation

1. Copy plugin to FLIKA plugins directory:
```bash
cp -r locsAndTracksPlotter ~/.FLIKA/plugins/
```

2. Required dependencies:
```bash
pip install numpy pandas scipy scikit-image trackpy
```

## Data Requirements

Input files should be CSV format with the following columns:
- `frame`: Frame number (int)
- `x`, `y`: Particle coordinates (pixels or nm)
- `track_number`: Track identifier (int)
- `intensity`: Particle intensity (optional)

Additional columns for analysis:
- `SVM`: Classification values
- `radius_gyration`: Motion metrics
- `asymmetry`, `skewness`, `kurtosis`: Shape metrics
- `velocity`, `direction`: Movement metrics

## Usage

### Basic Operation

1. Launch FLIKA and load image data:
```python
import flika
flika.start_flika()
window = open_file('my_data.tif')
```

2. Initialize plugin:
```python
from flika.plugins.locsAndTracksPlotter import LocsAndTracksPlotter
plotter = LocsAndTracksPlotter()
```

3. Load tracking data:
```python
plotter.loadData('tracking_results.csv')
```

### Visualization Options

#### Point Display
```python
# Configure point appearance
plotter.trackPlotOptions.pointSize_selector.setValue(5)
plotter.trackPlotOptions.pointColour_Box.setValue('green')

# Plot points
plotter.plotPointData()
```

#### Track Display
```python
# Set track appearance
plotter.trackPlotOptions.lineSize_selector.setValue(2)
plotter.trackPlotOptions.trackDefaultColour_Box.setValue('blue')

# Plot tracks
plotter.plotTrackData()
```

#### Color Coding
```python
# Color by property
plotter.trackPlotOptions.trackColourCol_Box.setValue('velocity')
plotter.trackPlotOptions.colourMap_Box.setValue('viridis')
```

### Analysis Tools

#### Track Selection
- `T`: Select track under cursor
- `R`: Select tracks within ROI

#### Filtering
```python
# Filter by property
plotter.filterOptionsWindow.filterCol_Box.setValue('intensity')
plotter.filterOptionsWindow.filterOp_Box.setValue('>')
plotter.filterOptionsWindow.filterValue_Box.setText('100')
plotter.filterData()
```

#### Statistics
```python
# Calculate track statistics
plotter.createStatsDFs()
```

## GUI Components

### Main Window
- Point/Track display controls
- File loading interface
- Visualization options

### Track Plot Options
- Point size and color
- Line thickness
- Color mapping
- Background subtraction

### Filter Options
- Property-based filtering
- ROI-based selection
- Sequential filtering

### Analysis Windows
- Track statistics
- Diffusion plots
- Intensity profiles
- Movement analysis

## Configuration

### Display Parameters
```python
settings = {
    'pixelSize': 108,  # nm/pixel
    'pointSize': 5,
    'lineWidth': 2,
    'colorMap': 'viridis',
    'backgroundSubtract': True
}
```

### Analysis Parameters
```python
analysis_settings = {
    'minTrackLength': 6,
    'maxGapSize': 2,
    'maxLinkingDistance': 5,
    'intensityThreshold': 100
}
```

## Output Files

- `*_tracks.csv`: Processed tracking data
- `*_stats.csv`: Track statistics
- `*_filtered.csv`: Filtered results

## Keyboard Shortcuts

- `T`: Select track
- `R`: ROI selection
- `Esc`: Clear selection
- `Space`: Toggle point display

## Tips

1. Data Preparation:
   - Ensure consistent frame numbering
   - Check coordinate system
   - Validate track IDs

2. Performance:
   - Limit points displayed
   - Use appropriate point sizes
   - Consider ROI selection for large datasets

3. Analysis:
   - Start with unfiltered data
   - Use sequential filtering
   - Validate results visually

## Troubleshooting

Common issues and solutions:

1. Display Problems:
   - Check data formatting
   - Verify coordinate ranges
   - Adjust display parameters

2. Performance Issues:
   - Reduce point count
   - Use ROI selection
   - Close unused windows

3. Analysis Errors:
   - Validate input data
   - Check parameter ranges
   - Ensure sufficient track length

## Contributing

Contributions welcome! Please:
1. Fork repository
2. Create feature branch
3. Test thoroughly
4. Submit pull request

## Citation

When using this plugin, please cite:
```
Bertaccini et al. (2023). PIEZO1-HaloTag hiPSCs: Bridging Molecular, 
Cellular and Tissue Imaging. bioRxiv 2023.12.22.573117
```

## Contact

For support:
- Open GitHub issue
- Contact authors via paper

## License

MIT License - See LICENSE file for details
