# Puff Simulator - FLIKA Plugin

A FLIKA plugin for simulating calcium puff events in imaging data. This plugin enables the generation of synthetic calcium signals with customizable parameters, supporting both single events and complex spatio-temporal patterns.

## Features

### Signal Generation
- Single puff simulation
- Multiple puff site generation
- Random temporal distribution
- Gaussian spatial profile
- Customizable event parameters

### Puff Parameters
- Amplitude control
- Spatial sigma (width)
- Duration (fixed or random)
- Start time selection
- Position specification 
- Multi-site patterns

### Distribution Options
- Exponential temporal distribution
- Random site placement
- Sequential or overlapping events
- Multiple sites within ROIs
- Minimum distance constraints

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- Python dependencies:
  - numpy
  - pandas
  - PyQt5
  - pyqtgraph
  - matplotlib

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/puff_simulator.git
```

2. Restart FLIKA to load the plugin

## Usage

### Basic Operation
1. Launch FLIKA
2. Load your base image stack
3. Navigate to `Plugins > Puff Simulator`
4. Configure parameters:
   - Duration
   - Amplitude
   - Position (x,y)
   - Sigma (spatial spread)
   - Start frame

### Single Puff Simulation
1. Set puff parameters
2. Use 'Preview Puff' to visualize
3. Click 'Add Puff' to insert into stack
4. Adjust parameters as needed

### Multiple Puff Generation
1. Draw ROI for puff locations
2. Set distribution parameters:
   - Mean exponential time
   - Number of sites
   - Duration distribution
3. Choose sequential or overlapping events
4. Click 'Add Puffs inside ROI'

### Parameters

#### Signal Parameters
- Duration: Number of frames for each event
- Amplitude: Peak intensity of puff
- Sigma: Spatial spread (Gaussian width)
- Position: X,Y coordinates
- Start Frame: Event onset time

#### Distribution Parameters
- Mean Exponential: Average time between events
- Random Duration: Enable variable event durations
- Mean Duration: Average event length
- Number of Sites: Total puff locations
- Sequential Events: Prevent temporal overlap

## Implementation Details

### Signal Generation
- 2D Gaussian spatial profile
- Temporal rectangular window
- Optional random duration sampling
- Intensity scaling for sub-frame events

### Site Distribution
- Random site placement within ROI
- Minimum distance enforcement
- Boundary checking
- Conflict avoidance

### Data Export
- Event timing export
- Duration records
- Site coordinates
- CSV format output

## Technical Notes

### Performance
- Efficient array operations
- Memory management for large stacks
- Edge case handling
- Stack boundary protection

### ROI Integration
- ROI-based site selection
- Coordinate transformation
- Boundary validation
- Multi-site spacing

## Version History

Current Version: 2020.10.02

## Author

George Dickinson (george.dickinson@gmail.com)
Adapted from Kyle Ellefsen's work

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Preview function helps optimize parameters
- Large sigmas may cause edge effects
- Random durations affect effective amplitudes
- Export data for analysis reproducibility
- ROI size affects possible site density