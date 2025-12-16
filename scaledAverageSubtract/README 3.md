# ScaledAverageSubtract - FLIKA Plugin

A FLIKA plugin for analyzing calcium imaging data by performing scaled average subtraction. This plugin automatically detects response peaks and subtracts a scaled average image from the imaging stack, helping to isolate calcium signals.

## Features

### Signal Processing
- Automatic peak detection
- Rolling average calculation
- Scaled image subtraction
- ROI-based analysis

### Analysis Tools
- Moving average window
- Peak frame detection 
- Signal scaling (0-1)
- Trace visualization

### Visualization
- Interactive ROI placement
- Signal trace plotting
- Peak identification
- Real-time analysis updates

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- Python dependencies:
  - numpy
  - PyQt5
  - pyqtgraph

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/scaledAverageSubtract.git
```

2. Restart FLIKA to load the plugin

## Usage

### Basic Operation
1. Launch FLIKA
2. Load your calcium imaging stack
3. Navigate to `Plugins > ScaledAverageSubtract`
4. Configure:
   - Rolling average window size
   - Peak average frame count
   - Select analysis window

### Analysis Process
1. Select or draw ROI:
   - Use existing ROI
   - Auto-generate central ROI
2. Run analysis:
   - Automatic peak detection
   - Average calculation
   - Scale and subtract
3. View results:
   - ROI trace plot
   - Rolling average overlay
   - Peak identification
   - Subtracted stack

### Parameters
- Window Size: Number of frames for rolling average calculation
- Average Size: Number of frames to average around peak
- Analysis Window: Image stack to analyze
- ROI: Region for signal extraction (automatic or user-defined)

## Technical Details

### Signal Processing Steps
1. Extract ROI trace
2. Calculate moving average
3. Detect signal peak
4. Average frames around peak
5. Scale averaged image
6. Subtract from original stack

### Algorithm Features
- Non-zero floor enforcement
- Edge case handling
- Frame boundary protection
- Signal normalization
- Stack-wise operations

### Output
- Subtracted image stack
- Signal trace plot
- Peak identification
- Moving average visualization

## Implementation Notes

### Performance
- Efficient array operations
- Memory-conscious processing
- Stack-based calculations
- Vectorized operations

### ROI Management
- Automatic ROI generation
- Custom ROI support
- Central positioning
- Size adjustment

## Version History

Current Version: 2020.07.11

## Author

George Dickinson (george.dickinson@gmail.com)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Best results with stable baseline signals
- ROI size affects peak detection sensitivity
- Window size influences smoothing level
- Large stacks may require more processing time
- Edge frames are handled specially due to rolling window
- Negative signal values are set to a small positive number