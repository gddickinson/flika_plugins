# FFT_Chunker FLIKA Plugin

A plugin for [FLIKA](https://github.com/flika-org/flika) that performs Fast Fourier Transform (FFT) analysis on time-series data in chunks, with capabilities for averaging and analyzing multiple segments of data.

## Overview

FFT_Chunker provides tools for:
- Analyzing time-series data in defined chunks
- Performing FFT analysis on each chunk
- Computing power spectra
- Averaging results across specified ranges
- Handling multiple ROI traces
- Visualizing results

## Features

- Customizable chunk size for analysis
- Adjustable time step parameters
- Multiple puff range analysis
- Baseline comparison
- Interactive visualization
- Batch processing capabilities
- Log-transformed data export
- Automatic averaging of multiple ROIs

## Requirements

- [FLIKA](https://github.com/flika-org/flika) version 0.2.23 or higher
- Python 3.x
- Dependencies:
  - NumPy
  - SciPy
  - pandas
  - matplotlib
  - PyQt5
  - PyQtGraph

## Installation

1. Ensure you have FLIKA installed
2. Install required dependencies:
```bash
pip install numpy scipy pandas matplotlib pyqtgraph
```
3. Copy this plugin to your FLIKA plugins directory:
```bash
~/.FLIKA/plugins/
```

## Usage

### Basic Workflow

1. Launch FLIKA
2. Go to Plugins → FFT_Chunker
3. Configure analysis parameters:
   - Chunk size
   - Time step
   - Baseline range
   - Puff ranges (up to 4)
4. Load your data file
5. Run analysis

### Parameters

- **Chunk Size**: Number of data points per FFT analysis segment
- **Time Step**: Time interval between data points (seconds)
- **Baseline Range**: Start and stop points for baseline analysis
- **Puff Ranges**: Up to 4 separate ranges for puff analysis
  - Set stop = 0 to ignore a range

### Input Format

- CSV files containing time-series data
- First column should be time/index
- Subsequent columns contain data traces
- Multiple ROI traces supported

### Output Files

1. **Individual Trace Results**: 
   - One file per trace
   - Contains power and frequency for each chunk
   - Log-transformed values

2. **Averaged Results**:
   - Power and frequency averages for each range
   - Log-transformed values
   - Separate file for multi-ROI averages

3. **Visualization**:
   - Time-domain plots
   - Power spectra
   - Averaged results comparison

## File Naming

- Default: `FFT_Chunker_Batch_[timestamp]_[trace#].csv`
- Custom: Set via "Set SaveName" button
- Averaged results: `[name]_AveragedTraces.csv`

## Tips

1. **Data Preparation**:
   - Ensure consistent sampling rate
   - Remove artifacts before analysis
   - Verify column headers

2. **Parameter Selection**:
   - Choose chunk size based on frequency of interest
   - Verify number of chunks is appropriate
   - Use ranges that capture events of interest

3. **Result Analysis**:
   - Check log-transformed plots
   - Compare baseline to puff ranges
   - Verify averaging results

## Known Issues

- Large numbers of chunks (>40) will not auto-plot
- All ranges must be within total number of chunks
- Stop values must be greater than start values

## Future Improvements

- [ ] Additional plotting options
- [ ] Custom chunk overlap
- [ ] Export plot options
- [ ] Additional statistical analysis
- [ ] Batch file processing

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- George Dickinson (george.dickinson@gmail.com)

## Version History

- 2020.06.25: Initial release

## Acknowledgments

- Built using the FLIKA plugin template
- Thanks to the FLIKA development team

## Citation

If you use this software in your research, please cite:
[Citation information to be added]
