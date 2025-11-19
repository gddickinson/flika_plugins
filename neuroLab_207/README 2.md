# NeuroLab - FLIKA Plugin

A FLIKA plugin for generating and analyzing electrophysiological data, specifically focusing on Miniature Endplate Potentials (MEPPs) and Endplate Potentials (EPPs). This plugin provides tools for simulating neurophysiological events with customizable parameters based on established statistical distributions.

## Features

### MEPP (Miniature Endplate Potential) Generation
- Generate MEPP amplitudes from Gaussian distributions
- Simulate number of MEPPs per interval using Poisson distributions
- Model intervals between MEPPs using exponential decay functions
- Visualize distributions through interactive histograms

### EPP (Endplate Potential) Generation
- Generate EPP quantal content using Poisson distributions
- Simulate EPP amplitudes using Gaussian distributions
- Model EPP amplitudes based on quantal content
- Visualize all distributions through interactive plots

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- Python libraries:
  - numpy
  - matplotlib
  - PyQt
  - pyqtgraph
  - pandas

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd /path/to/flika/plugins
git clone https://github.com/yourusername/neuroLab.git
```

2. Restart FLIKA to load the plugin

## Usage

1. Launch FLIKA
2. Navigate to `Plugins > NeuroLab > neuroLab`
3. The interface provides two main sections:

### MEPP Parameters
- Amplitude mean and standard deviation
- Number of MEPPs per interval
- Interval time constant
- Sample size for each distribution

### EPP Parameters
- Quantal content mean
- Amplitude mean and standard deviation
- Amplitude by quanta parameters
- Sample size for each distribution

### Workflow
1. Set desired parameters for either MEPP or EPP data
2. Click 'Generate MEPP Data' or 'Generate EPP Data'
3. Review the generated histograms
4. Set export path using 'Select Folder'
5. Save data using 'Save MEPP Data' or 'Save EPP Data'

## Output Files
The plugin generates CSV files containing:
- MEPP amplitudes
- Number of MEPPs per interval
- MEPP intervals
- EPP quanta
- EPP amplitudes
- EPP amplitudes by quanta

## Version History

Current Version: 2020.05.23

## Author

George Dickinson (george.dickinson@gmail.com)

## License

MIT License

## References

The statistical models used in this plugin are based on established biophysical principles in neuroscience, particularly relating to synaptic transmission and quantal analysis.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- If export button appears unresponsive, try clicking again
- Always set export path before attempting to save data
- Large sample sizes (>1,000,000) may impact performance