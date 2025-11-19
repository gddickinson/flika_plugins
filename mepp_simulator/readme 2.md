# MEPP Simulator - FLIKA Plugin

A FLIKA plugin for simulating Miniature Endplate Potential (MEPP) events in electrophysiological recordings. This plugin generates synthetic MEPP traces based on the mathematical model described in Segal et al. Biophys J, 1985.

## Features

- Generate synthetic MEPP traces with customizable parameters
- Simulate noise and baseline fluctuations
- Visualize individual MEPP waveforms
- Export MEPP event timestamps
- Interactive real-time trace visualization using PyQtGraph
- Configurable MEPP amplitude and duration distributions

## Installation

### Prerequisites

The plugin requires the following Python dependencies:
- skimage
- leastsqbound
- matplotlib
- PyOpenGL
- openpyxl
- FLIKA (version >= 0.1.0)

### Installing the Plugin

1. Clone this repository into your FLIKA plugins directory:
```bash
cd /path/to/flika/plugins
git clone https://github.com/yourusername/mepp_simulator.git
```

2. Restart FLIKA to load the plugin

## Usage

1. Launch FLIKA
2. Navigate to `Plugins > MEPP Simulator > Simulate MEPP`
3. Configure the simulation parameters:
   - Recording length
   - Start time
   - MEPP duration (mean and standard deviation)
   - Rise time constant
   - Decay time constant
   - MEPP amplitude (mean and standard deviation)
   - Exponential distribution mean (for event timing)
   - Baseline level
   - Noise sigma

4. Use the interface buttons to:
   - Generate a new trace
   - Plot single MEPP waveform
   - Export MEPP event times
   - View histogram of MEPP occurrence times

## Theory

The plugin implements MEPP simulation based on the following model:

- MEPPs are modeled as unitary events with amplitude h and time course F(t)
- The time course follows: F(t) = exp(-t/dT) - exp(-t/rT)
  - dT = decay time constant
  - rT = rise time constant
- MEPP timing follows an exponential distribution
- MEPP amplitudes follow a normal distribution
- Baseline noise follows a normal distribution

## Version History

Current Version: 2021.02.06

## Author

George Dickinson

## License

MIT

## References

Segal et al. Biophys J, 1985. MINIATURE ENDPLATE POTENTIAL FREQUENCY AND AMPLITUDE DETERMINED BY AN EXTENSION OF CAMPBELL'S THEOREM

## Contributing

Created for the Parker Lab, UCI