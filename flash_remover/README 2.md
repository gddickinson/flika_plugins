# Flash Remover FLIKA Plugin

A plugin for [FLIKA](https://github.com/flika-org/flika) that detects and removes flash artifacts from imaging data using either linear interpolation or noise-based scaling methods.

## Overview

The Flash Remover plugin provides tools to:
- Automatically detect flash artifacts in imaging data
- Remove flash artifacts using multiple methods
- Add realistic noise to corrected regions
- Handle both manual and automatic flash detection

## Features

- Automatic flash detection using moving averages
- Multiple flash removal methods:
  - Linear interpolation with optional noise addition
  - Noise-based scaling
- ROI-based or full-frame analysis
- Interactive visualization of detection results
- Customizable parameters for detection and removal

## Requirements

- [FLIKA](https://github.com/flika-org/flika) version 0.2.23 or higher
- Python 3.x
- Dependencies:
  - numpy
  - scipy
  - PyQt5
  - PyQtGraph
  - tqdm

## Installation

1. Install dependencies:
```bash
pip install numpy scipy tqdm
```

2. Copy this plugin to your FLIKA plugins directory:
```bash
~/.FLIKA/plugins/
```

## Usage

### Basic Workflow

1. Launch FLIKA
2. Load your imaging data
3. Go to Plugins → Flash Remover
4. Configure settings and parameters
5. Click "Remove Flash"

### Methods

#### 1. Linear Interpolation
- Interpolates values between pre and post-flash frames
- Optional noise addition based on pre-flash signal
- Best for simple flash artifacts

#### 2. Noise-based Scaling
- Scales flash artifact based on noise characteristics
- Uses ratio of flash to baseline noise
- Better for complex flash patterns

### Parameters

#### Flash Detection
- **Flash Range Start/End**: Search range for flash artifact
- **Moving Average Window Size**: For smoothing during detection
- **Manual Flash Start/End**: Override automatic detection

#### Processing Options
- **Add Noise**: Add realistic noise to interpolated regions
- **User Defined ROI**: Use specific region for detection
- **Plot Detection Result**: Visualize detection process

## Tips

1. **Method Selection**:
   - Use linear interpolation for simple flashes
   - Use noise scaling for complex artifacts
   - Try both methods to determine best results

2. **Detection Parameters**:
   - Start with default window size
   - Adjust range to focus on flash region
   - Use ROI for noisy data

3. **Verification**:
   - Use visualization option to verify detection
   - Check both pre and post removal data
   - Verify noise patterns match surrounding frames

## Known Issues

- Noise generation can be slow for large datasets
- Edge artifacts possible with noise scaling method
- ROI selection required for some noisy datasets

## Future Improvements

- [ ] Faster noise generation
- [ ] Additional removal methods
- [ ] Batch processing
- [ ] Advanced noise modeling
- [ ] Multi-flash support

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- George Dickinson (george.dickinson@gmail.com)

## Version History

- 2021.03.05: Initial release

## Technical Details

### Flash Detection Algorithm

```python
def moving_average(x, n=windowSize):
    return np.convolve(x, np.ones((n,))/n, mode='valid')
```

The plugin uses a moving average to:
1. Smooth the time series
2. Detect intensity peaks
3. Identify flash boundaries

### Flash Removal Methods

#### Linear Interpolation
```python
points = range(0,n)
xp = [0,n]
fp = [flash[0,row,col],flash[-1,row,col]]
interp_data = np.interp(points,xp,fp)
```

#### Noise-based Scaling
```python
baseNoise = np.std(img[flashStart-102:flashStart-2])
flashNoise = np.std(flash[2:12])
noiseRatio = flashNoise/baseNoise
flashReplace = np.divide(img[flashStart+1:flashEnd-1], noiseRatio)
```

## Acknowledgments

- FLIKA development team
- Based on common flash removal techniques in microscopy

## Citation

If you use this software in your research, please cite:
[Citation information to be added]
