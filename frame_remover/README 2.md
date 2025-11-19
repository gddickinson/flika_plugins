# Frame Remover FLIKA Plugin

A plugin for [FLIKA](https://github.com/flika-org/flika) that enables systematic removal of frames from image stacks at specified intervals.

## Overview

The Frame Remover plugin allows users to:
- Remove specific frames from image stacks
- Remove frames at regular intervals
- Specify frame ranges for removal
- Create new stacks with removed frames

## Features

- Customizable start and end frame selection
- Adjustable frame removal length
- Configurable interval between removals
- Preview of frame selection
- New window creation with modified stack

## Requirements

- [FLIKA](https://github.com/flika-org/flika) version 0.2.23 or higher
- Python 3.x
- Dependencies:
  - numpy
  - scipy
  - PyQt5
  - PyQtGraph

## Installation

1. Install required dependencies:
```bash
pip install numpy scipy
```

2. Copy this plugin to your FLIKA plugins directory:
```bash
~/.FLIKA/plugins/
```

## Usage

### Basic Workflow

1. Launch FLIKA
2. Load your image stack
3. Go to Plugins → Frame Remover
4. Configure parameters
5. Click "Remove Frames"

### Parameters

- **Start Frame**: First frame in the sequence to consider
- **End Frame**: Last frame in the sequence to consider
- **Number of Frames to Remove**: How many consecutive frames to remove at each interval
- **Interval**: Number of frames between removal points

### Example Usage

To remove 5 frames every 100 frames from frame 0 to 1000:
1. Set Start Frame = 0
2. Set End Frame = 1000
3. Set Number of Frames to Remove = 5
4. Set Interval = 100
5. Click "Remove Frames"

This will remove frames:
- 0-4
- 100-104
- 200-204
- etc...

## Technical Details

### Frame Removal Algorithm

The plugin uses numpy array operations to efficiently remove frames:

```python
start = np.arange(start_frame, end_frame, interval)
to_delete = []
for i in start:
    to_delete.extend(np.arange(i, i + length))
img = np.delete(img, to_delete, 0)
```

### Default Settings

```python
settings = {
    'start': 0,
    'end': 1000,
    'length': 100,
    'interval': 100
}
```

## Tips

1. **Parameter Selection**:
   - Ensure end frame doesn't exceed stack length 
   - Consider interval spacing relative to total frames
   - Verify frame removal length is appropriate

2. **Data Handling**:
   - Original stack is preserved
   - New window created with modified stack
   - Check frame count after removal

3. **Common Use Cases**:
   - Removing regular artifacts
   - Reducing frame rate
   - Removing damaged frames
   - Creating time-lapse from continuous recording

## Known Issues

- Large frame removals may require significant memory
- Settings reset to defaults on FLIKA restart
- No undo function (but original window is preserved)

## Future Improvements

- [ ] Preview function
- [ ] Frame selection visualization
- [ ] Batch processing support
- [ ] Undo/redo capability
- [ ] Advanced frame selection patterns
- [ ] Frame interpolation option

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- George Dickinson

## Version History

- 2021.04.03: Initial release

## Acknowledgments

- Built using the FLIKA plugin template
- Thanks to the FLIKA development team

## Support

For issues related to:
- Parameter configuration
- Frame selection
- Memory handling
- General usage

Please create an issue in the repository.

## Citation

If you use this software in your research, please cite:
[Citation information to be added]
