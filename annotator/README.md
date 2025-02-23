# Flika Annotator Plugin

A plugin for [Flika](https://github.com/flika-org/flika) that allows you to add numeric index annotations to image stacks. This plugin is useful for keeping track of frame numbers when analyzing image sequences.

## Features

- Add frame numbers to image stacks
- Customize font size
- Choose between black and white text
- Control text position using anchor points
- Non-destructive operation (creates new window with annotated images)

## Requirements

- [Flika](https://github.com/flika-org/flika) version 0.2.23 or higher
- Python 3.x
- PIL (Pillow) - Required for text rendering
- PyQt5
- NumPy
- PyQtGraph

## Installation

1. Ensure you have Flika installed
2. Install required dependencies:
```bash
pip install pillow
```
3. Copy this plugin to your Flika plugins directory:
```bash
~/.Flika/plugins/
```

## Usage

1. Launch Flika
2. Go to Plugins → Annotator
3. Select the image stack window you want to annotate
4. Configure annotation settings:
   - **Font Size**: Adjust the size of the frame numbers
   - **Font Color**: Choose between white and black text
   - **Anchor Position**: Select where the text should be positioned
     - ms: middle-start
     - ma: middle-ascender
     - ls: left-start
     - mb: middle-baseline
     - mt: middle-top
     - mm: middle-middle
     - md: middle-descent
     - rs: right-start
5. Click "Add Index" to create a new window with the annotated image stack

## Settings

The plugin saves your preferences in Flika's settings:
- Font size
- Font color
- Anchor position

These settings will persist between sessions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- George Dickinson

## Version History

- 2019.01.25: Initial release

## Acknowledgments

- Built using the Flika plugin template
- Uses PIL for text rendering
- Thanks to the Flika development team

## Known Issues

- Font path is currently hardcoded to Windows Arial font location
- Limited font color options (black/white only)

## Future Improvements

- [ ] Add custom font selection
- [ ] Support more text colors
- [ ] Add custom text formatting options
- [ ] Add more anchor position options
- [ ] Support custom text content beyond frame numbers
