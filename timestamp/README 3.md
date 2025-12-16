# Timestamp - FLIKA Plugin

A FLIKA plugin for adding and displaying time information on image sequences. This plugin enables real-time display of timestamps based on frame rate, providing temporal context for imaging data.

## Features

### Timestamp Display
- Real-time time display
- Customizable frame rate
- Show/hide functionality
- Maintains settings between sessions

### Configuration
- Precise frame rate control (up to 10 decimal places)
- Frame rate range: 0-1,000,000 Hz
- Window-specific settings
- White text on transparent background

### Integration
- Window-specific timestamps
- Settings persistence
- Dynamic updates
- Clean removal option

## Installation

### Prerequisites
- FLIKA (version >= 0.1.0)
- Python dependencies:
  - PyQt5
  - pyqtgraph
  - numpy

### Installing the Plugin
1. Clone this repository into your FLIKA plugins directory:
```bash
cd ~/.FLIKA/plugins
git clone https://github.com/yourusername/timestamp.git
```

2. Restart FLIKA to load the plugin

## Usage

### Basic Operation
1. Launch FLIKA
2. Load your image sequence
3. Navigate to `Plugins > Timestamp`
4. Configure:
   - Select target window
   - Set frame rate
   - Enable/disable display

### Parameters
- Window: Target window for timestamp
- Frame Rate: Acquisition speed in Hz
- Show: Toggle timestamp visibility

### Time Display
- Format: Milliseconds (ms)
- Position: Lower left corner
- Style: White text, 12pt
- Background: Transparent

## Implementation Details

### Time Calculation
- Based on frame number and frame rate
- Real-time updates with frame changes
- Millisecond precision
- Frame-based timing

### Display Integration
- PyQtGraph TextItem implementation
- Non-intrusive overlay
- Dynamic position adjustment
- Clean removal process

### Settings Management
- Frame rate persistence
- Window-specific settings
- Default value handling
- Setting restoration

## Technical Features

### Window Integration
- Signal connection for updates
- Clean disconnect on removal
- Memory leak prevention
- Multiple window support

### Display Formatting
- HTML-based styling
- Size: 12pt
- Color: White
- Background: None

## Version History

Current Version: Compatible with FLIKA version displayed at startup

## Author

Contact the FLIKA team for support and contributions

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Notes

- Frame rate setting persists between sessions
- Timestamps update in real-time during playback
- Multiple windows can have independent timestamps
- Clean removal available through show/hide toggle
- Frame rate precision up to 10 decimal places
- Settings are saved per window