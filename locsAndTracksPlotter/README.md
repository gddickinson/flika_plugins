# LocsAndTracksPlotter FLIKA Plugin

A powerful FLIKA plugin for visualizing and analyzing particle tracking data, with particular focus on studying the motion of intracellular proteins labeled with fluorescent tags.

## Features

### Data Import & Visualization
- Support for multiple data formats:
  - FLIKA pyinsight
  - ThunderSTORM
  - Custom XY coordinate formats
- Real-time visualization of tracks and points
- Customizable color schemes based on track properties
- Interactive track selection and display
- Multiple visualization modes:
  - Point display
  - Track paths
  - Heat maps
  - Flower plots (zeroed origin plots)

### Analysis Tools
- Advanced filtering capabilities:
  - Filter by ROI
  - Filter by track properties
  - Sequential filtering
  - Custom threshold filters
- Track statistics:
  - Mean and standard deviation calculations
  - Diffusion analysis
  - Velocity measurements
  - Neighbor density analysis
- Interactive ROI selection with real-time updates

### Specialized Plot Windows
- All Tracks Plot: View intensity profiles for all tracked particles
- Single Track Plot: Detailed analysis of individual tracks
- Track Display: Shows intensity, distance, velocity and more for selected tracks
- Flower Plot: Visualizes all tracks with a common origin
- Chart Window: Customizable scatter and line plots
- Diffusion Analysis: MSD and statistical analysis tools

### Additional Tools
- Background subtraction
- Track joining capabilities 
- Data export in multiple formats
- ROI-based analysis

## Installation

1. Ensure you have FLIKA installed
2. Clone this repository into your FLIKA plugins directory:
   ```bash
   cd ~/.FLIKA/plugins
   git clone https://github.com/yourusername/LocsAndTracksPlotter.git
   ```

## Dependencies

- FLIKA
- NumPy
- Pandas
- PyQt (via qtpy)
- PyQtGraph
- SciPy
- scikit-learn (for track classification)
- tqdm (for progress bars)

## Usage

### Basic Operation

1. Launch FLIKA and load your image stack
2. Open the LocsAndTracksPlotter plugin
3. Load tracking data using the "Load Data" button
4. Use the main interface to:
   - Plot points
   - Show/hide tracks
   - Apply filters
   - Access analysis tools

### Main Controls

- Point Display:
  - "Plot Points": Display all tracked points
  - "Toggle Points": Show/hide points
  - "Show Unlinked": Display untracked points
  
- Track Display:
  - "Plot Tracks": Show all tracks
  - "Clear Tracks": Remove track display
  - "Save Tracks": Export track data

- Analysis Windows:
  - "Show Charts": Open statistical analysis window
  - "Show Diffusion": Open diffusion analysis tools
  - "Plot Point Map": Display point density map
  - "Overlay": Open overlay options

### Configuration Options

#### Track Plot Options

- Point Color
- Track Color
- Line Width
- Point Size
- Color Maps
  - Support for both PyQtGraph and Matplotlib colormaps
  - Color coding by track properties
  
#### Filter Options

- Column Selection
- Operator Selection
- Value Thresholds
- ROI-based Filtering
- Sequential Filtering Support

#### Recording Parameters

- Frame Length (ms)
- Pixel Size (nm)
- Integration Options
- Background Subtraction

## Data Analysis Features 

### Track Analysis

- Intensity Profiles
- Distance Measurements
- Velocity Calculations
- Directional Analysis
- Neighbor Density
- Track Classification

### Diffusion Analysis

- Mean Square Displacement
- Diffusion Coefficient Calculation
- Multiple Component Fitting
- CDF Analysis

### ROI Analysis

- Intensity Integration
- Background Subtraction
- Custom ROI Shapes
- Real-time Updates

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

George Dickinson (george.dickinson@gmail.com)

## Acknowledgments

- FLIKA development team
- Based on the FLIKA plugin template
