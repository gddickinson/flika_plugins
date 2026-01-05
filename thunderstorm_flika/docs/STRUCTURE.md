# ThunderSTORM for FLIKA - Plugin Structure Overview

## Directory Structure

```
thunderstorm_flika/
├── __init__.py                    # Main plugin file with FLIKA integration
├── info.xml                       # Plugin metadata for FLIKA Plugin Manager
├── about.html                     # HTML documentation
├── README.md                      # Comprehensive documentation
├── QUICKSTART.md                  # Quick start guide
├── CHANGELOG.md                   # Version history
├── examples.py                    # Example usage scripts
├── install.sh                     # Automatic installation script
│
└── thunderstorm_python/           # Core thunderSTORM implementation
    ├── __init__.py               # Package initialization
    ├── filters.py                # Image filtering algorithms
    ├── detection.py              # Molecule detection methods
    ├── fitting.py                # PSF fitting algorithms
    ├── postprocessing.py         # Post-processing tools
    ├── visualization.py          # Rendering methods
    ├── simulation.py             # Data simulation tools
    ├── utils.py                  # Utility functions
    └── pipeline.py               # Main analysis pipeline
```

## Component Overview

### 1. FLIKA Integration (__init__.py)

The main plugin file that integrates thunderSTORM with FLIKA. Contains:

#### Main Classes:
- **ThunderSTORM_RunAnalysis**: Complete analysis pipeline with GUI
  - Tabbed interface for parameter configuration
  - Filtering, detection, fitting, camera, and rendering settings
  - Save localizations to CSV
  - Menu: `Plugins → ThunderSTORM → Run Analysis`

- **ThunderSTORM_PostProcessing**: Post-processing tools
  - Quality filtering (intensity, uncertainty, sigma)
  - Density-based filtering
  - Molecule merging (blinking correction)
  - Menu: `Plugins → ThunderSTORM → Post-Processing`

- **ThunderSTORM_DriftCorrection**: Drift correction
  - Cross-correlation method
  - Fiducial marker method
  - Menu: `Plugins → ThunderSTORM → Drift Correction`

- **ThunderSTORM_Rendering**: Visualization tools
  - Gaussian, Histogram, ASH, Scatter rendering
  - Configurable pixel size
  - Menu: `Plugins → ThunderSTORM → Rendering`

#### Standalone Functions:
- **quick_analysis_simple()**: Fast analysis with defaults
  - Menu: `Plugins → ThunderSTORM → Quick Analysis`

- **simulate_smlm_data()**: Generate test data
  - Menu: `Plugins → ThunderSTORM → Simulate Data`

### 2. Core thunderSTORM Package (thunderstorm_python/)

#### 2.1 filters.py
Image filtering for feature enhancement:
- **GaussianFilter**: Lowpass smoothing
- **WaveletFilter**: B-spline wavelet (à trous algorithm) - recommended
- **DifferenceOfGaussians**: Band-pass filtering
- **LoweredGaussian**: Gaussian minus averaging
- **DifferenceOfAveraging**: Difference of box filters
- **MedianFilter**: Salt-and-pepper noise removal
- **BoxFilter**: Simple averaging
- **NoFilter**: Pass-through

**Key Functions:**
- `create_filter(filter_type, **kwargs)`: Factory for creating filters
- `compute_threshold_expression(image, filtered, expr)`: Evaluate threshold

#### 2.2 detection.py
Molecule detection methods:
- **LocalMaximumDetector**: Peak detection (recommended)
- **NonMaximumSuppression**: Morphological detection
- **CentroidDetector**: Connected component centroids
- **GridDetector**: Regular grid (for testing)

**Key Functions:**
- `create_detector(detector_type, **kwargs)`: Factory for detectors
- `refine_detections()`: Subpixel refinement
- `filter_detections_by_intensity()`: Intensity filtering
- `remove_border_detections()`: Remove edge detections

#### 2.3 fitting.py
PSF fitting algorithms:
- **GaussianLSQ**: Least squares Gaussian fitting - fast, good quality
- **GaussianWLSQ**: Weighted least squares - better for low SNR
- **GaussianMLE**: Maximum likelihood estimation - best accuracy
- **RadialSymmetryFitter**: Fast, non-iterative - good for preview
- **CentroidFitter**: Simple intensity-weighted centroid
- **MultiEmitterFitter**: Fit multiple overlapping molecules

**Key Classes:**
- `FitResult`: Container for fit results
- `BaseGaussianFitter`: Base class for Gaussian fitters

**Key Functions:**
- `create_fitter(fitter_type, **kwargs)`: Factory for fitters
- `compute_uncertainty()`: Cramér-Rao lower bound calculation

#### 2.4 postprocessing.py
Post-processing and refinement:
- **LocalizationFilter**: Quality-based filtering
  - Filter by intensity, background, sigma, uncertainty
- **LocalDensityFilter**: Density-based filtering
  - Remove isolated localizations
- **MolecularMerger**: Merge reappearing molecules
  - Handle blinking with frame gap tolerance
- **DuplicateRemover**: Remove duplicate detections
- **DriftCorrector**: Correct sample drift
  - Cross-correlation method
  - Fiducial marker tracking

**Key Classes:**
- `LocalizationFilter`: Quality filtering
- `LocalDensityFilter`: Density filtering
- `MolecularMerger`: Blinking correction
- `DriftCorrector`: Drift correction

#### 2.5 visualization.py
Super-resolution rendering:
- **GaussianRenderer**: High-quality rendering
  - Each molecule as 2D Gaussian
  - Variable or computed sigma
- **HistogramRenderer**: Fast rendering
  - Simple binning
  - Optional jittering
- **AverageShiftedHistogram**: ASH rendering
  - Fast alternative to Gaussian
  - 2, 4, or 8 shifts
- **ScatterRenderer**: Simple position markers

**Key Functions:**
- `create_renderer(renderer_type, **kwargs)`: Factory for renderers
- `render_3d_projection()`: 3D visualization
- `apply_colormap()`: Colormap application

#### 2.6 simulation.py
Data generation and evaluation:
- **SMLMSimulator**: Generate synthetic SMLM data
  - Realistic PSF and noise
  - Blinking dynamics
  - Custom spatial patterns
- **PerformanceEvaluator**: Evaluate algorithm performance
  - Recall, Precision, F1 score
  - RMSE calculation
  - True/false positive counting

**Key Functions:**
- `create_test_pattern()`: Generate test patterns
  - Siemens star, grid, circles, random

#### 2.7 utils.py
Utility functions:
- **File I/O**:
  - `load_localizations_csv()`: Load CSV localizations
  - `save_localizations_csv()`: Save CSV localizations
  - `load_image_stack()`: Load TIFF/NPY images
  - `save_image_stack()`: Save images

- **Analysis**:
  - `compute_nearest_neighbor_distances()`: NN analysis
  - `compute_ripley_k()`: Ripley's K function
  - `compute_localization_density()`: Density maps
  - `compute_statistics()`: Summary statistics

- **Transformations**:
  - `convert_coordinates()`: Coordinate system conversion
  - `filter_by_roi()`: ROI filtering

**Key Classes:**
- `LocalizationTable`: Pandas-like interface for localizations

#### 2.8 pipeline.py
Main analysis pipeline:
- **ThunderSTORM**: Complete analysis pipeline
  - Integrate all components
  - Manage workflow
  - Handle parameters

**Key Functions:**
- `create_default_pipeline()`: Default parameter pipeline
- `quick_analysis()`: Fast analysis with defaults

## Data Flow

### Complete Analysis Workflow:

```
1. Raw Image Stack
   ↓
2. Image Filtering (filters.py)
   - Wavelet, Gaussian, DoG, etc.
   ↓
3. Threshold Computation (filters.py)
   - Evaluate threshold expression
   ↓
4. Molecule Detection (detection.py)
   - Find approximate positions
   ↓
5. PSF Fitting (fitting.py)
   - Precise sub-pixel localization
   - Intensity, background, sigma estimation
   ↓
6. Post-Processing (postprocessing.py)
   - Drift correction
   - Molecule merging
   - Quality filtering
   - Density filtering
   ↓
7. Visualization (visualization.py)
   - Render super-resolution image
   ↓
8. Output
   - CSV localization table
   - Super-resolution image
   - Statistics
```

## Algorithm Parameters

### Critical Parameters:

1. **Filtering**:
   - `filter_type`: 'wavelet' (recommended), 'gaussian', 'dog', etc.
   - `scale`: Wavelet scale (typically 2)
   - `sigma`: Gaussian sigma (typically 1.6)

2. **Detection**:
   - `detector_type`: 'local_maximum' (recommended)
   - `threshold_expression`: 'std(Wave.F1)' (adaptive)

3. **Fitting**:
   - `fitter_type`: 'gaussian_lsq' (fast) or 'gaussian_mle' (best)
   - `fit_radius`: 3-5 pixels
   - `initial_sigma`: 1.0-1.6 pixels
   - `integrated`: True (recommended)

4. **Camera**:
   - `pixel_size`: Camera pixel size in nm
   - `photons_per_adu`: Conversion factor
   - `baseline`: Camera baseline
   - `em_gain`: EM gain (1 for sCMOS)

## Performance Characteristics

### Speed (relative, for 1000 molecules):
- **Radial Symmetry**: ~1x (fastest)
- **Centroid**: ~1x
- **Gaussian LSQ**: ~5x
- **Gaussian WLSQ**: ~8x
- **Gaussian MLE**: ~20x (slowest, best quality)

### Memory Usage:
- **Small dataset** (<1000 frames, 512×512): <1 GB
- **Medium dataset** (1000-5000 frames): 1-4 GB
- **Large dataset** (>5000 frames): >4 GB

### Accuracy (Cramér-Rao bound):
- **Theoretical limit**: σ_loc ≈ σ_PSF / √N
  - σ_PSF: PSF width (~150 nm)
  - N: Photon count
- **Typical precision**: 10-30 nm
- **Best achievable**: 5-10 nm (high photons, low background)

## Dependencies

### Required:
- **FLIKA** >= 0.2.25
- **NumPy**: Array operations
- **SciPy**: Scientific computing
- **scikit-image**: Image processing
- **Pandas**: Data handling
- **PyWavelets**: Wavelet transforms

### Optional:
- **matplotlib**: Plotting (for standalone use)
- **tifffile**: TIFF I/O enhancement

## Integration with FLIKA

### Menu Structure:
```
Plugins
└── ThunderSTORM
    ├── Run Analysis           (ThunderSTORM_RunAnalysis)
    ├── Quick Analysis         (quick_analysis_simple)
    ├── Post-Processing        (ThunderSTORM_PostProcessing)
    ├── Drift Correction       (ThunderSTORM_DriftCorrection)
    ├── Rendering              (ThunderSTORM_Rendering)
    └── Simulate Data          (simulate_smlm_data)
```

### FLIKA Integration Points:
- **g.win**: Current window access
- **Window()**: Create result windows
- **g.alert()**: User notifications
- **g.m.statusBar()**: Status messages
- **BaseProcess**: GUI framework
- **SliderLabel, CheckBox, ComboBox**: GUI components

## File Formats

### Input:
- TIFF (single/multi-page): Primary format
- All FLIKA-supported formats
- CSV: For post-processing

### Output:
- **CSV**: Localization tables
  - Columns: x, y, frame, intensity, background, sigma_x, sigma_y, uncertainty, chi_squared
- **TIFF**: Super-resolution images
- **NPY**: NumPy arrays (optional)

## Extending the Plugin

### Adding New Filters:
```python
# In filters.py
class MyCustomFilter(BaseFilter):
    def __init__(self, param1=1.0):
        super().__init__("MyFilter")
        self.param1 = param1
    
    def apply(self, image):
        # Your filtering logic
        return filtered_image
```

### Adding New Detectors:
```python
# In detection.py
class MyCustomDetector(BaseDetector):
    def detect(self, image, threshold):
        # Your detection logic
        return coordinates  # Nx2 array
```

### Adding New Fitters:
```python
# In fitting.py
class MyCustomFitter(BaseFitter):
    def fit_single(self, roi, position):
        # Your fitting logic
        return FitResult(...)
```

## Testing

### Unit Tests:
Run individual component tests:
```python
python -m pytest thunderstorm_python/tests/
```

### Integration Tests:
Test with simulated data:
```python
from thunderstorm_python.simulation import SMLMSimulator
simulator = SMLMSimulator()
movie, gt = simulator.generate_movie(n_frames=100)
# Analyze and compare to ground truth
```

### Performance Tests:
Benchmark on your data:
```python
import time
start = time.time()
pipeline.analyze_stack(movie)
print(f"Time: {time.time() - start:.1f}s")
```

## Troubleshooting Tips

1. **Check FLIKA console for errors**
2. **Verify thunderstorm_python import**
3. **Test with simulated data first**
4. **Start with default parameters**
5. **Use Quick Analysis for initial testing**
6. **Compare different fitting methods**
7. **Validate with ground truth when available**

## Support & Development

- **GitHub**: Repository for issues and contributions
- **Email**: george@research.edu
- **Documentation**: See README.md and about.html

---

**ThunderSTORM for FLIKA v1.0.0**  
Complete SMLM analysis in FLIKA
