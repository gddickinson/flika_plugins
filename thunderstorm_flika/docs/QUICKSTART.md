# ThunderSTORM for FLIKA - Quick Start Guide

## Installation

### Option 1: Automatic Installation (Recommended)

```bash
# Navigate to the plugin directory
cd thunderstorm_flika

# Run installation script
./install.sh

# Restart FLIKA
```

### Option 2: Manual Installation

1. **Install dependencies:**
   ```bash
   pip install numpy scipy scikit-image matplotlib pandas pywavelets tifffile
   ```

2. **Copy plugin:**
   ```bash
   cp -r thunderstorm_flika ~/.FLIKA/plugins/
   ```

3. **Restart FLIKA**

## First Analysis (5 Minutes)

### Step 1: Load Your Data
1. Open FLIKA
2. Load your SMLM movie: `File → Open` or drag-and-drop

### Step 2: Quick Analysis
1. Go to: `Plugins → ThunderSTORM → Quick Analysis`
2. Wait for analysis to complete (progress shown in status bar)
3. View your super-resolution image!

That's it! You now have a super-resolution reconstruction.

## Next Steps

### Save Your Results

The quick analysis creates a super-resolution image, but doesn't save the localization data. To save:

1. Use `Run Analysis` instead of Quick Analysis
2. After analysis completes, click "Save Localizations"
3. Choose a filename and location
4. Your localizations are saved as CSV

### Customize Analysis

For better results, use the full analysis pipeline:

1. **Go to:** `Plugins → ThunderSTORM → Run Analysis`

2. **Configure in tabs:**
   - **Filtering Tab:**
     - Filter Type: `wavelet` (recommended)
     - Wavelet Scale: `2`
     - Wavelet Order: `3`
   
   - **Detection Tab:**
     - Detector: `local_maximum`
     - Threshold: `std(Wave.F1)` (adaptive)
   
   - **Fitting Tab:**
     - Fitter: `gaussian_lsq` (fast) or `gaussian_mle` (best quality)
     - Fit Radius: `3` pixels
     - Initial Sigma: `1.3`
   
   - **Camera/Render Tab:**
     - Pixel Size: Your camera's pixel size (typically 100-160 nm)
     - Photons per ADU: Your camera's gain (check manual)
     - Renderer: `gaussian`
     - Render Pixel Size: `10` nm

3. **Click "Run"**

4. **Save localizations** using the "Save Localizations" button

### Post-Processing

Improve your results with post-processing:

1. **Load your localizations:**
   - `Plugins → ThunderSTORM → Post-Processing`
   - Click "Load CSV File"
   - Select your saved localizations

2. **Enable filters:**
   - Quality Filtering: Remove low-quality localizations
     - Min Intensity: `300` (adjust based on your data)
     - Max Uncertainty: `50` nm
   
   - Molecule Merging: Combine blinking events
     - Max Distance: `50` nm
     - Max Frame Gap: `2` frames

3. **Click "Run"**

4. **Save filtered localizations**

### Drift Correction

For long acquisitions, correct sample drift:

1. **Load localizations:**
   - `Plugins → ThunderSTORM → Drift Correction`
   - Click "Load Localizations CSV"

2. **Configure:**
   - Method: `cross_correlation`
   - Bin Size: `5` frames

3. **Click "Run"**

4. **Save corrected localizations**

## Example Workflow: Complete Analysis

Here's a complete workflow from raw data to final image:

```
1. Load SMLM movie in FLIKA

2. Run Analysis
   - Plugins → ThunderSTORM → Run Analysis
   - Use wavelet filter, local maximum detector, LSQ fitter
   - Save localizations as "raw_localizations.csv"

3. Drift Correction (if needed)
   - Plugins → ThunderSTORM → Drift Correction
   - Load "raw_localizations.csv"
   - Apply cross-correlation method
   - Save as "drift_corrected.csv"

4. Post-Processing
   - Plugins → ThunderSTORM → Post-Processing
   - Load "drift_corrected.csv"
   - Apply quality filtering (min intensity, max uncertainty)
   - Apply molecule merging (max distance 50 nm, max gap 2 frames)
   - Save as "final_localizations.csv"

5. Final Rendering
   - Plugins → ThunderSTORM → Rendering
   - Load "final_localizations.csv"
   - Use Gaussian renderer with 10 nm pixel size
   - Save resulting image as TIFF
```

## Testing with Simulated Data

Before analyzing your real data, test with simulated data:

1. **Generate test data:**
   - `Plugins → ThunderSTORM → Simulate Data`
   - Choose pattern (Siemens star recommended)
   - Save ground truth if desired

2. **Analyze simulated data:**
   - Use Quick Analysis or Run Analysis
   - Compare results to ground truth

3. **Tune parameters:**
   - Adjust threshold if too many/few detections
   - Try different fitting methods
   - Compare rendering methods

## Troubleshooting

### No molecules detected
- **Lower threshold:** Try `0.5*std(Wave.F1)` instead of `std(Wave.F1)`
- **Check filter:** Wavelet works best for most data
- **Verify data:** Make sure you have single-molecule data, not diffuse signal

### Analysis too slow
- **Use faster fitter:** Try `radial_symmetry` instead of `gaussian_mle`
- **Process subset:** Analyze first 100 frames to test parameters
- **Reduce fit radius:** Try 2 or 3 pixels instead of 5

### Poor localization precision
- **Use better fitter:** Switch from LSQ to MLE
- **Check photons:** Make sure molecules are bright enough
- **Adjust sigma:** Initial sigma should match your PSF (typically 1.0-1.6 pixels)

### Image looks noisy
- **Apply post-processing:** Use quality filtering
- **Adjust rendering:** Try different pixel sizes or rendering methods
- **Filter by uncertainty:** Keep only localizations with <30 nm uncertainty

## Parameter Tuning Tips

### Detection Threshold
Start with `std(Wave.F1)` and adjust:
- **Too many detections?** Increase: `2*std(Wave.F1)`
- **Missing molecules?** Decrease: `0.5*std(Wave.F1)`

### PSF Sigma
Should match your microscope's PSF:
- **Typical range:** 1.0-1.6 pixels
- **Too small?** You'll see fitting errors
- **Too large?** Precision decreases

### Fit Radius
Determines region used for fitting:
- **Bright spots:** 4-5 pixels
- **Dim spots:** 3 pixels
- **Dense regions:** 2-3 pixels

## Getting Help

- **Documentation:** See `about.html` in plugin directory
- **README:** Full README.md with detailed information
- **GitHub:** Issues and discussions at [repository URL]
- **Email:** george@research.edu

## Next Steps

- Read the full README.md for detailed documentation
- View about.html for comprehensive parameter descriptions
- Check the thunderSTORM paper (Ovesný et al., 2014) for algorithm details
- Join the FLIKA community for support

---

**Happy super-resolution imaging!** 🔬✨
