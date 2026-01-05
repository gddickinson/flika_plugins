# ThunderSTORM for FLIKA - Complete Plugin Package

## 🎉 Plugin Complete!

This package contains a complete, production-ready FLIKA plugin implementing the full thunderSTORM functionality for Single Molecule Localization Microscopy (SMLM) analysis.

## 📦 Package Contents

### Core Plugin Files (Required)
- `__init__.py` (41 KB) - Main plugin with FLIKA integration and GUI
- `info.xml` (2.1 KB) - Plugin metadata for FLIKA Plugin Manager  
- `about.html` (17 KB) - Comprehensive HTML documentation

### Documentation (Highly Recommended)
- `README.md` (11 KB) - Complete documentation and usage guide
- `QUICKSTART.md` (6.2 KB) - 5-minute quick start guide
- `STRUCTURE.md` (13 KB) - Detailed plugin architecture overview
- `CHANGELOG.md` (2.7 KB) - Version history and future plans

### Installation & Examples
- `install.sh` (2.7 KB) - Automatic installation script (Unix/Mac)
- `examples.py` (14 KB) - Seven comprehensive example scripts

### Core thunderSTORM Implementation
- `thunderstorm_python/` - Complete Python implementation
  - `__init__.py` - Package initialization
  - `filters.py` (8.5 KB) - Image filtering algorithms
  - `detection.py` (9.1 KB) - Molecule detection methods
  - `fitting.py` (20 KB) - PSF fitting algorithms
  - `postprocessing.py` (19 KB) - Post-processing tools
  - `visualization.py` (14 KB) - Rendering methods
  - `simulation.py` (15 KB) - Data simulation & evaluation
  - `utils.py` (13 KB) - Utility functions
  - `pipeline.py` (14 KB) - Main analysis pipeline

**Total Size:** ~200 KB of code + documentation

## ✨ Features Implemented

### Analysis Pipeline
✅ Complete SMLM analysis workflow  
✅ Multiple filtering methods (Wavelet, Gaussian, DoG, etc.)  
✅ Multiple detection algorithms (Local Maximum, NMS, Centroid)  
✅ Advanced PSF fitting (LSQ, WLSQ, MLE, Radial Symmetry)  
✅ Threshold expression evaluation  
✅ Camera parameter handling  

### Post-Processing
✅ Quality-based filtering  
✅ Density-based filtering  
✅ Molecule merging (blinking correction)  
✅ Drift correction (cross-correlation & fiducial)  
✅ Duplicate removal  

### Visualization
✅ Gaussian rendering  
✅ Histogram rendering with jittering  
✅ Average Shifted Histogram (ASH)  
✅ Scatter plot rendering  
✅ Configurable pixel size  

### Simulation & Testing
✅ SMLM data generation  
✅ Multiple test patterns (Siemens star, grid, circles)  
✅ Blinking dynamics  
✅ Performance evaluation (Recall, Precision, F1, RMSE)  
✅ Ground truth export  

### User Interface
✅ Tabbed GUI for main analysis  
✅ Separate tools for each function  
✅ Quick Analysis mode  
✅ Save/load CSV localizations  
✅ Real-time status updates  
✅ Error handling and user feedback  

## 🚀 Quick Installation

### Method 1: Automatic (Recommended)
```bash
cd thunderstorm_flika
./install.sh
# Restart FLIKA
```

### Method 2: Manual
```bash
# Install dependencies
pip install numpy scipy scikit-image matplotlib pandas pywavelets tifffile

# Copy to FLIKA plugins
cp -r thunderstorm_flika ~/.FLIKA/plugins/

# Restart FLIKA
```

## 📖 Quick Start (5 Minutes)

1. **Start FLIKA**
2. **Load your SMLM data** (File → Open)
3. **Run Quick Analysis** (Plugins → ThunderSTORM → Quick Analysis)
4. **View results!** 🎉

For detailed workflows, see `QUICKSTART.md`

## 🎯 Key Plugin Classes

### 1. ThunderSTORM_RunAnalysis
Complete analysis pipeline with tabbed GUI
- **Menu:** Plugins → ThunderSTORM → Run Analysis
- **Features:** Full parameter control, CSV export
- **Use:** Primary analysis tool

### 2. ThunderSTORM_PostProcessing  
Quality filtering and molecule merging
- **Menu:** Plugins → ThunderSTORM → Post-Processing
- **Features:** Quality/density filtering, merging
- **Use:** Refine localization data

### 3. ThunderSTORM_DriftCorrection
Correct sample drift
- **Menu:** Plugins → ThunderSTORM → Drift Correction
- **Features:** Cross-correlation, fiducial tracking
- **Use:** Long acquisition correction

### 4. ThunderSTORM_Rendering
Create super-resolution images
- **Menu:** Plugins → ThunderSTORM → Rendering
- **Features:** Multiple rendering methods
- **Use:** Final image generation

### 5. Quick Analysis & Simulation
Simple functions for fast work
- **Quick Analysis:** Default parameter analysis
- **Simulate Data:** Generate test datasets

## 📊 Algorithm Comparison

### PSF Fitting Methods:
| Method | Speed | Accuracy | Best For |
|--------|-------|----------|----------|
| LSQ | ⚡⚡⚡ | ⭐⭐⭐ | General use |
| WLSQ | ⚡⚡⚡ | ⭐⭐⭐⭐ | Low SNR |
| MLE | ⚡ | ⭐⭐⭐⭐⭐ | Publication |
| Radial | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | Preview |
| Centroid | ⚡⚡⚡⚡⚡ | ⭐⭐ | Simple |

### Rendering Methods:
| Method | Speed | Quality | Memory |
|--------|-------|---------|--------|
| Gaussian | ⚡⚡ | ⭐⭐⭐⭐⭐ | 🔴 High |
| Histogram | ⚡⚡⚡⚡⚡ | ⭐⭐⭐ | 🟢 Low |
| ASH | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ | 🟢 Low |
| Scatter | ⚡⚡⚡⚡⚡ | ⭐⭐ | 🟢 Low |

## 💡 Usage Examples

### Example 1: Quick Analysis
```python
from flika.process.file_ import open_file
from thunderstorm_flika import quick_analysis_simple

window = open_file('data.tif')
result = quick_analysis_simple()
```

### Example 2: Custom Analysis
```python
from thunderstorm_flika import ThunderSTORM_RunAnalysis

analysis = ThunderSTORM_RunAnalysis()
analysis.show()  # Opens GUI
```

### Example 3: Programmatic
```python
from thunderstorm_python import ThunderSTORM

pipeline = ThunderSTORM(
    filter_type='wavelet',
    fitter_type='gaussian_mle',
    pixel_size=100.0
)
locs = pipeline.analyze_stack(images)
pipeline.save('locs.csv')
```

See `examples.py` for 7 comprehensive examples!

## 📝 Typical Workflow

```
1. Load Data → Open SMLM movie in FLIKA
         ↓
2. Run Analysis → Plugins → ThunderSTORM → Run Analysis
         ↓
3. Save Locs → Click "Save Localizations" button
         ↓
4. Drift Correct → Load CSV, apply drift correction
         ↓
5. Post-Process → Quality filtering, merging
         ↓
6. Re-Render → Choose rendering method
         ↓
7. Export → Save final CSV and TIFF
```

## 🔧 System Requirements

### Software:
- FLIKA >= 0.2.25
- Python >= 3.6
- NumPy, SciPy, scikit-image, Pandas, PyWavelets

### Hardware (Recommended):
- **CPU:** Multi-core (4+ cores)
- **RAM:** 8 GB minimum, 16 GB recommended
- **Storage:** SSD for large datasets

### Performance Notes:
- **Small dataset** (<1000 frames): <1 GB RAM
- **Medium dataset** (1000-5000 frames): 1-4 GB RAM  
- **Large dataset** (>5000 frames): 4+ GB RAM
- **MLE fitting:** CPU-intensive, benefits from multiple cores

## 📚 Documentation Hierarchy

1. **New Users:** Start with `QUICKSTART.md`
2. **General Use:** See `README.md`
3. **Plugin Details:** See `about.html` (via FLIKA)
4. **Architecture:** See `STRUCTURE.md`
5. **Examples:** See `examples.py`
6. **Changes:** See `CHANGELOG.md`

## 🐛 Troubleshooting

### Plugin doesn't load?
- Check thunderstorm_python is in plugin directory
- Verify dependencies: `pip list | grep -E "numpy|scipy|scikit"`
- Check FLIKA console for errors

### No molecules detected?
- Lower threshold: try `0.5*std(Wave.F1)`
- Check filter type (wavelet usually best)
- Verify data is single-molecule

### Fitting fails?
- Adjust initial_sigma (should be 1.0-1.6)
- Increase fit_radius for bright spots
- Try radial_symmetry for difficult data

See README.md for complete troubleshooting guide.

## 🤝 Contributing

We welcome contributions! To contribute:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## 📄 License

GNU General Public License v3.0 (GPL-3.0)

Consistent with original thunderSTORM ImageJ plugin.

## 🙏 Acknowledgments

- **Original thunderSTORM:** Martin Ovesný et al.
- **Reference:** Ovesný, M., et al. (2014). ThunderSTORM: a comprehensive 
  ImageJ plugin for PALM and STORM data analysis and super-resolution 
  imaging. *Bioinformatics*, 30(16), 2389-2390.
- **FLIKA:** FLIKA development team
- **Algorithm Authors:** All cited researchers

## 📞 Support

- **Developer:** George
- **Email:** george@research.edu  
- **GitHub:** [Repository URL]
- **Issues:** Use GitHub issue tracker

## 🎓 Citation

If you use this plugin in your research, please cite:

1. **Original thunderSTORM paper:**
   ```
   Ovesný, M., Křížek, P., Borkovec, J., Švindrych, Z., & Hagen, G. M. (2014).
   ThunderSTORM: a comprehensive ImageJ plugin for PALM and STORM data analysis 
   and super-resolution imaging. Bioinformatics, 30(16), 2389-2390.
   ```

2. **FLIKA:**
   ```
   Ellefsen, K., Settle, B., Parker, I. & Smith, I. (2014).
   An algorithm for automated detection, localization and measurement of local 
   calcium signals from camera-based imaging. Cell Calcium, 56:147-156.
   ```

## 🚀 Future Enhancements

Planned for future versions:
- [ ] Batch processing GUI
- [ ] Automated 3D calibration
- [ ] GPU acceleration
- [ ] Additional file formats
- [ ] Live preview during acquisition
- [ ] Integration with other FLIKA tools
- [ ] Advanced visualization options

See `CHANGELOG.md` for details.

---

## ✅ Installation Checklist

- [ ] Dependencies installed
- [ ] Plugin copied to ~/.FLIKA/plugins/
- [ ] FLIKA restarted
- [ ] ThunderSTORM appears in Plugins menu
- [ ] Quick Analysis tested on sample data
- [ ] Documentation reviewed

---

## 🎉 Ready to Use!

Your ThunderSTORM for FLIKA plugin is **complete and ready to use**!

**Next steps:**
1. Install using `./install.sh`
2. Restart FLIKA
3. Try Quick Analysis on your data
4. Explore the documentation
5. Customize parameters for your needs

**Happy super-resolution imaging!** 🔬✨

---

**ThunderSTORM for FLIKA v1.0.0**  
*Bringing the power of thunderSTORM to FLIKA*

Package created: December 7, 2024
