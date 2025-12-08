# ThunderSTORM for FLIKA - Installation Package

## 📦 You've Successfully Created a Complete FLIKA Plugin!

This package contains a production-ready FLIKA plugin implementing the complete thunderSTORM functionality for Single Molecule Localization Microscopy (SMLM) analysis.

## 📁 Package Structure

```
thunderstorm_flika/
├── Core Plugin Files (Required for FLIKA)
│   ├── __init__.py           (41 KB) - Main plugin code
│   ├── info.xml              (2 KB)  - Plugin metadata
│   └── about.html            (17 KB) - HTML documentation
│
├── Documentation (Highly Recommended)
│   ├── PACKAGE_SUMMARY.md    - This overview
│   ├── README.md             - Complete documentation
│   ├── QUICKSTART.md         - 5-minute guide
│   ├── STRUCTURE.md          - Architecture details
│   └── CHANGELOG.md          - Version history
│
├── Installation & Examples
│   ├── install.sh            - Auto-install script
│   └── examples.py           - 7 example scripts
│
└── thunderstorm_python/      - Core Implementation
    ├── __init__.py           - Package init
    ├── filters.py            - Image filtering
    ├── detection.py          - Molecule detection
    ├── fitting.py            - PSF fitting
    ├── postprocessing.py     - Post-processing
    ├── visualization.py      - Rendering
    ├── simulation.py         - Data simulation
    ├── utils.py              - Utilities
    └── pipeline.py           - Main pipeline
```

## ⚡ Quick Installation

### Option 1: Automatic (Unix/Mac - Recommended)
```bash
cd thunderstorm_flika
chmod +x install.sh
./install.sh
# Restart FLIKA
```

### Option 2: Manual
```bash
# Install Python dependencies
pip install numpy scipy scikit-image matplotlib pandas pywavelets tifffile

# Copy plugin to FLIKA
cp -r thunderstorm_flika ~/.FLIKA/plugins/

# Restart FLIKA
```

### Option 3: Windows
```powershell
# Install dependencies
pip install numpy scipy scikit-image matplotlib pandas pywavelets tifffile

# Copy folder to: %USERPROFILE%\.FLIKA\plugins\
# (Usually: C:\Users\YourName\.FLIKA\plugins\)

# Restart FLIKA
```

## ✅ Verify Installation

After restarting FLIKA, you should see:

```
Plugins → ThunderSTORM
    ├── Run Analysis
    ├── Quick Analysis
    ├── Post-Processing
    ├── Drift Correction
    ├── Rendering
    └── Simulate Data
```

## 🚀 First Use (5 Minutes)

1. **Load test data:**
   - Use Simulate Data to generate test dataset, OR
   - Load your own SMLM movie

2. **Run Quick Analysis:**
   ```
   Plugins → ThunderSTORM → Quick Analysis
   ```

3. **View super-resolution image!** 🎉

## 📖 Documentation Guide

**Read in this order:**

1. **First time users:** `QUICKSTART.md`
2. **General usage:** `README.md` 
3. **Plugin details:** `about.html` (accessible in FLIKA)
4. **Architecture:** `STRUCTURE.md`
5. **Code examples:** `examples.py`

## 🎯 Key Features

✅ **Complete SMLM Analysis Pipeline**
- Multiple filtering methods (Wavelet, Gaussian, DoG, etc.)
- Advanced detection (Local Maximum, NMS, Centroid)
- PSF fitting (LSQ, WLSQ, MLE, Radial Symmetry)

✅ **Post-Processing Suite**
- Quality filtering
- Drift correction
- Molecule merging
- Density filtering

✅ **Visualization Tools**
- Gaussian rendering
- Histogram with jittering
- Average Shifted Histogram
- Scatter plots

✅ **Simulation & Testing**
- Generate test data
- Performance evaluation
- Ground truth comparison

✅ **User-Friendly Interface**
- Tabbed GUI
- Real-time feedback
- CSV import/export
- Status updates

## 💻 System Requirements

**Software:**
- FLIKA >= 0.2.25
- Python >= 3.6
- Dependencies: NumPy, SciPy, scikit-image, Pandas, PyWavelets

**Hardware (Recommended):**
- CPU: 4+ cores
- RAM: 8 GB minimum, 16 GB recommended
- Storage: SSD for large datasets

## 🔬 Scientific Reference

Based on the thunderSTORM ImageJ plugin:

> Ovesný, M., Křížek, P., Borkovec, J., Švindrych, Z., & Hagen, G. M. (2014).  
> **ThunderSTORM: a comprehensive ImageJ plugin for PALM and STORM data analysis 
> and super-resolution imaging.**  
> *Bioinformatics*, 30(16), 2389-2390.

## 🐛 Troubleshooting

### Plugin not appearing in menu?
- Verify files are in `~/.FLIKA/plugins/thunderstorm_flika/`
- Check FLIKA console for import errors
- Ensure all dependencies are installed
- Restart FLIKA

### Import errors?
```bash
# Install missing dependencies
pip install numpy scipy scikit-image matplotlib pandas pywavelets tifffile
```

### No molecules detected?
- Lower detection threshold
- Try different filter (wavelet recommended)
- Check that data contains single molecules

## 📞 Support

- **Developer:** George
- **Email:** george@research.edu
- **Documentation:** See README.md and about.html
- **Issues:** GitHub issue tracker

## 📄 License

GNU General Public License v3.0 (GPL-3.0)

## 🙏 Acknowledgments

- Original thunderSTORM by Martin Ovesný et al.
- FLIKA development team
- All algorithm authors cited in documentation

---

## ✨ Ready to Analyze!

Your plugin is **complete and ready to use**!

**Next Steps:**
1. Run `./install.sh` (or manual installation)
2. Restart FLIKA
3. Try **Quick Analysis** on test data
4. Read `QUICKSTART.md` for workflows
5. Explore `examples.py` for advanced usage

**Happy super-resolution imaging!** 🔬

---

**ThunderSTORM for FLIKA v1.0.0**  
*Professional SMLM analysis in FLIKA*

Created: December 7, 2024
