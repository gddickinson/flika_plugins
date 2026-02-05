# Calcium Event Data Labeler for FLIKA

A comprehensive FLIKA plugin for creating, managing, and validating training data for calcium event detection models.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## 🎯 **Overview**

This plugin provides a complete workflow for creating high-quality training datasets for deep learning models that detect calcium events (sparks, puffs, and waves) in confocal microscopy data.

### **Key Features**

✅ **Interactive Manual Labeling**
- Frame-by-frame navigation with keyboard shortcuts
- Multiple drawing tools (rectangle, ellipse, polygon, freehand)
- Class selection (spark/puff/wave)
- Undo/redo support
- Real-time overlay visualization

✅ **Automated Label Generation**
- Intensity-based thresholding
- Temporal filtering
- Morphological cleanup
- Connected component analysis
- Auto-classification by size/duration

✅ **Label Quality Assessment**
- Class distribution analysis
- Spatial/temporal coverage
- Consistency checking
- Completeness verification
- Inter-annotator agreement

✅ **Dataset Creation**
- Patch extraction from full frames
- Train/val/test splitting
- Class balancing
- Data augmentation
- Multiple output formats

✅ **Data Augmentation**
- Rotation and flipping
- Elastic deformation
- Intensity variations
- Temporal warping
- Noise addition

---

## 📦 **Installation**

### **Prerequisites**

- FLIKA >= 0.2.23
- Python 3.7+
- NumPy, SciPy
- scikit-image
- scikit-learn
- pandas
- tifffile

### **Install Plugin**

```bash
# 1. Install dependencies
pip install numpy scipy scikit-image scikit-learn pandas tifffile

# 2. Copy plugin to FLIKA plugins directory
cp -r ca_labeler_flika ~/.FLIKA/plugins/

# 3. Restart FLIKA
```

### **Verify Installation**

After restarting FLIKA:
1. Open **Plugins** menu
2. Look for **Calcium Event Data Labeler**
3. You should see 5 menu items

---

## 🚀 **Quick Start**

### **1. Manual Labeling Workflow**

```
1. Open your calcium imaging data in FLIKA
2. Plugins → Calcium Event Data Labeler → Manual Labeling
3. Select the image window
4. Click "Launch Interactive Labeler"

In the labeler:
- Use slider to navigate frames
- Select event class (Spark/Puff/Wave)
- Choose drawing tool
- Draw ROIs on overlay image
- Save labels when done
```

### **2. Automated Labeling Workflow**

```
1. Load image in FLIKA
2. Plugins → Calcium Event Data Labeler → Automated Labeling
3. Set parameters:
   - Intensity threshold (0.3 recommended)
   - Temporal filter (3 frames recommended)
   - Min event size (10 pixels)
   - Min duration (2 frames)
   - Enable auto-classification
4. Click "Run"
5. Review and refine labels manually if needed
```

### **3. Quality Check Workflow**

```
1. Plugins → Calcium Event Data Labeler → Check Label Quality
2. Select directory containing images/ and masks/ subdirectories
3. Choose which checks to perform:
   ☑ Class Distribution
   ☑ Spatial/Temporal Coverage
   ☑ Label Consistency
   ☑ Annotation Completeness
4. Click "Run"
5. Review quality report
```

### **4. Dataset Creation Workflow**

```
1. Plugins → Calcium Event Data Labeler → Create Dataset
2. Select input directory (with images/ and masks/)
3. Select output directory
4. Configure:
   - Patch size (64x64x16 recommended)
   - Train/val/test split (70/15/15)
   - Enable augmentation
   - Enable class balancing
5. Click "Run"
6. Dataset will be created with train/val/test splits
```

---

## 📖 **Detailed User Guide**

### **Interactive Manual Labeling**

The interactive labeler provides a comprehensive GUI for manual annotation.

#### **Interface Components**

1. **Frame Navigation**
   - Slider for quick frame selection
   - Keyboard shortcuts: ← → for prev/next frame

2. **Drawing Tools**
   - Rectangle: Click and drag
   - Ellipse: Click and drag
   - Polygon: Click points, double-click to complete
   - Freehand: Click and drag
   - Erase: Remove labels

3. **Class Selection**
   - Spark (green): Small, fast events
   - Puff (orange): Medium events
   - Wave (red): Large, propagating events

4. **Views**
   - Original image (left): Grayscale calcium signal
   - Labels overlay (right): Color-coded annotations

#### **Keyboard Shortcuts**

- `Ctrl+Z`: Undo
- `Ctrl+Y` / `Ctrl+Shift+Z`: Redo
- `Ctrl+S`: Save labels
- `Ctrl+O`: Load labels
- `←` / `→`: Previous/next frame
- `1` / `2` / `3`: Select spark/puff/wave class

#### **Saving Annotations**

Supports multiple formats:
- **TIFF** (.tif): Standard image format
- **NumPy** (.npy): Fast binary format
- **JSON** (.json): Text format with metadata

---

### **Automated Label Generation**

Generate initial labels automatically using signal processing.

#### **Parameters**

**Intensity Threshold** (0-1)
- Controls sensitivity of event detection
- Lower = more sensitive (more false positives)
- Higher = less sensitive (fewer events)
- Recommended: 0.3

**Temporal Filter** (frames)
- Median filter size in time
- Reduces noise
- Recommended: 3 frames

**Min Event Size** (pixels)
- Minimum spatial size
- Filters out noise
- Recommended: 10 pixels

**Min Duration** (frames)
- Minimum temporal extent
- Filters transient noise
- Recommended: 2 frames

**Auto-Classify**
- Automatically assigns classes based on:
  - Size (sparks < puffs < waves)
  - Duration (sparks < puffs < waves)
- Recommended: Enabled

#### **Classification Criteria**

Default thresholds:
- **Sparks**: Size ≤ 50 pixels, Duration ≤ 10 frames
- **Puffs**: 30-200 pixels, 5-30 frames
- **Waves**: Size ≥ 150 pixels, Duration ≥ 20 frames

---

### **Label Quality Assessment**

Comprehensive quality metrics for labeled data.

#### **Class Distribution**

Analyzes:
- Pixel counts per class
- Class percentages
- Files containing each class
- Background:event ratio
- Spark:puff:wave ratios

**Warnings:**
- Severe imbalance (>100:1)
- Missing classes
- Unbalanced event types

#### **Spatial/Temporal Coverage**

Measures:
- % frames with labels (spatial)
- % area covered (temporal)
- Mean and std dev for each

**Warnings:**
- Very sparse temporal annotations (<10%)
- Limited spatial coverage (<5%)

#### **Label Consistency**

Checks for:
- Isolated single-pixel labels
- Temporal discontinuities (gaps >5 frames)
- Unusual size distributions

#### **Annotation Completeness**

Verifies:
- All images have masks
- No orphan masks
- File name matching

---

### **Dataset Creation**

Create training-ready datasets with patches and splits.

#### **Patch Extraction**

Extracts overlapping 3D patches:
- Default size: 16 frames × 64 × 64 pixels
- Default stride: 50% overlap
- Only patches with signal are kept

#### **Data Splitting**

Stratified splitting:
- Train: 70% (default)
- Validation: 15% (default)
- Test: 15% (default)
- Random but reproducible (seed=42)

#### **Class Balancing**

Oversampling strategy:
- Counts pixels per class in each patch
- Duplicates patches with minority classes
- Balances to match majority class

#### **Augmentation**

Training set augmentation:
- 2-3 augmented copies per patch
- Random transforms applied:
  - Rotation (90°, 180°, 270°)
  - Flipping (horizontal, vertical)
  - Elastic deformation
  - Intensity variations
  - Noise addition

#### **Output Structure**

```
output_directory/
├── train/
│   ├── images/
│   │   ├── train_000000.tif
│   │   ├── train_000001.tif
│   │   └── ...
│   └── masks/
│       ├── train_000000.tif
│       ├── train_000001.tif
│       └── ...
├── val/
│   ├── images/
│   └── masks/
├── test/
│   ├── images/
│   └── masks/
└── dataset_metadata.json
```

---

## 🔬 **Advanced Features**

### **Inter-Annotator Agreement**

Compare annotations from multiple annotators:

```python
from ca_labeler_flika.label_quality import LabelQualityChecker

checker = LabelQualityChecker()
agreement = checker.compare_annotators(
    mask_files_annotator_1,
    mask_files_annotator_2
)

print(f"Cohen's Kappa: {agreement['mean_kappa']:.3f}")
print(f"Agreement level: {agreement['agreement_level']}")
```

### **Custom Augmentation Pipelines**

Create custom augmentation:

```python
from ca_labeler_flika.augmentation import AugmentationEngine

engine = AugmentationEngine()

# Custom probabilities
aug_image, aug_mask = engine.augment_patch(
    image, mask,
    p_rotate=0.8,      # 80% chance
    p_flip=0.5,        # 50% chance
    p_elastic=0.0,     # Disabled
    p_intensity=1.0,   # Always
    p_noise=0.2        # 20% chance
)
```

### **Batch Processing**

Process multiple files:

```python
from ca_labeler_flika.automated_labeling import AutomatedLabeler
from pathlib import Path
from tifffile import imread, imwrite

labeler = AutomatedLabeler()

for img_file in Path('data/images').glob('*.tif'):
    image = imread(img_file)
    
    labels = labeler.generate_labels(
        image,
        intensity_threshold=0.3,
        auto_classify=True
    )
    
    output_file = Path('data/masks') / f"{img_file.stem}_labels.tif"
    imwrite(output_file, labels)
```

---

## 📊 **Best Practices**

### **For Manual Labeling**

1. **Consistent Criteria**
   - Define clear size/duration thresholds
   - Document classification decisions
   - Use same criteria across all data

2. **Labeling Strategy**
   - Label representative frames first
   - Focus on clear, unambiguous events
   - Mark uncertain regions for later review

3. **Quality Control**
   - Regularly check previous labels
   - Compare with automated suggestions
   - Get second opinions on difficult cases

### **For Automated Labeling**

1. **Parameter Tuning**
   - Start with recommended defaults
   - Adjust threshold based on SNR
   - Validate on subset first

2. **Review and Refine**
   - Always manually review automated labels
   - Use interactive labeler to correct
   - Pay attention to boundaries

3. **Iterative Improvement**
   - Use automated labels as starting point
   - Refine manually
   - Regenerate with better parameters

### **For Dataset Creation**

1. **Data Diversity**
   - Include various conditions
   - Multiple cells/preparations
   - Different event types and sizes

2. **Balance Considerations**
   - Aim for 1:1:1 spark:puff:wave ratio
   - Background should be <100:1 ratio
   - Test set should match real data distribution

3. **Augmentation Strategy**
   - Use conservative augmentation
   - Verify augmented samples are realistic
   - Don't over-augment (diminishing returns)

---

## 🔧 **Troubleshooting**

### **Common Issues**

**Issue**: Interactive labeler doesn't open
- **Solution**: Check that window is selected in dropdown
- **Solution**: Ensure image has 3D shape (T, H, W)

**Issue**: Automated labeling finds no events
- **Solution**: Lower intensity threshold
- **Solution**: Check temporal filter isn't too aggressive
- **Solution**: Reduce min event size/duration

**Issue**: Quality check shows severe imbalance
- **Solution**: Label more events in minority classes
- **Solution**: Use class balancing in dataset creation
- **Solution**: Consider weighted loss during training

**Issue**: Dataset creation fails
- **Solution**: Check images/ and masks/ directories exist
- **Solution**: Verify file naming matches (image_video.tif → image_class.tif)
- **Solution**: Ensure enough disk space

---

## 📚 **API Reference**

### **InteractiveLabeler**

```python
from ca_labeler_flika.interactive_labeler import InteractiveLabeler

labeler = InteractiveLabeler(image_window)
labeler.show()
```

### **AutomatedLabeler**

```python
from ca_labeler_flika.automated_labeling import AutomatedLabeler

labeler = AutomatedLabeler()
labels = labeler.generate_labels(
    image,
    intensity_threshold=0.3,
    temporal_filter_size=3,
    min_event_size=10,
    min_event_duration=2,
    auto_classify=True
)
```

### **LabelQualityChecker**

```python
from ca_labeler_flika.label_quality import LabelQualityChecker

checker = LabelQualityChecker()
results = checker.assess_quality(
    data_dir,
    check_distribution=True,
    check_coverage=True,
    check_consistency=True,
    check_completeness=True
)
```

### **DatasetCreator**

```python
from ca_labeler_flika.dataset_creator import DatasetCreator

creator = DatasetCreator()
stats = creator.create_dataset(
    input_dir=Path('labeled_data'),
    output_dir=Path('training_dataset'),
    patch_size=(16, 64, 64),
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    augment=True,
    balance_classes=True
)
```

---

## 🤝 **Integration with Calcium Event Detector**

This labeler plugin pairs perfectly with the **Calcium Event Detector** plugin:

1. **Create Labels** → Use this plugin
2. **Create Dataset** → Use this plugin
3. **Train Model** → Use Detector plugin
4. **Detect Events** → Use Detector plugin
5. **Refine Labels** → Back to this plugin (iterative improvement)

---

## 📖 **Citations**

If you use this plugin in your research, please cite:


```
Dotti, P., Davey, G. I. J., Higgins, E. R., & Shirokova, N. (2024).
A deep learning-based approach for efficient detection and classification
of local Ca²⁺ release events in Full-Frame confocal imaging.
Cell Calcium, 121, 102893.
```

---

## 📞 **Support**

- **Author**: George Dickinson
- **Institution**: UC Irvine, Neurobiology & Behavior
- **Lab**: Dr. Medha Pathak Laboratory
- **Issues**: https://github.com/gddickinson/ca_labeler_flika/issues

---

## 📄 **License**

MIT License - See LICENSE file for details

---

**Version 1.0.0** - December 26, 2024  
*Complete Training Data Solution for Calcium Event Detection*
