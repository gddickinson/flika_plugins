# PIEZO1 Calcium Tracker

**Hybrid deep learning system combining single-molecule PIEZO1 puncta localization with calcium signal detection**

This project implements a novel architecture that:
- Localizes PIEZO1-HaloTag puncta with sub-pixel precision
- Detects calcium signals at identified puncta locations
- Uses a shared encoder with specialized decoder heads
- Trains on synthetic + bootstrapped + real labeled data

---

## Project Overview

### Architecture

```
Input: Dual-channel stack (HaloTag + Calcium)
         ↓
    Shared Encoder (U-Net backbone, 16 channels)
         ↓
    ┌────────┴────────┐
    ↓                 ↓
Localization      Calcium
Decoder           Decoder
    ↓                 ↓
(x, y, σ, N)      (puncta signal)
```

### Key Features

- **Multi-source training data**:
  - Synthetic puncta (PSF-based simulation)
  - ThunderSTORM bootstrapped labels
  - Adapted calcium event labels (puncta-only)

- **Hybrid architecture**:
  - Shared encoder from existing U-Net (16 channels)
  - DECODE-style localization decoder
  - Calcium signal decoder (binary: signal/no-signal at puncta)

- **Multi-task learning**:
  - Kendall uncertainty weighting
  - Balanced gradient updates
  - Independent task monitoring

---

## Project Structure

```
piezo1_calcium_tracker/
├── README.md                          # This file
├── setup.py                           # Package installation
├── requirements.txt                   # Dependencies
│
├── piezo1_tracker/                    # Main package
│   ├── __init__.py
│   │
│   ├── data/                          # Data generation & loading
│   │   ├── __init__.py
│   │   ├── synthetic_generator.py    # PSF-based synthetic data
│   │   ├── thunderstorm_parser.py    # Parse ThunderSTORM output
│   │   ├── calcium_adapter.py        # Adapt existing calcium labels
│   │   ├── dataset.py                # PyTorch dataset
│   │   └── augmentation.py           # Data augmentation
│   │
│   ├── models/                        # Network architectures
│   │   ├── __init__.py
│   │   ├── shared_encoder.py         # Shared U-Net encoder
│   │   ├── localization_decoder.py   # DECODE-style decoder
│   │   ├── calcium_decoder.py        # Calcium signal decoder
│   │   └── hybrid_model.py           # Combined model
│   │
│   ├── training/                      # Training components
│   │   ├── __init__.py
│   │   ├── losses.py                 # Multi-task losses
│   │   ├── trainer.py                # Training loop
│   │   └── metrics.py                # Evaluation metrics
│   │
│   ├── inference/                     # Inference & post-processing
│   │   ├── __init__.py
│   │   ├── localization.py           # Extract puncta coordinates
│   │   ├── tracking.py               # LAP-based tracking
│   │   └── correlation.py            # Puncta-calcium correlation
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       ├── psf_models.py             # PSF generation
│       ├── visualization.py          # Plotting functions
│       └── config.py                 # Configuration management
│
├── scripts/                           # Executable scripts
│   ├── 01_generate_synthetic_data.py
│   ├── 02_extract_thunderstorm_labels.py
│   ├── 03_prepare_calcium_data.py
│   ├── 04_train_localization_only.py
│   ├── 05_train_hybrid_model.py
│   ├── 06_evaluate_model.py
│   └── 07_run_inference.py
│
├── configs/                           # Configuration files
│   ├── synthetic_data.yaml
│   ├── training_localization.yaml
│   ├── training_hybrid.yaml
│   └── inference.yaml
│
├── notebooks/                         # Jupyter notebooks
│   ├── 01_explore_synthetic_data.ipynb
│   ├── 02_visualize_training.ipynb
│   └── 03_analyze_results.ipynb
│
└── tests/                            # Unit tests
    ├── test_synthetic_generator.py
    ├── test_model.py
    └── test_training.py
```

---

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone/download the project
cd piezo1_calcium_tracker

# Create conda environment
conda create -n piezo1_tracker python=3.11
conda activate piezo1_tracker

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Dependencies

Core packages:
- `torch`, `torchvision` - Deep learning
- `numpy`, `scipy` - Numerical computing
- `scikit-image` - Image processing
- `tifffile` - TIFF I/O
- `pandas` - Data handling
- `matplotlib`, `seaborn` - Visualization
- `pyyaml` - Configuration
- `tqdm` - Progress bars
- `tensorboard` - Training monitoring

Optional:
- `thunderstorm` - For ThunderSTORM integration
- `trackpy` - For tracking evaluation
- `jupyterlab` - For notebooks

---

## Quick Start

### 1. Generate Synthetic Training Data

```bash
python scripts/01_generate_synthetic_data.py \
    --config configs/synthetic_data.yaml \
    --output data/synthetic_puncta \
    --num_samples 10000
```

Creates realistic puncta images with ground truth coordinates.

### 2. Extract ThunderSTORM Labels

```bash
python scripts/02_extract_thunderstorm_labels.py \
    --input /path/to/piezo1/images \
    --output data/thunderstorm_labels \
    --thunderstorm_path /path/to/ThunderSTORM
```

Runs ThunderSTORM on real data and extracts localizations as pseudo-labels.

### 3. Prepare Calcium Data

```bash
python scripts/03_prepare_calcium_data.py \
    --input /path/to/calcium/data \
    --output data/calcium_puncta \
    --filter_puncta_only
```

Adapts existing calcium labels to focus on puncta-localized signals only.

### 4. Train Localization Decoder

```bash
python scripts/04_train_localization_only.py \
    --config configs/training_localization.yaml \
    --checkpoint /path/to/pretrained_unet.pth
```

Trains just the localization decoder using your pre-trained U-Net encoder.

### 5. Train Full Hybrid Model

```bash
python scripts/05_train_hybrid_model.py \
    --config configs/training_hybrid.yaml \
    --checkpoint checkpoints/localization_pretrained.pth
```

Trains both decoders jointly with multi-task loss.

### 6. Run Inference

```bash
python scripts/07_run_inference.py \
    --model checkpoints/hybrid_best.pth \
    --input /path/to/test/data \
    --output results/
```

Detects puncta and calcium signals in new data.

---

## Training Approach

### Phase 1: Localization Pre-training (Recommended Start)

**Data**: Synthetic puncta + ThunderSTORM labels  
**Model**: Shared encoder (frozen) + localization decoder  
**Duration**: ~2-3 days  
**Goal**: Learn to localize puncta accurately

### Phase 2: Hybrid Joint Training

**Data**: All three sources  
**Model**: Full hybrid with both decoders  
**Duration**: ~3-5 days  
**Goal**: Optimize both tasks simultaneously

### Phase 3: Fine-tuning

**Data**: Real PIEZO1-calcium pairs  
**Model**: Entire network  
**Duration**: ~1-2 days  
**Goal**: Final performance boost on real data

---

## Key Design Decisions

### Why Hybrid Architecture?

✅ **Leverage existing work**: Use your trained 16-channel U-Net encoder  
✅ **Modular development**: Can train/debug each decoder independently  
✅ **Shared features**: Low-level features benefit both tasks  
✅ **Flexibility**: Easy to add/remove components

### Why Three Data Sources?

✅ **Synthetic**: Unlimited training data, perfect ground truth  
✅ **ThunderSTORM**: Bridge to real data characteristics  
✅ **Calcium labels**: Real puncta-calcium correlations

### Why Puncta-Only Calcium?

✅ **Specificity**: JF646-BAPTA reports calcium AT the channel  
✅ **Simplicity**: Binary classification (signal/no-signal) vs 4-class  
✅ **Correlation**: Directly links puncta to calcium activity

---

## Expected Performance

### Localization
- **Precision**: 20-40 nm (similar to ThunderSTORM)
- **Recall**: >90% for SNR > 5
- **Speed**: ~100 frames/sec on GPU

### Calcium Detection
- **Sensitivity**: >85% for puncta-localized signals
- **Specificity**: >90% (reject diffuse events)
- **Temporal resolution**: Limited by acquisition rate

### Combined System
- **Puncta-calcium correlation**: >80% co-localization
- **Track completeness**: >70% for >10 frame tracks

---

## Troubleshooting

### Common Issues

**Synthetic data doesn't match real data**
→ Adjust PSF parameters, noise levels, background

**Localization decoder won't converge**
→ Start with frozen encoder, higher learning rate for decoder

**Multi-task loss unstable**
→ Use Kendall uncertainty weighting, check gradient magnitudes

**Poor calcium detection**
→ Verify puncta-only filtering, check label quality

---

## Citation

If you use this code, please cite:

```
Your future paper here!
```

And the key methods:
- DECODE: Speiser et al., Nature Methods 2021
- Multi-task learning: Kendall et al., CVPR 2018
- PIEZO1-HaloTag: Bertaccini et al., Nature Commun 2025

---

## License

MIT License - See LICENSE file for details

---

## Contact

George Dickinson  
UC Irvine, Department of Neurobiology & Behavior  
[Your contact info]

---

## Acknowledgments

- Original U-Net calcium detector
- ThunderSTORM team
- DECODE developers
- Bertaccini et al. for PIEZO1-HaloTag methodology
