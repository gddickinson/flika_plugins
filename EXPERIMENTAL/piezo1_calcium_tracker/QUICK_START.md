# QUICK START GUIDE

Get the PIEZO1 Calcium Tracker running in 30 minutes!

## Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- ~10 GB disk space for initial testing

## Installation (5 minutes)

```bash
# Extract the project
tar -xzf piezo1_calcium_tracker.tar.gz
cd piezo1_calcium_tracker

# Create conda environment
conda create -n piezo1_tracker python=3.11
conda activate piezo1_tracker

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Test Installation (2 minutes)

```bash
# Test PSF models
cd piezo1_tracker/utils
python psf_models.py
```

You should see: `Saved psf_comparison.png`

```bash
# Test hybrid model
cd ../models
python hybrid_model.py
```

You should see: `✅ Model architecture validated`

## Generate Test Data (10 minutes)

```bash
# Return to project root
cd ../..

# Generate 100 synthetic samples for testing
python scripts/01_generate_synthetic_data.py \
    --output data/synthetic_test \
    --num_samples 100 \
    --frames_per_sample 10 \
    --image_size 256 256 \
    --num_puncta_mean 10 \
    --seed 42
```

Expected output:
```
Generating 100 synthetic samples...
Gaussian PSF: σ = 137.7 nm (1.06 px)
Generating samples: 100%|███████| 100/100
✅ Generated 100 samples
   Total puncta: ~10,000
```

## Verify Data (5 minutes)

```bash
# Check the generated data
ls data/synthetic_test/sample_00000/
```

You should see:
- `movie.tif` - The synthetic puncta movie
- `ground_truth.csv` - Exact puncta positions
- `metadata.json` - Sample information

```bash
# Quick visualization (optional)
python -c "
import tifffile
import matplotlib.pyplot as plt

movie = tifffile.imread('data/synthetic_test/sample_00000/movie.tif')
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(movie[0], cmap='hot')
plt.title('Frame 0')
plt.subplot(122)
plt.imshow(movie[5], cmap='hot')
plt.title('Frame 5')
plt.tight_layout()
plt.savefig('test_frames.png')
print('Saved test_frames.png')
"
```

## Test Multi-Task Loss (3 minutes)

```bash
cd piezo1_tracker/training
python losses.py
```

You should see:
```
Testing individual losses:
Localization loss: 0.xxxx
Calcium loss: 0.xxxx
...
✅ All tests passed!
```

## Next Steps

✅ **If all tests passed**, you're ready to:

1. **Read IMPLEMENTATION_GUIDE.md** for the 4-week roadmap
2. **Generate full synthetic dataset** (10k samples)
3. **Start training the localization decoder**

✅ **To continue with real data**:

1. Extract ThunderSTORM labels from your PIEZO1 data
2. Prepare your calcium-labeled data
3. Train the hybrid model

⚠️ **If tests failed**, check:

- GPU available: `python -c "import torch; print(torch.cuda.is_available())"`
- Packages installed: `pip list | grep torch`
- File permissions: `ls -la scripts/`

## Quick Reference

### Key Files

- `README.md` - Project overview
- `IMPLEMENTATION_GUIDE.md` - **START HERE** for detailed instructions
- `requirements.txt` - Python dependencies
- `setup.py` - Package installation

### Key Scripts

- `scripts/01_generate_synthetic_data.py` - Create synthetic training data
- `scripts/02_extract_thunderstorm_labels.py` - Get ThunderSTORM labels (TODO)
- `scripts/03_prepare_calcium_data.py` - Prepare calcium data (TODO)
- `scripts/04_train_localization_only.py` - Train localization first (TODO)
- `scripts/05_train_hybrid_model.py` - Train full hybrid model (TODO)

### Key Modules

- `piezo1_tracker/models/hybrid_model.py` - **Main architecture**
- `piezo1_tracker/training/losses.py` - **Multi-task loss**
- `piezo1_tracker/utils/psf_models.py` - PSF simulation

## Getting Help

### Common Issues

**Import errors**
```bash
# Reinstall in development mode
pip install -e .
```

**CUDA out of memory**
```bash
# Reduce batch size or image size in configs
```

**Slow data generation**
```bash
# Reduce num_samples or use multiple processes
```

### Questions?

1. Check `IMPLEMENTATION_GUIDE.md` troubleshooting section
2. Review the specific module's documentation
3. Ask for help with specific error messages

## Success Checklist

After running this quick start, you should have:

- [ ] Package installed and importable
- [ ] PSF models working (psf_comparison.png created)
- [ ] Hybrid model architecture validated
- [ ] 100 synthetic samples generated
- [ ] Multi-task loss tested
- [ ] Ready to proceed with full implementation

---

**Total time**: ~30 minutes  
**Next step**: Read `IMPLEMENTATION_GUIDE.md` for Week 1 → Week 4 roadmap

Good luck! 🚀
