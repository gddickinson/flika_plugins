# IMPLEMENTATION GUIDE: PIEZO1 Calcium Tracker
# Step-by-Step Guide to Building the Hybrid System

═══════════════════════════════════════════════════════════════════════
QUICK START: 30-MINUTE TEST DRIVE
═══════════════════════════════════════════════════════════════════════

Want to see if this approach works before committing? Here's a minimal test:

## Step 1: Install (5 minutes)

```bash
cd piezo1_calcium_tracker
conda create -n piezo1_tracker python=3.11
conda activate piezo1_tracker
pip install -r requirements.txt
pip install -e .
```

## Step 2: Generate Test Data (10 minutes)

```bash
# Generate 100 synthetic samples (takes ~5-10 min)
python scripts/01_generate_synthetic_data.py \
    --output data/synthetic_test \
    --num_samples 100 \
    --frames_per_sample 10 \
    --image_size 256 256 \
    --num_puncta_mean 10
```

## Step 3: Test Model Architecture (5 minutes)

```bash
# Test the hybrid model
cd piezo1_tracker/models
python hybrid_model.py
```

You should see:
```
Model architecture:
  Encoder params: ~1,200,000
  Loc decoder params: ~800,000
  Ca decoder params: ~400,000
  Total params: ~2,400,000

Output shapes:
  Localization prob: torch.Size([2, 1, 5, 64, 64])
  ...

✅ Model architecture validated
```

If this works → Continue to full implementation below!

═══════════════════════════════════════════════════════════════════════
FULL IMPLEMENTATION: 4-WEEK ROADMAP
═══════════════════════════════════════════════════════════════════════

## Week 1: Data Preparation & Baseline

### Goal: Get all three data sources ready

### Day 1-2: Synthetic Data

```bash
# Generate full synthetic training set (10k samples)
python scripts/01_generate_synthetic_data.py \
    --output data/synthetic_puncta \
    --num_samples 10000 \
    --frames_per_sample 50 \
    --image_size 512 512 \
    --num_puncta_mean 20 \
    --psf_type gaussian
```

**Validation**:
- Check data/synthetic_puncta/sample_00000/
- Should have: movie.tif, ground_truth.csv, metadata.json
- Visualize a few samples in notebook

### Day 3-4: ThunderSTORM Labels

```bash
# Extract puncta from your real PIEZO1 data
python scripts/02_extract_thunderstorm_labels.py \
    --input /path/to/piezo1/images \
    --output data/thunderstorm_labels \
    --batch_size 10
```

**What this does**:
1. Runs ThunderSTORM on each TIFF file
2. Extracts localizations as CSV
3. Converts to same format as synthetic data
4. Creates pseudo-ground-truth for real data

**Requirements**:
- ThunderSTORM plugin installed (Fiji/ImageJ)
- Or use Python wrapper (will create this if needed)

### Day 5-6: Calcium Data Adaptation

```bash
# Adapt your existing calcium labels
python scripts/03_prepare_calcium_data.py \
    --input /path/to/calcium/data_patches \
    --output data/calcium_puncta \
    --filter_puncta_only \
    --ignore_waves
```

**What this does**:
1. Loads your existing calcium event labels
2. Filters to keep only puncta-localized signals
3. Converts to binary (signal/no-signal)
4. Ignores sparks/puffs/waves
5. Creates matched PIEZO1-calcium pairs

**Output**:
- Calcium channel images
- Binary labels (0=no signal, 1=signal at puncta)
- Metadata linking to PIEZO1 frames

### Day 7: Data Validation

Run the exploratory notebook:
```bash
jupyter lab notebooks/01_explore_synthetic_data.ipynb
```

**Check**:
- [ ] Synthetic data looks realistic
- [ ] ThunderSTORM localizations are accurate
- [ ] Calcium labels match puncta locations
- [ ] Data formats are consistent

**Decision point**: If data looks good → Week 2. If issues → debug first.

---

## Week 2: Train Localization Decoder

### Goal: Get puncta detection working before adding complexity

### Day 1-2: Setup Training

Create training configuration:

```yaml
# configs/training_localization.yaml
model:
  in_channels: 1  # Just HaloTag channel
  base_channels: 16
  pretrained_encoder: /path/to/your/unet_16ch/best_model.pth
  freeze_encoder: true  # Start with frozen encoder

data:
  synthetic_dir: data/synthetic_puncta
  thunderstorm_dir: data/thunderstorm_labels
  train_split: 0.9
  batch_size: 4
  num_workers: 4

training:
  num_iterations: 20000
  learning_rate: 1e-3  # Higher for decoder only
  scheduler: cosine
  val_frequency: 500
  save_frequency: 1000

loss:
  detection_weight: 1.0
  offset_weight: 1.0
  photon_weight: 0.1
```

### Day 3-5: Train Localization Only

```bash
python scripts/04_train_localization_only.py \
    --config configs/training_localization.yaml \
    --output checkpoints/localization_only \
    --gpu 0
```

**Monitor**:
- Detection precision/recall
- Localization RMSE vs ground truth
- Check TensorBoard: `tensorboard --logdir checkpoints/localization_only/logs`

**Expected results (by iter 20k)**:
- Detection recall: >90%
- Localization RMSE: <50 nm
- False positive rate: <10%

### Day 6-7: Evaluate & Visualize

```bash
python scripts/06_evaluate_model.py \
    --checkpoint checkpoints/localization_only/best_model.pth \
    --test_data data/synthetic_puncta \
    --output results/localization_only
```

**Check visualizations**:
- Detected puncta overlay on images
- Error maps (GT vs predicted)
- Photon count accuracy

**Decision point**: 
- If RMSE < 50 nm → Continue to Week 3
- If RMSE > 100 nm → Debug (PSF model? Training?)
- If precision < 80% → Check detection threshold

---

## Week 3: Add Calcium Detection

### Goal: Train full hybrid model with both decoders

### Day 1-2: Prepare Hybrid Training

Update configuration:

```yaml
# configs/training_hybrid.yaml
model:
  in_channels: 2  # HaloTag + Calcium
  base_channels: 16
  pretrained_encoder: /path/to/your/unet_16ch/best_model.pth
  pretrained_localization: checkpoints/localization_only/best_model.pth
  freeze_encoder: false  # Now train everything

data:
  paired_data: data/paired_piezo1_calcium  # Matched pairs
  batch_size: 2  # Smaller for dual-channel
  
training:
  num_iterations: 30000
  learning_rate: 1e-4  # Lower for joint training
  
loss:
  use_uncertainty_weighting: true  # Kendall method
  localization_weight: 1.0  # Will be learned
  calcium_weight: 1.0  # Will be learned
```

### Day 3-5: Train Hybrid Model

```bash
python scripts/05_train_hybrid_model.py \
    --config configs/training_hybrid.yaml \
    --output checkpoints/hybrid \
    --gpu 0
```

**Monitor both tasks**:
- Localization: precision, recall, RMSE
- Calcium: F1-score, IoU
- Loss components: localization vs calcium
- Uncertainty weights: log(σ₁), log(σ₂)

**Expected dynamics**:
- First 5k iters: Both tasks learn
- 5k-15k iters: Calcium improves faster (easier task)
- 15k+ iters: Localization catches up
- Uncertainty weights stabilize around iter 10k

### Day 6-7: Evaluate Combined System

```bash
python scripts/06_evaluate_model.py \
    --checkpoint checkpoints/hybrid/best_model.pth \
    --test_data data/paired_test \
    --evaluate_both_tasks \
    --output results/hybrid
```

**Success criteria**:
- [ ] Localization RMSE still <50 nm (not degraded)
- [ ] Calcium F1-score >85%
- [ ] Puncta-calcium correlation >80%
- [ ] Inference speed >50 frames/sec

**Decision point**:
- If both tasks good → Week 4 (real data)
- If one task degraded → Adjust loss weights
- If correlation low → Check temporal alignment

---

## Week 4: Real Data Testing & Refinement

### Goal: Test on real PIEZO1-calcium data

### Day 1-2: Run Inference on Real Data

```bash
python scripts/07_run_inference.py \
    --model checkpoints/hybrid/best_model.pth \
    --input /path/to/real/piezo1_calcium/movies \
    --output results/real_data \
    --save_tracks \
    --save_calcium_events
```

**Outputs**:
- Detected puncta coordinates (CSV)
- Tracked trajectories (linked over time)
- Calcium event detections (when/where)
- Visualizations (overlays)
- Statistics (correlation analysis)

### Day 3-4: Validate Results

Compare against your existing methods:
- ThunderSTORM detections
- Manual annotations
- Your original U-Net calcium detector

**Metrics to check**:
- Detection rate vs ThunderSTORM
- False positive rate
- Tracking continuity
- Calcium-puncta co-occurrence

### Day 5-6: Fine-tune on Real Data

If performance gaps exist:

```bash
# Fine-tune on small set of manually validated real data
python scripts/05_train_hybrid_model.py \
    --config configs/training_hybrid.yaml \
    --checkpoint checkpoints/hybrid/best_model.pth \
    --finetune_data data/validated_real_data \
    --num_iterations 5000 \
    --learning_rate 1e-5  # Much lower
    --output checkpoints/hybrid_finetuned
```

### Day 7: Publication-Quality Analysis

Run full analysis pipeline:
```bash
python scripts/08_analyze_results.py \
    --detections results/real_data/puncta.csv \
    --tracks results/real_data/trajectories.csv \
    --calcium results/real_data/calcium_events.csv \
    --output figures/
```

**Creates**:
- Trajectory plots (mobile vs immobile)
- Calcium correlation heatmaps
- Localization precision histograms
- Co-occurrence statistics
- Publication-quality figures

═══════════════════════════════════════════════════════════════════════
TROUBLESHOOTING COMMON ISSUES
═══════════════════════════════════════════════════════════════════════

## Issue 1: Synthetic data doesn't match real data

**Symptoms**: Good performance on synthetic, terrible on real

**Solutions**:
1. Adjust PSF parameters:
   ```python
   # In synthetic generator
   sigma_nm = 150  # Increase if real PSFs are broader
   ```

2. Add more realistic noise:
   ```python
   baseline = 150  # Increase background
   read_noise_std = 3.0  # Increase noise
   ```

3. Use bootstrapped ThunderSTORM labels instead

## Issue 2: Multi-task loss unstable

**Symptoms**: One task improves, other gets worse

**Solutions**:
1. Check gradient magnitudes:
   ```python
   # In training loop
   print(f"Loc grad: {loc_grad_norm:.2e}")
   print(f"Ca grad: {ca_grad_norm:.2e}")
   ```

2. Adjust Kendall weights:
   ```python
   # If calcium dominates
   log_var_calcium.data += 1.0  # Reduce calcium influence
   ```

3. Train decoders separately first, then jointly

## Issue 3: Localization precision poor (RMSE > 100 nm)

**Symptoms**: Detections are offset from ground truth

**Solutions**:
1. Check PSF sigma matches your microscope
2. Increase model capacity (base_channels = 32)
3. Train longer (50k+ iterations)
4. Use Airy PSF instead of Gaussian

## Issue 4: Calcium detection only finds background

**Symptoms**: All pixels predicted as "no signal"

**Solutions**:
1. Check class balance:
   ```python
   # In dataset
   print(f"Signal: {(labels == 1).sum()}")
   print(f"No signal: {(labels == 0).sum()}")
   ```

2. Use focal loss instead of cross-entropy
3. Verify calcium labels are correctly binary
4. Check that puncta-only filtering worked

## Issue 5: Training too slow

**Symptoms**: <1 iter/sec on GPU

**Solutions**:
1. Reduce image size: 512→256
2. Reduce batch size: 4→2
3. Reduce num_workers: 4→0
4. Use mixed precision:
   ```python
   from torch.cuda.amp import autocast
   with autocast():
       outputs = model(inputs)
   ```

═══════════════════════════════════════════════════════════════════════
ADVANCED: OPTIMIZATION STRATEGIES
═══════════════════════════════════════════════════════════════════════

## Strategy 1: Progressive Training

Train components in stages:

```
Stage 1: Localization on synthetic (freeze encoder)
         → 20k iters, RMSE < 50 nm

Stage 2: Localization on real (ThunderSTORM labels)
         → 10k iters, finetune

Stage 3: Add calcium (freeze encoder + loc decoder)
         → 20k iters, calcium only

Stage 4: Joint training (everything unfrozen)
         → 30k iters, both tasks
```

## Strategy 2: Curriculum Learning

Start easy, increase difficulty:

```python
# In training loop
if iteration < 10000:
    # Easy: Low puncta density, high SNR
    num_puncta = 5
    photon_counts = 2000
elif iteration < 20000:
    # Medium: Normal density
    num_puncta = 15
    photon_counts = 1000
else:
    # Hard: High density, low SNR
    num_puncta = 30
    photon_counts = 500
```

## Strategy 3: Ensemble Localization

Average multiple models for better precision:

```python
# Train 5 models with different seeds
models = [load_model(f'model_{i}.pth') for i in range(5)]

# Average predictions
with torch.no_grad():
    preds = [model(x)['localization'] for model in models]
    avg_prob = torch.stack([p['prob'] for p in preds]).mean(0)
    avg_offset = torch.stack([p['offset'] for p in preds]).mean(0)
```

Typically improves RMSE by 5-10%

═══════════════════════════════════════════════════════════════════════
EXPECTED PERFORMANCE BENCHMARKS
═══════════════════════════════════════════════════════════════════════

Based on literature and similar systems:

## Localization (vs ThunderSTORM)

┌──────────────────┬────────────┬────────────┬──────────┐
│ Metric           │ Target     │ Good       │ Excellent│
├──────────────────┼────────────┼────────────┼──────────┤
│ RMSE (nm)        │ < 50       │ < 30       │ < 20     │
│ Precision        │ > 85%      │ > 90%      │ > 95%    │
│ Recall           │ > 85%      │ > 90%      │ > 95%    │
│ Speed (fps)      │ > 20       │ > 50       │ > 100    │
└──────────────────┴────────────┴────────────┴──────────┘

## Calcium Detection (vs manual annotations)

┌──────────────────┬────────────┬────────────┬──────────┐
│ Metric           │ Target     │ Good       │ Excellent│
├──────────────────┼────────────┼────────────┼──────────┤
│ F1-score         │ > 80%      │ > 85%      │ > 90%    │
│ Sensitivity      │ > 85%      │ > 90%      │ > 95%    │
│ Specificity      │ > 80%      │ > 85%      │ > 90%    │
│ Temporal res     │ Frame rate │ Frame rate │ Sub-frame│
└──────────────────┴────────────┴────────────┴──────────┘

## Combined System

┌──────────────────────┬────────────┬────────────┬──────────┐
│ Metric               │ Target     │ Good       │ Excellent│
├──────────────────────┼────────────┼────────────┼──────────┤
│ Puncta-Ca correlation│ > 70%      │ > 80%      │ > 90%    │
│ Track completeness   │ > 60%      │ > 75%      │ > 85%    │
│ Co-localization      │ < 200 nm   │ < 100 nm   │ < 50 nm  │
└──────────────────────┴────────────┴────────────┴──────────┘

═══════════════════════════════════════════════════════════════════════
NEXT FILES TO CREATE
═══════════════════════════════════════════════════════════════════════

I've created the core architecture. You'll need these additional scripts:

Priority 1 (Essential):
[ ] scripts/02_extract_thunderstorm_labels.py
[ ] scripts/03_prepare_calcium_data.py
[ ] scripts/04_train_localization_only.py
[ ] scripts/05_train_hybrid_model.py

Priority 2 (Important):
[ ] piezo1_tracker/training/losses.py (multi-task loss)
[ ] piezo1_tracker/training/trainer.py (training loop)
[ ] scripts/06_evaluate_model.py
[ ] scripts/07_run_inference.py

Priority 3 (Nice to have):
[ ] piezo1_tracker/inference/tracking.py (LAP tracking)
[ ] notebooks/01_explore_synthetic_data.ipynb
[ ] notebooks/02_visualize_training.ipynb

Would you like me to create any of these next?

═══════════════════════════════════════════════════════════════════════
DECISION TREE: IS THIS APPROACH RIGHT FOR YOU?
═══════════════════════════════════════════════════════════════════════

✅ YES, use this approach if:
   - You want sub-pixel puncta localization (< 50 nm)
   - You have dual-channel data (PIEZO1 + calcium)
   - You want automatic puncta-calcium correlation
   - You're willing to invest 4 weeks of development
   - You have GPU for training

⚠️  MAYBE, consider alternatives if:
   - You only need pixel-level localization (>100 nm)
     → Simpler: Just use ThunderSTORM + your U-Net
   - You don't have matched dual-channel data
     → Pipeline: Process channels separately
   - Limited compute resources
     → Use pre-trained DECODE + existing U-Net

❌ NO, use different approach if:
   - You only need calcium detection (no puncta)
     → Just use your existing 16-channel U-Net!
   - You only need puncta tracking (no calcium)
     → Use ThunderSTORM + trackpy
   - You need 3D localization
     → Use DeepSTORM3D instead

═══════════════════════════════════════════════════════════════════════

Ready to start? Run the 30-minute test drive above!
Questions? Issues? Let me know what you need next.
