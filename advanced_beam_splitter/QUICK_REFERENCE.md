# Advanced Beam Splitter - Quick Reference Card

## Essential Keyboard Shortcuts

| Key | Action | Details |
|-----|--------|---------|
| **Arrow Keys** | Translate | ↑↓←→ move image 1 pixel |
| **R** | Rotate CCW | Rotate 0.5° counterclockwise |
| **T** | Rotate CW | Rotate 0.5° clockwise |
| **+ / =** | Scale Up | Increase scale by 0.01 |
| **-** | Scale Down | Decrease scale by 0.01 |
| **A** | Auto-Align | Automatic correlation-based alignment |
| **U** | Undo/Revert | Return to original images |
| **Enter** | Apply | Apply transformations and close |

## Typical Workflow

### 1. Calibration (with beads)
```
Load bead images → Auto-Align (A) → Fine-tune manually → 
Record parameters → Apply
```

### 2. Experimental Data
```
Load images → Enter calibration parameters → 
Select background method → Enable bleach correction (if time-lapse) → 
Enable normalization (if quantitative) → Apply
```

## Parameter Quick Guide

### Background Subtraction

| Method | Best For | Typical Radius |
|--------|----------|----------------|
| **rolling_ball** | Uneven illumination | 50-100 pixels |
| **gaussian** | Smooth gradients | 20-50 pixels |
| **manual** | Simple threshold | 5% percentile |

### Photobleaching Correction

| Method | Best For |
|--------|----------|
| **exponential** | Uniform, predictable bleaching |
| **histogram** | Complex or non-uniform bleaching |

### Scale Factor
- Typically 0.95 - 1.05
- Measure from bead calibration
- Accounts for magnification differences

### Rotation
- Usually < 5° unless mechanical misalignment
- Measure from calibration or straight edges

## Common Issues & Solutions

| Problem | Solution |
|---------|----------|
| Auto-align fails | Manual coarse align first, then auto |
| Too much background removed | Lower radius/sigma parameter |
| Preview blank | Check x/y shift values aren't too large |
| Memory error | Process smaller subsets |

## Processing Tips

✓ **Always** calibrate with beads first  
✓ **Keep** imaging parameters consistent  
✓ **Document** transformation parameters  
✓ **Validate** corrections on controls  
✓ **Save** raw data separately  

## RGB Overlay Colors
- **Red channel**: Reference (align TO)
- **Green channel**: Transformed (aligned)
- **Yellow overlap**: Perfect alignment

## Plugin Features Checklist

- [x] XY translation alignment
- [x] Rotation correction  
- [x] Scale/magnification correction
- [x] Live preview with RGB overlay
- [x] Rolling ball background subtraction
- [x] Gaussian background subtraction
- [x] Manual background subtraction
- [x] Exponential photobleaching correction
- [x] Histogram photobleaching correction
- [x] Intensity normalization
- [x] Auto-alignment (cross-correlation)
- [x] Revert to original capability
- [x] Keyboard shortcuts
- [x] Batch processing support

---

## Quick Python Example

```python
from plugins.advanced_beam_splitter import advanced_beam_splitter

# Apply with standard parameters
result = advanced_beam_splitter(
    red_window=red_win,
    green_window=green_win,
    x_shift=5,
    y_shift=-3,
    rotation=1.2,
    scale_factor=0.998,
    background_method='rolling_ball',
    background_radius=50,
    photobleach_correction='exponential',
    normalize_intensity=True
)
```

---

**For full documentation, see README.md**

**Questions? Check the FLIKA documentation or forums**
