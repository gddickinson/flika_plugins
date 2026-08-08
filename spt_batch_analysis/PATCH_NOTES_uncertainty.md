# Patch — localisation uncertainty was not using photons or a background SD

**Date:** 2026-08-07
**Files:** `thunderstorm_python/fitting.py`, `tests/test_uncertainty_units.py` (new)
**Backup of the original:** `thunderstorm_python/fitting.py.bak`

## What was wrong

`compute_localization_precision` implements the Thompson form used by ThunderSTORM,

    sigma = sqrt( sa2/N * (F + 4*tau) ),   tau = 2*pi*sa2*b^2 / (N*a^2)

and its docstring documents `intensity` as **photons** and `background` as **background
standard deviation per pixel** (it enters as `b^2`, a variance). Both call-site
assumptions were wrong:

1. **Camera calibration never reached the calculation.** `set_camera_params` stored
   `photons_per_adu`, `baseline` and `em_gain`, but all five call sites passed the fitter's
   raw **ADU** values straight through. Measured effect: sweeping `photons_per_adu` across a
   25-fold range and toggling `is_emccd` changed the reported uncertainty by **less than
   0.1 nm**. The calibration was inert.

2. **The fitted DC offset was passed as the background SD.** `results[:, 5]` is the offset
   term of the Gaussian model — the background *level* — not its standard deviation. On
   Andor iXon data the fitted offset was **324 ADU** against a true per-pixel noise SD of
   **57 ADU**, so `b^2` was inflated ~32-fold. This dominated the result.

## The fix

A single helper on `BaseFitter`, applied at all five call sites:

```python
def _photons_and_background_sd(self, intensity_adu, background_adu):
    ppa = self.photons_per_adu if self.photons_per_adu > 0 else 1.0
    n_photons = max(float(intensity_adu), 0.0) * ppa
    b_level = max(float(background_adu) - self.baseline, 0.0) * ppa
    return n_photons, np.sqrt(b_level)
```

The background standard deviation is taken as `sqrt(level)` because a shot-noise-limited
background at B photons per pixel has variance B. Baseline subtraction is clamped at zero so
a pre-subtracted stack cannot produce a negative level.

Call sites patched: `gaussian_lsq`, `gaussian_wlsq`, `gaussian_mle`,
`elliptical_gaussian_mle`, and the multi-emitter fitter.

## Effect, on real data

150317 STORM (Andor iXon DU-897, EM gain 300, 1x conversion gain, 160 nm pixels):

| photons/ADU | EMCCD | before | after |
|---|---|---|---|
| 1.0 | no | 184.8 nm | 9.2 nm |
| 0.0393 | no | 184.8 nm | 46.5 nm |
| 0.0393 | yes | 184.8 nm | 51.0 nm |

The reported precision now responds to the calibration, which is the point. Insight3 on the
same file reports 11.3 nm.

## Behaviour change to be aware of

**This changes reported uncertainties for existing users**, including anyone running with
default camera parameters, because the background term is now `sqrt(level)` rather than the
level itself. The new values are the ones the documented formula intends. Previously stored
uncertainty columns are not comparable with new ones.

## What this does NOT fix

With the correct calibration the pipeline reports ~51 nm where Insight3 reports 11.3 nm on
the same file. That residual is **not** in the precision formula. The fitter returns a median
PSF sigma of **245 nm** where an NA 1.49 objective at 160 nm pixels should give ~120-150 nm,
and localisation counts run 4-11x above the Insight3 export. Both point at the detection and
filtering stage finding extended structures rather than single molecules. Suggested next
step: sweep filter type, threshold expression and detector against the same 150317 reference
until sigma and count match, then re-check precision.

## Tests

`tests/test_uncertainty_units.py` pins four behaviours: the photon conversion is applied and
scales the count; `b` is `sqrt(level)` and below the level; baseline subtraction cannot go
negative; and uncalibrated defaults remain a no-op on intensity. Full suite: 8 passed.
