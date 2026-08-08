"""Regression tests for the photon/background-SD conversion in the precision calculation.

These pin two behaviours that were previously wrong and silently so:

1. Camera calibration must reach the uncertainty. Before the fix, changing
   `photons_per_adu` by 25x moved the reported uncertainty by <0.1 nm, because the fitter
   passed raw ADU to a function documented as taking photons.
2. The background term must be a standard deviation, not the fitted DC offset. The formula
   uses b^2 as a variance; passing the offset inflated it by the ratio of level to variance
   (~32x on Andor iXon data), which dominated the result.
"""
import numpy as np
import pytest
from thunderstorm_python.fitting import BaseFitter, compute_localization_precision


def _fitter(**kw):
    f = BaseFitter()
    f.set_camera_params(**kw)
    return f


def test_photon_conversion_is_applied():
    """Uncertainty must respond to the camera calibration."""
    lo = _fitter(pixel_size=160.0, photons_per_adu=1.0, baseline=116.0)
    hi = _fitter(pixel_size=160.0, photons_per_adu=11.8 / 300, baseline=116.0)
    n_lo, b_lo = lo._photons_and_background_sd(3231.0, 324.3)
    n_hi, b_hi = hi._photons_and_background_sd(3231.0, 324.3)
    assert n_lo > n_hi * 20, 'photons_per_adu must scale the photon count'
    u_lo = compute_localization_precision(n_lo, b_lo, 1.5, 1.0)
    u_hi = compute_localization_precision(n_hi, b_hi, 1.5, 1.0)
    # fewer photons must give worse precision
    assert u_hi > u_lo


def test_background_is_a_standard_deviation_not_the_offset():
    """b must be sqrt(level), so it is far below the fitted offset."""
    f = _fitter(pixel_size=160.0, photons_per_adu=11.8 / 300, baseline=116.0)
    _, b_sd = f._photons_and_background_sd(3231.0, 324.3)
    level_photons = (324.3 - 116.0) * 11.8 / 300
    assert b_sd == pytest.approx(np.sqrt(level_photons), rel=1e-6)
    assert b_sd < level_photons, 'a shot-noise SD is below the level it comes from'


def test_baseline_subtraction_cannot_go_negative():
    f = _fitter(pixel_size=160.0, photons_per_adu=0.04, baseline=500.0)
    n, b = f._photons_and_background_sd(100.0, 300.0)   # offset below baseline
    assert b == 0.0 and n > 0


def test_uncalibrated_defaults_still_work():
    """With defaults the helper must be a no-op on intensity and not raise."""
    f = BaseFitter()
    n, b = f._photons_and_background_sd(1000.0, 50.0)
    assert n == 1000.0
    assert b == pytest.approx(np.sqrt(50.0))
