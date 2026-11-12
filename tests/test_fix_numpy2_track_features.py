"""Regression: the track-feature functions must run on NumPy >= 2.5.

Two NumPy 2 changes broke them:
- np.cross of 2-vectors (deprecated in 2.0, an error in 2.5) in summed_sines_cosines / SummedSinesCosines;
- np.linalg.eig returns complex arrays in 2.5 even for the real symmetric gyration tensor, so math.log, max() and
  scipy.stats.skew failed or returned complex numbers in radius_gyration_asymmetry /
  RadiusGyrationAsymmetrySkewnessKurtosis.
The fix uses the explicit 2D cross product and the real part of eig (the tensor is symmetric, so it is exact), which
leaves every value that the old code computed on NumPy < 2.5 unchanged.
"""
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from web.source_loader import load  # noqa: E402

TRACK = np.array([[10.0, 10.0], [11.2, 10.4], [12.1, 11.9], [12.0, 13.1], [13.5, 13.0], [15.1, 14.2], [15.0, 16.0]])
DF = pd.DataFrame({"x": TRACK[:, 0], "y": TRACK[:, 1]})


def _expected():
    c = TRACK - TRACK.mean(0)
    t = c.T @ c / len(TRACK)
    w, v = np.linalg.eigh(t)
    ab, bc = TRACK[1:-1] - TRACK[:-2], TRACK[2:] - TRACK[1:-1]
    mag = np.linalg.norm(ab, axis=1) * np.linalg.norm(bc, axis=1)
    sin = (ab[:, 0] * bc[:, 1] - ab[:, 1] * bc[:, 0]) / mag
    asym = -math.log(1 - (w[0] - w[1]) ** 2 / (2 * (w[0] + w[1]) ** 2))
    return math.sqrt(w.sum()), asym, sin


@pytest.fixture(params=["spt", "join", "join_original"])
def fns(request):
    if request.param == "spt":
        fc = load(ROOT / "spt_batch_analysis/__init__.py", ["FeatureCalculator"])["FeatureCalculator"]
        return fc.radius_gyration_asymmetry, fc.summed_sines_cosines
    folder = "locsAndTracksPlotter" + ("/locsAndTracksPlotter_original" if request.param == "join_original" else "")
    f = load(ROOT / folder / "joinTracks.py",
             ["JoinTracks.RadiusGyrationAsymmetrySkewnessKurtosis", "JoinTracks.SummedSinesCosines"])
    return (lambda df: f["JoinTracks.RadiusGyrationAsymmetrySkewnessKurtosis"](None, df),
            lambda pts: f["JoinTracks.SummedSinesCosines"](None, pts))


def test_features_are_real_and_correct(fns):
    rg_fn, sc_fn = fns
    rg_exp, asym_exp, sin_exp = _expected()
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        rg, asym, skew, kurt = rg_fn(DF)
        sin_mean, sin_vals, cos_mean, _ = sc_fn(TRACK.copy())
    for v in (rg, asym, skew, kurt, sin_mean, cos_mean):
        assert not np.iscomplexobj(v) and np.isfinite(v)
    assert rg == pytest.approx(rg_exp, rel=1e-12)
    assert asym == pytest.approx(asym_exp, rel=1e-9)
    np.testing.assert_allclose(sin_vals, sin_exp, rtol=1e-12)
