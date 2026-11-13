"""Regression: kymograph_analyzer measure_velocities reports the speed of a moving feature (George, 2026-11-12).

For intensity carried along the line at speed v, I(t, x) = f(x - v t), so dI/dt = -v dI/dx and v = -(dI/dt)/(dI/dx).
The code computed (dI/dx)/(dI/dt + 1e-10), i.e. -1/v, and averaged it over every pixel (including flat background,
where both gradients are ~0). Now the map is -(dI/dt)/(dI/dx) where the spatial gradient is not negligible, and
'velocity' is the least-squares estimate -sum(It Ix)/sum(Ix^2) over the kymograph.
Units: samples along the line per kymograph row (row = temporal_binning frames); pass spacing and row_time to get
physical units. Positive = moving towards the end of the line.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from web.source_loader import load  # noqa: E402

measure = load(ROOT / "kymograph_analyzer/kymograph_analyzer.py",
               ["KymographAnalyzer.measure_velocities"])["KymographAnalyzer.measure_velocities"]


def kymo(v, rows=60, cols=120, x0=30.0, sigma=4.0, background=10.0):
    t, x = np.mgrid[0:rows, 0:cols].astype(float)
    return background + 100 * np.exp(-0.5 * ((x - x0 - v * t) / sigma) ** 2)


@pytest.mark.parametrize("v", [0.7, -0.4, 1.3])
def test_velocity_of_a_moving_spot(v):
    me = SimpleNamespace(kymograph=None, analysis_results={})
    r = measure(me, kymo=kymo(v, x0=30.0 if v > 0 else 90.0))
    assert r["velocity"] == pytest.approx(v, rel=0.02)
    assert r["median_velocity"] == pytest.approx(v, rel=0.05)
    assert r["units"] == "samples/row"


def test_physical_units():
    me = SimpleNamespace(kymograph=None, analysis_results={})
    r = measure(me, kymo=kymo(0.7), spacing=0.16, row_time=0.05)   # um per sample, s per row
    assert r["velocity"] == pytest.approx(0.7 * 0.16 / 0.05, rel=0.02) and r["units"] == "distance/time"
