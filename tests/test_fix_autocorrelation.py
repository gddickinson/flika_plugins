"""Regression: spt_batch_analysis AutocorrelationAnalyzer (directional autocorrelation, Gorelik & Gautreau 2014).

1. process_track_data concatenates the tracks and calculate_normed_vectors found track boundaries only where the frame
   number does not increase. A track that starts at a later frame than the previous track ended was therefore joined
   onto it: the jump between the two tracks became a step, and its direction was correlated with both tracks.
2. calculate_scalar_products stopped at lag num_intervals - 1, so the last requested lag was always empty (NaN).
   The same off-by-one was fixed in direction_autocorrelation_analysis (bd451bf).
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from web.source_loader import load  # noqa: E402

AA = load(ROOT / "spt_batch_analysis/__init__.py", ["AutocorrelationAnalyzer"])["AutocorrelationAnalyzer"]


def _two_tracks():
    a = pd.DataFrame({"track_number": 1, "frame": range(0, 10), "x": np.arange(10.0), "y": 5.0})
    b = pd.DataFrame({"track_number": 2, "frame": range(20, 30), "x": 40.0, "y": 30 + np.arange(10.0)})
    return pd.concat([a, b], ignore_index=True)


def _run(df, n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return AA.process_track_data(df, 1, n)


def test_tracks_starting_later_are_not_joined():
    products, averages, individual = _run(_two_tracks(), 3)
    assert individual["track_id"].nunique() == 2
    # each track is a straight line, so every lag-1 direction pair is identical
    np.testing.assert_allclose(products[1].dropna(), 1.0)
    assert averages.loc["AVG", 1] == 1.0


def test_last_lag_is_computed():
    products, averages, _ = _run(_two_tracks(), 3)
    assert products[3].notna().sum() == 2 * 6   # 9 vectors per track -> 6 pairs at lag 3
    assert np.isfinite(averages.loc["AVG", 3])
