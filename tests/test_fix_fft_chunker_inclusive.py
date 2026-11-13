"""Regression: fft_Chunker group ranges include the stop chunk (George, 2026-11-12).

getAverage(start, stop) looped over range(start, stop): the stop chunk was left out, so the last chunk (the spinbox
maximum) could never be in a group, and start == stop (allowed by the `stop >= start` check) crashed with
UnboundLocalError on an empty range. Groups now average chunks start..stop inclusive; start == stop is one chunk.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from web.source_loader import load  # noqa: E402

F = load(ROOT / "fft_Chunker/fft_Chunker.py", ["fft_chunks", "FFT_Chunker.makeAverage"])


def _box(v):
    return SimpleNamespace(value=lambda: v)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # the plugin prints log10 of a 0 Hz frequency
def test_stop_chunk_is_included_and_single_chunk_groups_work():
    t = np.arange(400) * 0.01
    df = F["fft_chunks"](np.sin(2 * np.pi * 5 * t) + 0.3 * np.cos(2 * np.pi * 9 * t ** 1.5), 100, 0.01)[0]
    groups = dict(baseline=(1, 1), puff1=(2, 3), puff2=(4, 4), puff3=(0, 0), puff4=(1, 4))
    me = SimpleNamespace(result_dict={0: df}, **{f"{g}_{e}": _box(v[i]) for g, v in groups.items()
                                                 for i, e in enumerate(("start", "stop"))})
    F["FFT_Chunker.makeAverage"](me)
    r = me.averageResults_dict["AverageResult_0"]
    p = {i: df[f"power_{i}"].to_numpy() for i in range(1, 5)}
    np.testing.assert_array_equal(r["baseline_PowerAverage"], p[1])
    np.testing.assert_allclose(r["puff1_PowerAverage"], (p[2] + p[3]) / 2)
    np.testing.assert_array_equal(r["puff2_PowerAverage"], p[4])
    np.testing.assert_allclose(r["puff4_PowerAverage"], (p[1] + p[2] + p[3] + p[4]) / 4)
    assert "puff3_PowerAverage" not in r
