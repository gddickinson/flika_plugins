"""Regression: fft_Chunker makeAverage kept only the last ROI trace.

`self.averageResults_dict.update(...)` sat after the `for key in self.result_dict` loop instead of inside it, so with
several ROI traces in the input file only the last trace's group averages were stored, and exportAverage wrote one
file instead of one per trace.
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
def test_every_trace_gets_its_averages():
    t = np.arange(400) * 0.01
    traces = {0: np.sin(2 * np.pi * 5 * t), 1: 0.5 * np.sin(2 * np.pi * 12 * t)}
    result = {k: F["fft_chunks"](v, 100, 0.01)[0] for k, v in traces.items()}
    groups = dict(baseline=(1, 2), puff1=(3, 4), puff2=(0, 0), puff3=(0, 0), puff4=(0, 0))
    me = SimpleNamespace(result_dict=result, **{f"{g}_{e}": _box(v[i]) for g, v in groups.items()
                                                for i, e in enumerate(("start", "stop"))})
    F["FFT_Chunker.makeAverage"](me)
    assert set(me.averageResults_dict) == {"AverageResult_0", "AverageResult_1"}
    a0, a1 = (me.averageResults_dict[f"AverageResult_{k}"]["baseline_PowerAverage"] for k in (0, 1))
    assert not np.allclose(a0, a1, equal_nan=True)
