"""Regression: kymograph_analyzer detect_events ('threshold' and 'zscore' methods) paired crossings wrongly.

1. The up- and down-crossings were zipped in order. When a trace starts above threshold, its first down-crossing
   has no up-crossing, so every event was paired with the previous event's end (negative durations, events lost).
2. np.diff(above) == 1 at index i means frame i+1 is the first frame above threshold, but i was used as the start:
   the reported time was the last frame below threshold, and the amplitude window trace[start:end] included that
   frame and missed the last frame above threshold.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from web.source_loader import load  # noqa: E402

detect = load(ROOT / "kymograph_analyzer/kymograph_analyzer.py",
              ["_event_bounds", "KymographAnalyzer.detect_events"])["KymographAnalyzer.detect_events"]


def _trace():
    t = np.zeros(60)
    t[0:4] = 10.0      # already above threshold at the start
    t[20:26] = [6, 8, 10, 9, 7, 6]
    t[40:44] = [6, 9, 7, 6]
    return t


@pytest.mark.parametrize("method,thr", [("threshold", 1.0), ("zscore", 1.0)])
def test_events_are_paired_and_start_on_the_first_frame_above(method, thr):
    me = SimpleNamespace(kymograph=None, analysis_results={})
    events = detect(me, kymo=_trace()[:, None], method=method, threshold=thr, min_duration=3)
    assert [(e["time"], e["duration"], e["amplitude"]) for e in events] == [(20, 6, 10.0), (40, 4, 9.0)]
