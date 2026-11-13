"""The demo fixture is still what the plugin code computes, and the demo data and registry point at real things.

tests/fixtures/demo_cases.json (and localization_frames.json, the input frames) are what the node tests compare the
JS ports against. Regenerate with: python scripts/make_demo_fixture.py
Floats are compared with rtol 1e-6 / atol 1e-9. The localization part involves discrete choices from float maths
(local maxima, accept/reject steps of the fits) and numba's fastmath, so it is skipped when CI is set; the node tests
still check the JS against the committed fixture there.
"""
import json
import math
import os
import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import make_demo_fixture as mk  # noqa: E402
from web.demos import DEMOS, localization_data, track_data  # noqa: E402
from web.source_loader import load  # noqa: E402

FIXTURE = json.loads((ROOT / "tests/fixtures/demo_cases.json").read_text())


def _close(a, b, path="$"):
    if isinstance(b, dict):
        assert set(a) == set(b), path
        for k in b:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(b, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(b, float) and isinstance(a, (int, float)):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), f"{path}: {a} vs {b}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _fresh(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy's skew/kurtosis of constant data, log10(0) in the plugin's prints
        return mk.clean(json.loads(json.dumps(mk.clean(fn()))))


def test_tracks_fixture_is_current():
    _close(_fresh(mk.tracks), FIXTURE["tracks"])


def test_kymograph_fixture_is_current():
    _close(_fresh(mk.kymograph), FIXTURE["kymograph"])


def test_fft_fixture_is_current():
    _close(_fresh(mk.fft), FIXTURE["fft"])


@pytest.mark.skipif(bool(os.environ.get("CI")), reason="discrete detections from float maths; node tests cover CI")
def test_localization_fixture_is_current():
    pytest.importorskip("skimage")
    _close(_fresh(mk.localization), FIXTURE["localization"])


def test_frames_fixture_matches_demo_data():
    frames = json.loads((ROOT / "tests/fixtures/localization_frames.json").read_text())
    _close(json.loads(json.dumps(localization_data())), frames)


def test_demo_data():
    loc = localization_data()
    assert [s["id"] for s in loc["sources"]] == ["real", "synthetic"]
    real = loc["sources"][0]
    assert real["camera"] == {"photons_per_adu": 0.01, "baseline": 0.0, "is_emccd": True}
    for f in real["frames"]:
        assert len(f["image"]) == f["width"] * f["height"] == 102 * 95 and len(f["reference"]) > 20
    tr = track_data()["tracks"]
    assert len(tr) == 371 and sum(len(t["frame"]) for t in tr) == 4963


def test_every_demo_function_exists():
    for d in DEMOS:
        assert (ROOT / d["plugins"][0] / "info.xml").exists()
        for f in d["functions"]:
            src = (ROOT / f["path"]).read_text()
            for name in f["names"]:
                cls, _, meth = name.partition(".")
                assert f"def {meth or cls}" in src or f"class {cls}" in src, name
    spt = load(ROOT / "spt_batch_analysis/__init__.py", ["FeatureCalculator", "AutocorrelationAnalyzer"])
    assert hasattr(spt["FeatureCalculator"], "summed_sines_cosines")
