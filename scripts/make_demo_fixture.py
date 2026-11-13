"""Write tests/fixtures/demo_cases.json: what the plugins' own Python computes on the demo inputs.

The node tests (web/static/*_engine.test.mjs) check the JavaScript ports against these numbers, and
tests/test_demos.py checks that the fixture is still what the Python computes. Everything here calls the plugin code
itself: thunderstorm_python is imported as a package, and the spt_batch_analysis / fft_Chunker classes are loaded with
web/source_loader.py (their modules import flika and Qt).

Run: python scripts/make_demo_fixture.py
"""
import contextlib
import io
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from web.demos import TS, localization_data, track_data  # noqa: E402
from web.source_loader import load  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "demo_cases.json"
FRAMES = ROOT / "tests" / "fixtures" / "localization_frames.json"  # the input frames (web.demos)
SPT = ROOT / "spt_batch_analysis" / "__init__.py"


def clean(v):
    """JSON-safe: numpy to Python, NaN/inf to None."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (pd.Series, pd.Index)):
        v = v.to_numpy()
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not math.isfinite(float(v)) else float(v)
    return v


# --- localization -------------------------------------------------------------------------------------------------
LOC_CONFIGS = [
    {"connectivity": 8, "fitter": "lsq", "initial_sigma": 1.3, "threshold": 1.0, "fit_radius": 3},
    {"connectivity": 4, "fitter": "wlsq", "initial_sigma": 1.6, "threshold": 1.0, "fit_radius": 3},
    {"connectivity": 8, "fitter": "wlsq", "initial_sigma": 1.3, "threshold": 2.0, "fit_radius": 4},
]


def thunderstorm():
    sys.path.insert(0, str(ROOT / TS).rsplit("/", 1)[0])
    with contextlib.redirect_stdout(io.StringIO()):
        import thunderstorm_python as ts
    return ts


def localization_case(ts, image, camera, cfg, pixel_size):
    pipe = ts.ThunderSTORM(
        filter_type="wavelet", filter_params={"scale": 2, "order": 3}, detector_type="local_maximum",
        detector_params={"connectivity": f"{cfg['connectivity']}-neighbourhood"},
        fitter_type=f"gaussian_{cfg['fitter']}", fitter_params={"initial_sigma": cfg["initial_sigma"]},
        threshold_expression=f"{cfg['threshold']}*std(Wave.F1)", pixel_size=pixel_size,
        photons_per_adu=camera["photons_per_adu"], baseline=camera["baseline"], is_emccd=camera["is_emccd"])
    filtered = pipe.filter.apply(image)
    thr = ts.filters.compute_threshold_expression(image, filtered, pipe.threshold_expression,
                                                  wave_f1=pipe.filter.F1, filter_obj=pipe.filter)
    positions = pipe.detector.detect(filtered, thr)
    res = pipe.analyze_frame(image, fit_radius=cfg["fit_radius"])
    keys = ("x", "y", "intensity", "background", "sigma_x", "chi_squared", "uncertainty")
    return {"config": cfg, "threshold": float(thr), "positions": np.asarray(positions).tolist(),
            "result": {k: np.asarray(res.get(k, [])) for k in keys}}


def localization() -> dict:
    ts = thunderstorm()
    kernel = ts.filters.WaveletFilter._create_bspline_kernel(3, 2.0)
    cases = []
    data = localization_data()
    picks = [("real", 0), ("synthetic", 0), ("synthetic", 1), ("synthetic", 3)]
    for sid, fi in picks:
        src = next(s for s in data["sources"] if s["id"] == sid)
        fr = src["frames"][fi]
        img = np.array(fr["image"], dtype=float).reshape(fr["height"], fr["width"])
        for cfg in LOC_CONFIGS:
            case = localization_case(ts, img, src["camera"], cfg, src["pixel_size"])
            cases.append({"source": sid, "frame": fi, **case})
    # the whole filtered image for one small frame, to check the wavelet filter pixel by pixel
    src = next(s for s in data["sources"] if s["id"] == "synthetic")
    fr = src["frames"][0]
    img = np.array(fr["image"], dtype=float).reshape(fr["height"], fr["width"])
    w = ts.filters.WaveletFilter(scale=2, order=3)
    F = w.apply(img)
    precision = [{"args": a, "value": ts.fitting.compute_localization_precision(*a[:4], is_emccd=a[4])}
                 for a in [(1000.0, 3.0, 1.3, 1.0, False), (250.0, 10.0, 1.6, 1.0, True), (40.0, 0.5, 1.1, 1.0, True)]]
    return {"kernel": kernel, "wavelet": {"source": "synthetic", "frame": 0, "F": F.ravel(), "F1": w.F1.ravel()},
            "cases": cases, "precision": precision}


# --- tracks -------------------------------------------------------------------------------------------------------
def synthetic_tracks():
    """Seeded random walks (not data): free, confined, directed, plus a collinear and a repeated-point track."""
    rng = np.random.default_rng(7)
    out = []
    for kind in ("free", "confined", "directed"):
        n, x, y = 40, [0.0], [0.0]
        ang = rng.uniform(0, 2 * np.pi)
        for _ in range(n - 1):
            dx, dy = rng.normal(size=2)
            if kind == "directed":
                dx, dy = dx + 0.8 * np.cos(ang), dy + 0.8 * np.sin(ang)
            if kind == "confined":
                dx, dy = dx - 0.4 * x[-1], dy - 0.4 * y[-1]
            x.append(x[-1] + dx)
            y.append(y[-1] + dy)
        out.append({"name": kind, "frame": list(range(n)), "x": [v + 30 for v in x], "y": [v + 30 for v in y]})
    out.append({"name": "collinear", "frame": [0, 1, 2, 3, 4], "x": [1.0, 2, 3, 4, 5], "y": [2.0, 4, 6, 8, 10]})
    out.append({"name": "repeats", "frame": [0, 1, 2, 3, 4, 5, 6], "x": [5.0, 5, 6, 6, 8, 7, 7],
                "y": [5.0, 5, 5, 7, 7, 6, 6]})
    return out


def track_features(fc, ga, da, t):
    pts = np.column_stack([t["x"], t["y"]]).astype(float)
    df = pd.DataFrame({"x": pts[:, 0], "y": pts[:, 1]})
    rg, asym, skew, kurt = fc.radius_gyration_asymmetry(df)
    net, eff = fc.net_displacement_efficiency(pts)
    sin_m, sin_v, cos_m, cos_v = fc.summed_sines_cosines(pts.copy())
    rgs = ga.get_radius_of_gyration_simple(pts)
    step = ga.get_mean_step_length(pts)
    dirs = np.array([da.calculate_direction(*pts[i], *pts[i + 1]) for i in range(len(pts) - 1)])
    return {"rg": rg, "asymmetry": asym, "skewness": skew, "kurtosis": kurt, "fractal_dimension": fc.fractal_dimension(pts),
            "net_displacement": net, "efficiency": eff, "sin_mean": sin_m, "sin_vals": sin_v, "cos_mean": cos_m,
            "cos_vals": cos_v, "rg_simple": rgs, "mean_step": step, "scaled_rg": ga.get_scaled_rg(rgs, step),
            "linear": ga.classify_linear_motion_simple(pts), "directions": dirs,
            "persistence": da.calculate_directional_persistence(dirs),
            "autocorrelation": da.calculate_directional_autocorrelation(pts[:, 0], pts[:, 1])}


def tracks() -> dict:
    cls = load(SPT, ["FeatureCalculator", "GeometricAnalyzer", "DirectionAnalyzer", "AutocorrelationAnalyzer"])
    fc, ga, da, aa = (cls[k] for k in ("FeatureCalculator", "GeometricAnalyzer", "DirectionAnalyzer",
                                       "AutocorrelationAnalyzer"))
    real = track_data()["tracks"]
    inputs = synthetic_tracks() + [{"name": f"real track {t['id']}", **{k: t[k] for k in ("frame", "x", "y")}}
                                   for t in (real[0], real[5], real[40])]
    single = [{"input": t, **track_features(fc, ga, da, t)} for t in inputs]
    sets = []
    for name, chosen, dt, n in [("synthetic", synthetic_tracks()[:3], 1.0, 8), ("real 60", real[:60], 0.1, 12)]:
        df = pd.concat([pd.DataFrame({"track_number": i, "frame": t["frame"], "x": t["x"], "y": t["y"]})
                        for i, t in enumerate(chosen)], ignore_index=True)
        with contextlib.redirect_stdout(io.StringIO()):
            products, averages, individual = aa.process_track_data(df, dt, n)
        sets.append({"name": name, "time_interval": dt, "num_intervals": n,
                     "inputs": [{k: t[k] for k in ("frame", "x", "y")} for t in chosen],
                     "lags": [float(c) for c in averages.columns], "avg": averages.loc["AVG"].tolist(),
                     "sem": averages.loc["SEM"].tolist(), "count": [int(products[c].notna().sum()) if c in products
                                                                     else 0 for c in averages.columns],
                     "individual": individual.to_dict(orient="list")})
    return {"single": single, "autocorrelation": sets}


# --- fft ----------------------------------------------------------------------------------------------------------
def fake_signal():
    """The test signal of fft_Chunker/fakeData.py: 50 Hz + 0.5 x 80 Hz, 600 samples at 1/800 s."""
    N, T = 600, 1.0 / 800.0
    x = np.linspace(0.0, N * T, N)
    return np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x), T


def fft() -> dict:
    f = load(ROOT / "fft_Chunker/fft_Chunker.py", ["fft_chunks", "FFT_Chunker.makeAverage"])
    sig, T = fake_signal()
    noise = np.random.default_rng(3).normal(size=500).cumsum()
    cases = []
    for name, trace, dt, n in [("fakeData", sig, T, 100), ("fakeData", sig, T, 128), ("random walk", noise, 0.05, 64)]:
        df, *ranges = f["fft_chunks"](trace, n, dt)
        chunks = [{"power": df[f"power_{i}"].dropna(), "frequency": df[f"frequency_{i}"].dropna(),
                   "n": int(df[f"chunk_{i}"].notna().sum())} for i in range(1, len(df.columns) // 3 + 1)]
        cases.append({"name": name, "trace": trace, "time_step": dt, "chunk": n, "chunks": chunks, "ranges": ranges,
                      "rows": len(df), "power_raw": [df[f"power_{i}"].tolist() for i in range(1, len(chunks) + 1)]})
    df = f["fft_chunks"](noise, 64, 0.05)[0]
    box = lambda v: type("Box", (), {"value": staticmethod(lambda: v)})  # noqa: E731
    groups = {"baseline": (1, 2), "puff1": (3, 5), "puff2": (2, 2), "puff3": (0, 0), "puff4": (6, 8)}
    me = type("Me", (), {})()
    me.result_dict = {0: df}
    for g, (a, b) in groups.items():
        setattr(me, f"{g}_start", box(a))
        setattr(me, f"{g}_stop", box(b))
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f["FFT_Chunker.makeAverage"](me)
    return {"cases": cases, "average": {"chunk": 64, "time_step": 0.05, "groups": groups,
                                        "result": me.averageResults_dict["AverageResult_0"]}}


def build() -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return clean({"localization": localization(), "tracks": tracks(), "fft": fft()})


if __name__ == "__main__":
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    FRAMES.write_text(json.dumps(localization_data(), separators=(",", ":")))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} kB)")
