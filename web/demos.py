"""The interactive demos: what each one ports, and the input data they run on (read from files in the repository).

The computations themselves run in the browser (static/*_engine.js); tests/fixtures/demo_cases.json, written by
scripts/make_demo_fixture.py from the plugins' own Python, is what the node tests compare the JS ports against.
"""
import csv
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPT = "spt_batch_analysis"
TS = f"{SPT}/thunderstorm_python"
REAL_TIF = f"{SPT}/test_data/real/Endothelial_NonBapta_bin10_crop.tif"
REAL_LOCS = f"{SPT}/test_data/real/Endothelial_NonBapta_bin10_crop_locs.csv"
REAL_PROTOCOL = f"{SPT}/test_data/real/Endothelial_NonBapta_bin10_crop_locs-protocol.txt"
SYN_DIR = f"{SPT}/test_data/synthetic"
TRACKS_CSV = "EXPERIMENTAL/advanced_particle_tracking/Endothelial_NonBapta_bin10_crop.tif_tracking_results.csv"

DEMOS = [
    {"id": "localization", "title": "Spot detection and localization",
     "blurb": "The ThunderSTORM pipeline of spt_batch_analysis on single frames: B-spline wavelet filter, "
              "std(Wave.F1) threshold, local maxima, integrated-Gaussian least-squares fits and the Thompson "
              "localization uncertainty. Real and synthetic frames from the repository's test data.",
     "engine": "localize_engine.js", "plugins": [SPT],
     "functions": [{"path": f"{TS}/filters.py", "names": ["WaveletFilter", "compute_threshold_expression"]},
                   {"path": f"{TS}/detection.py", "names": ["LocalMaximumDetector", "remove_border_detections"]},
                   {"path": f"{TS}/fitting.py", "names": ["fit_gaussian_lsq_batch_numba", "fit_gaussian_wlsq_batch_numba",
                                                          "compute_localization_precision",
                                                          "BaseFitter._photons_and_background_sd"]},
                   {"path": f"{TS}/pipeline.py", "names": ["ThunderSTORM.analyze_frame"]}]},
    {"id": "tracks", "title": "Track features and directional autocorrelation",
     "blurb": "The per-track features of spt_batch_analysis (radius of gyration, asymmetry, skewness, kurtosis, "
              "fractal dimension, net displacement, straightness, direction and persistence) and its directional "
              "autocorrelation (Gorelik & Gautreau 2014), on tracks you choose, draw or paste.",
     "engine": "track_engine.js", "plugins": [SPT],
     "functions": [{"path": f"{SPT}/__init__.py", "names": ["FeatureCalculator", "GeometricAnalyzer",
                                                            "DirectionAnalyzer", "AutocorrelationAnalyzer"]}]},
    {"id": "fft", "title": "FFT of a trace in chunks",
     "blurb": "fft_Chunker: a trace is cut into chunks of a set length, each chunk's FFT magnitude is plotted "
              "against frequency, and ranges of chunks are averaged (the baseline and puff groups).",
     "engine": "fft_engine.js", "plugins": ["fft_Chunker"],
     "functions": [{"path": "fft_Chunker/fft_Chunker.py", "names": ["fft_chunks", "FFT_Chunker.makeAverage"]}]},
]


def demo_list() -> list:
    return DEMOS


def _rows(rel: str) -> list[dict]:
    with open(ROOT / rel, newline="") as f:
        return list(csv.DictReader(f))


def _read_tif(rel: str):
    import tifffile
    return tifffile.imread(ROOT / rel)


def real_camera() -> dict:
    """Camera settings from the ImageJ ThunderSTORM protocol file (the JSON line after "Camera settings:")."""
    import json
    lines = (ROOT / REAL_PROTOCOL).read_bytes().decode().replace("\r", " ").split("\n")
    cam = json.loads(lines[lines.index("Camera settings:") + 1])
    gain = cam["gain"] if cam["isEmGain"] else 1.0
    return {"raw": cam, "photons_per_adu": cam["photons2ADU"] / gain, "baseline": float(cam["offset"]),
            "is_emccd": bool(cam["isEmGain"]), "pixel_size": float(cam["pixelSize"])}


REAL_FRAMES = (0, 49, 99)
SYN_CASES = (("sparse_108nm", 0), ("medium_108nm", 0), ("low_snr_108nm", 0), ("dense_108nm", 0))
SYN_CROP = (32, 96)   # rows/cols kept from the 128 x 128 synthetic frames


@lru_cache(maxsize=1)
def localization_data() -> dict:
    """Frames for the localization demo, each with reference positions in pixel-centre coordinates."""
    import json
    real = _read_tif(REAL_TIF)
    locs = _rows(REAL_LOCS)
    cam = real_camera()
    px = cam["pixel_size"]
    raw = cam["raw"]
    sources = [{
        "id": "real", "title": "Endothelial cell, TIRF (real data)", "pixel_size": px,
        "note": f"{REAL_TIF}: 100 frames, {real.shape[1]} x {real.shape[2]} px. Reference = ImageJ ThunderSTORM "
                f"localizations committed beside it ({REAL_LOCS}), shown at x_nm/{px:g} - 0.5 (ThunderSTORM's "
                "corner origin to pixel centres).",
        "protocol": [f"Camera (protocol file): offset {raw['offset']:g}, photons2ADU {raw['photons2ADU']:g}, "
                     f"EM gain {raw['gain']:g}, pixel {px:g} nm, so {cam['photons_per_adu']:g} photons per ADU.",
                     "ImageJ settings: wavelet filter (scale 2, order 3), local maximum 4-neighbourhood, threshold "
                     "std(Wave.F1), integrated Gaussian, weighted least squares, initial sigma 1.6, fit radius 3."],
        "camera": {k: cam[k] for k in ("photons_per_adu", "baseline", "is_emccd")},
        "defaults": {"connectivity": 4, "fitter": "wlsq", "initial_sigma": 1.6},
        "frames": [{"index": i, "width": real.shape[2], "height": real.shape[1], "image": real[i].ravel().tolist(),
                    "reference": [[float(r["x [nm]"]) / px - 0.5, float(r["y [nm]"]) / px - 0.5]
                                  for r in locs if int(float(r["frame"])) == i + 1]} for i in REAL_FRAMES]}]
    meta = json.loads((ROOT / SYN_DIR / "all_metadata.json").read_text())
    a, b = SYN_CROP
    frames = []
    for name, i in SYN_CASES:
        img = _read_tif(f"{SYN_DIR}/{name}.tif")[i, a:b, a:b].astype(float)
        m = meta[name]
        gt = [[float(r["x_nm"]) / m["pixel_size_nm"] - a, float(r["y_nm"]) / m["pixel_size_nm"] - a]
              for r in _rows(f"{SYN_DIR}/{name}_ground_truth.csv") if int(r["frame"]) == i + 1]
        gt = [p for p in gt if -0.5 <= p[0] < b - a - 0.5 and -0.5 <= p[1] < b - a - 0.5]
        frames.append({"index": i, "name": name, "description": m.get("description", ""), "width": b - a,
                       "height": b - a, "image": img.ravel().tolist(), "reference": gt})
    cam = meta["sparse_108nm"]["camera"]
    sources.append({
        "id": "synthetic", "title": "Synthetic frames with ground truth", "pixel_size": 108.0,
        "note": f"{SYN_DIR}/*.tif written by {SPT}/tests/synthetic/generate_synthetic_data.py (seed 42), centre "
                f"{b - a} x {b - a} px of frame 1. Reference = the ground-truth positions (pixel centres).",
        "protocol": [f"Camera: baseline {cam['baseline_adu']} ADU, {cam['photons_per_adu']} photons/ADU, EM gain "
                     f"{cam['em_gain']} (so {cam['photons_per_adu'] / cam['em_gain']:g} photons per ADU after gain)."],
        "camera": {"photons_per_adu": cam["photons_per_adu"] / cam["em_gain"], "baseline": float(cam["baseline_adu"]),
                   "is_emccd": bool(cam["is_emccd"])},
        "defaults": {"connectivity": 8, "fitter": "lsq", "initial_sigma": 1.3},
        "frames": frames})
    return {"sources": sources}


@lru_cache(maxsize=1)
def track_data() -> dict:
    """Tracks from the experimental advanced_particle_tracking plugin, run on the same real movie."""
    tracks = {}
    for r in _rows(TRACKS_CSV):
        t = tracks.setdefault(int(r["track_id"]), {"id": int(r["track_id"]), "movement_type": r["movement_type"],
                                                   "frame": [], "x": [], "y": []})
        t["frame"].append(int(r["frame"]))
        t["x"].append(float(r["x"]))
        t["y"].append(float(r["y"]))
    out = sorted(tracks.values(), key=lambda t: (-len(t["frame"]), t["id"]))
    return {"source": TRACKS_CSV, "movie": REAL_TIF, "tracks": out,
            "note": "Output of the EXPERIMENTAL advanced_particle_tracking plugin on the endothelial test movie, "
                    "committed in the repository. Positions are in pixels; many are whole pixels."}
