"""Headless integration test for single-frame TIFF handling.

Runs detection + the main analysis pipeline on a single-frame TIFF and
verifies the output skips linking, produces one-point "tracks", and emits
the expected schema (with NaN for analyses that require >=2 frames).

Also runs the same pipeline on a multi-frame TIFF to confirm the existing
behaviour is unchanged.

Invoke with: python tests/test_single_frame_passthrough.py
"""
import os
import sys
import shutil
import tempfile
import importlib

import numpy as np
import pandas as pd
import skimage.io as skio


SINGLE_FRAME_TIF = "/Users/george/Desktop/test_singleFrame/1Frame.tif"
MULTI_FRAME_TIF = "/Users/george/Desktop/test_multiFrame/Endothelial_NonBapta_bin10_crop.tif"
NO_TRACKS_TIF = (
    "/Users/george/claude_test/data/IC305_Unlabeled_FOV30_561_1/"
    "IC305_Unlabeled_FOV30_561_1_MMStack_Default.ome.tif"
)
NO_TRACKS_LOCS = (
    "/Users/george/claude_test/data/IC305_Unlabeled_FOV30_561_1/"
    "IC305_Unlabeled_FOV30_561_1_MMStack_Default.ome_locs.csv"
)


def _patch_numpy_for_flika():
    """flika imports np.VisibleDeprecationWarning which was removed; shim it."""
    if not hasattr(np, "VisibleDeprecationWarning"):
        np.VisibleDeprecationWarning = DeprecationWarning


def _load_plugin_module():
    sys.path.insert(0, "/Users/george/claude_test")
    _patch_numpy_for_flika()
    mod = importlib.import_module("spt_batch_analysis")
    return mod


def _copy_tif_to_workdir(src_path, workdir):
    """Copy a TIFF into a temp dir so detection/analysis outputs don't pollute the source folder."""
    dst = os.path.join(workdir, os.path.basename(src_path))
    shutil.copy2(src_path, dst)
    return dst


def _make_headless_plugin(mod, workdir):
    """Instantiate the plugin parameters object + a minimal object that has
    the methods process_file / run_detection_for_file need.

    The SPTAnalyzerGUI class is Qt-heavy, so we build a lightweight stand-in
    that inherits from it but skips its __init__ and wires up just the
    attributes the pipeline touches.
    """
    # The main class is the one with process_file on it. Find it.
    target_cls = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and hasattr(obj, "process_file") and hasattr(obj, "run_detection_for_file"):
            target_cls = obj
            break
    if target_cls is None:
        raise RuntimeError("Could not locate the plugin class on the module")

    # Instantiate without calling __init__ (avoids Qt widget construction)
    inst = target_cls.__new__(target_cls)

    # Build a parameters object using the AnalysisParameters class from the module
    params_cls = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if isinstance(obj, type) and hasattr(obj, "to_dict") and hasattr(obj, "from_dict"):
            # AnalysisParameters defines to_dict / from_dict
            if name.lower().startswith("analysis") or "param" in name.lower():
                params_cls = obj
                break
    if params_cls is None:
        raise RuntimeError("Could not locate AnalysisParameters class")

    inst.parameters = params_cls()
    # Tweak a couple of defaults for the headless run so detection is quick
    inst.parameters.detection_method = "thunderstorm"
    inst.parameters.detection_skip_existing = False
    inst.parameters.detection_output_directory = workdir
    inst.parameters.enable_svm_classification = False
    inst.parameters.enable_enhanced_interpolation = False
    inst.parameters.enable_missing_points_integration = False
    inst.parameters.enable_autocorrelation_analysis = False
    inst.parameters.enable_background_subtraction = False
    # Multi-frame linking will use builtin
    inst.parameters.linking_method = "builtin"
    inst.parameters.min_track_segments = 1  # lenient for small test data
    inst.parameters.max_gap_frames = 3
    inst.parameters.max_link_distance = 3

    # Minimal logger stubs
    class _NoopLogger:
        def log(self, *a, **k):
            pass

        def log_error(self, *a, **k):
            import traceback as _tb
            print("  LOG_ERROR:", a, k)
            _tb.print_exc()

        def log_performance(self, *a, **k):
            pass

        def log_data_summary(self, *a, **k):
            pass

    inst.file_logger = _NoopLogger()
    inst.log_message = lambda msg: print(msg)
    inst.setup_logging_for_analysis = lambda fp: None
    inst.get_experiment_name_for_file = lambda fp: "test"
    inst.validate_linking_method = lambda: None
    inst.log_final_track_statistics = lambda df: None

    # Seed dict attributes so hasattr() short-circuits without hitting
    # Qt's C++ __getattr__ (which raises RuntimeError on __new__-only
    # instances).
    inst.export_column_checkboxes = {}
    inst.export_enhanced_checkbox = None

    # save_analysis_files probes Qt widget attrs. Replace with a minimal
    # writer so the rest of process_file is testable without a running
    # QApplication.
    def _stub_save(tracks_df, file_path, use_export_control, suffix=""):
        out = file_path.replace('.tif', '_enhanced_analysis.csv')
        tracks_df.to_csv(out, index=False)
        print(f"  [stub save] {os.path.basename(out)}  cols={len(tracks_df.columns)} rows={len(tracks_df)}")

    inst.save_analysis_files = _stub_save
    return inst


def _run_detection(inst, tif_path):
    ok = inst.run_detection_for_file(tif_path)
    if not ok:
        raise RuntimeError(f"Detection failed for {tif_path}")


def _run_analysis(inst, tif_path):
    ok = inst.process_file(tif_path)
    if not ok:
        raise RuntimeError(f"process_file returned False for {tif_path}")


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ✓ {msg}")


def test_single_frame():
    mod = _load_plugin_module()
    if not os.path.exists(SINGLE_FRAME_TIF):
        print(f"  ! single-frame test TIFF not found at {SINGLE_FRAME_TIF}; skipping")
        return
    with tempfile.TemporaryDirectory(prefix="spt_sf_") as workdir:
        tif = _copy_tif_to_workdir(SINGLE_FRAME_TIF, workdir)

        # Sanity: file really is single-frame
        A = skio.imread(tif, plugin="tifffile")
        _assert(A.ndim == 2 or (A.ndim == 3 and A.shape[0] == 1),
                f"input is single-frame (shape={A.shape})")

        inst = _make_headless_plugin(mod, workdir)
        _run_detection(inst, tif)

        locs_csv = os.path.join(workdir, "1Frame_locsID.csv")
        _assert(os.path.exists(locs_csv), "detection produced _locsID.csv")
        locs = pd.read_csv(locs_csv)
        _assert(len(locs) > 0, f"detected {len(locs)} localizations")
        _assert(set(locs["frame"].unique()) == {1}, "all detections are in frame 1")

        _run_analysis(inst, tif)

        enhanced_csv = os.path.join(workdir, "1Frame_enhanced_analysis.csv")
        _assert(os.path.exists(enhanced_csv), "enhanced_analysis.csv emitted")

        df = pd.read_csv(enhanced_csv)
        _assert(len(df) == len(locs),
                f"one output row per detection ({len(df)} == {len(locs)})")
        _assert(df["track_number"].isna().all(),
                "track_number is NaN for every unlinked detection")
        _assert(df["n_segments"].isna().all(),
                "n_segments is NaN for every unlinked detection")
        _assert((df["frame"] == 0).all(), "all rows in frame 0 (0-based)")

        # Core columns we expect to be present regardless of single-frame mode
        core_cols = {"track_number", "frame", "x", "y", "intensity", "n_segments"}
        missing = core_cols - set(df.columns)
        _assert(not missing, f"core columns present; missing: {missing}")

        # Schema-pad columns must exist (as NaN) when auto-disabled
        pad_cols = {"Elected_Label", "is_interpolated"}
        missing_pad = pad_cols - set(df.columns)
        _assert(not missing_pad, f"padded columns present; missing: {missing_pad}")

        # Intensity must be populated (not all zero)
        _assert(df["intensity"].notna().any() and (df["intensity"] > 0).any(),
                f"intensity non-trivial (mean={df['intensity'].mean():.2f})")

        print(f"\n[single-frame] schema: {len(df.columns)} columns, {len(df)} rows")


def test_multi_frame_regression():
    mod = _load_plugin_module()
    if not os.path.exists(MULTI_FRAME_TIF):
        print(f"  ! multi-frame test TIFF not found at {MULTI_FRAME_TIF}; skipping")
        return

    with tempfile.TemporaryDirectory(prefix="spt_mf_") as workdir:
        tif = _copy_tif_to_workdir(MULTI_FRAME_TIF, workdir)

        A = skio.imread(tif, plugin="tifffile")
        _assert(A.ndim == 3 and A.shape[0] > 1, f"input is multi-frame (shape={A.shape})")

        inst = _make_headless_plugin(mod, workdir)
        # Keep full multi-frame defaults (we already seeded sensible ones)
        inst.parameters.min_track_segments = 2  # exercise the filter
        _run_detection(inst, tif)

        base = os.path.splitext(os.path.basename(tif))[0]
        locs_csv = os.path.join(workdir, f"{base}_locsID.csv")
        _assert(os.path.exists(locs_csv), "detection produced _locsID.csv")
        locs = pd.read_csv(locs_csv)
        _assert(len(locs) > 0, f"detected {len(locs)} localizations")
        _assert(len(locs["frame"].unique()) > 1,
                f"detections span {len(locs['frame'].unique())} frames")

        _run_analysis(inst, tif)

        enhanced_csv = os.path.join(workdir, f"{base}_enhanced_analysis.csv")
        _assert(os.path.exists(enhanced_csv), "enhanced_analysis.csv emitted")
        df = pd.read_csv(enhanced_csv)
        track_counts = df.groupby("track_number").size()
        _assert((track_counts >= 2).any(),
                f"multi-point tracks formed (max track length = {track_counts.max()})")

        print(f"\n[multi-frame]  schema: {len(df.columns)} columns, {len(df)} rows, "
              f"{df['track_number'].nunique()} tracks, "
              f"max track length = {track_counts.max()}")


def test_no_tracks_fallback():
    """Multi-frame TIFF with min_track_segments high enough that no track passes
    the filter. With continue_without_tracks=True (the default), the pipeline
    should fall back to unlinked-particle passthrough instead of failing."""
    mod = _load_plugin_module()
    if not os.path.exists(NO_TRACKS_TIF):
        print(f"  ! no-tracks test TIFF not found at {NO_TRACKS_TIF}; skipping")
        return
    if not os.path.exists(NO_TRACKS_LOCS):
        print(f"  ! no-tracks locs file not found at {NO_TRACKS_LOCS}; skipping")
        return

    with tempfile.TemporaryDirectory(prefix="spt_nt_") as workdir:
        tif = _copy_tif_to_workdir(NO_TRACKS_TIF, workdir)
        # Co-locate the precomputed _locs.csv next to the TIFF so detection
        # can be skipped.
        shutil.copy2(NO_TRACKS_LOCS,
                     os.path.join(workdir, os.path.basename(NO_TRACKS_LOCS)))

        A = skio.imread(tif, plugin="tifffile")
        _assert(A.ndim == 3 and A.shape[0] > 1,
                f"input is multi-frame (shape={A.shape})")

        inst = _make_headless_plugin(mod, workdir)
        # Match the user's reported failing config: filter so high that
        # no real track survives.
        inst.parameters.min_track_segments = 10
        inst.parameters.continue_without_tracks = True
        inst.parameters.max_gap_frames = 36
        inst.parameters.max_link_distance = 3.0
        inst.parameters.linking_method = "builtin"

        n_locs = len(pd.read_csv(os.path.join(
            workdir, os.path.basename(NO_TRACKS_LOCS))))

        _run_analysis(inst, tif)

        base = os.path.splitext(os.path.basename(tif))[0]
        enhanced_csv = os.path.join(workdir, f"{base}_enhanced_analysis.csv")
        _assert(os.path.exists(enhanced_csv),
                "enhanced_analysis.csv emitted despite zero qualifying tracks")
        df = pd.read_csv(enhanced_csv)
        _assert(len(df) == n_locs,
                f"one output row per detection ({len(df)} == {n_locs})")
        _assert(df["track_number"].isna().all(),
                "track_number is NaN for every unlinked detection")
        _assert(df["n_segments"].isna().all(),
                "n_segments is NaN for every unlinked detection")
        _assert(df["intensity"].notna().any() and (df["intensity"] > 0).any(),
                f"intensity non-trivial (mean={df['intensity'].mean():.2f})")

        # Confirm we didn't lose information about the original frames.
        _assert(df["frame"].nunique() > 1,
                f"detections still span {df['frame'].nunique()} frames")

        print(f"\n[no-tracks] schema: {len(df.columns)} columns, {len(df)} rows")


def test_no_tracks_disabled():
    """When continue_without_tracks=False, the legacy failure path must still
    return False so users can opt out of the new behaviour."""
    mod = _load_plugin_module()
    if not os.path.exists(NO_TRACKS_TIF):
        print(f"  ! no-tracks test TIFF not found at {NO_TRACKS_TIF}; skipping")
        return

    with tempfile.TemporaryDirectory(prefix="spt_nt_off_") as workdir:
        tif = _copy_tif_to_workdir(NO_TRACKS_TIF, workdir)
        shutil.copy2(NO_TRACKS_LOCS,
                     os.path.join(workdir, os.path.basename(NO_TRACKS_LOCS)))

        inst = _make_headless_plugin(mod, workdir)
        inst.parameters.min_track_segments = 10
        inst.parameters.continue_without_tracks = False
        inst.parameters.linking_method = "builtin"

        ok = inst.process_file(tif)
        _assert(ok is False,
                "process_file returns False when continue_without_tracks=False")
        base = os.path.splitext(os.path.basename(tif))[0]
        _assert(not os.path.exists(
                    os.path.join(workdir, f"{base}_enhanced_analysis.csv")),
                "no enhanced_analysis.csv when fallback is disabled")


if __name__ == "__main__":
    print("=== Single-frame passthrough test ===")
    test_single_frame()
    print("\n=== Multi-frame regression test ===")
    test_multi_frame_regression()
    print("\n=== No-tracks fallback test (continue_without_tracks=True) ===")
    test_no_tracks_fallback()
    print("\n=== No-tracks opt-out test (continue_without_tracks=False) ===")
    test_no_tracks_disabled()
    print("\nAll tests passed.")
