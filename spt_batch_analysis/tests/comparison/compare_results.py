#!/usr/bin/env python3
"""
Compare FLIKA thunderSTORM results against ImageJ thunderSTORM results.

Reads paired CSV files produced by:
  - generate_comparison_macros.py  (FLIKA results)
  - ImageJ macro run_all_tests.ijm (ImageJ results)

For each test configuration it reports:
  - Number of localizations found by each implementation
  - Matched localizations (nearest-neighbour within tolerance)
  - Position, sigma, intensity, and uncertainty error statistics
  - Per-frame detection count comparison
  - Overall summary and scoring

Usage:
    python compare_results.py -d /path/to/comparison_tests
    python compare_results.py -d /path/to/comparison_tests --test wavelet_default
    python compare_results.py -d /path/to/comparison_tests --plot
"""

import sys
import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Optional plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# --------------------------------------------------------------------------
# Loading helpers
# --------------------------------------------------------------------------

def load_thunderstorm_csv(csv_path):
    """Load a thunderSTORM CSV and normalise column names.

    Handles both ImageJ thunderSTORM output and FLIKA output.
    Returns a DataFrame with standardised columns:
        frame, x_nm, y_nm, sigma_nm, intensity, offset, bkgstd, chi2, uncertainty_nm
    """
    df = pd.read_csv(csv_path)

    # Normalise column names (handle quoting and unit suffixes)
    col_map = {}
    for col in df.columns:
        c = col.strip().strip('"').lower()
        if c == 'frame':
            col_map[col] = 'frame'
        elif c in ('x [nm]', 'x_nm'):
            col_map[col] = 'x_nm'
        elif c in ('y [nm]', 'y_nm'):
            col_map[col] = 'y_nm'
        elif c in ('sigma [nm]', 'sigma_nm'):
            col_map[col] = 'sigma_nm'
        elif c in ('intensity [photon]', 'intensity_photon'):
            col_map[col] = 'intensity_photon'
        elif c in ('intensity [au]', 'intensity_au'):
            col_map[col] = 'intensity_au'
        elif c in ('offset [photon]', 'offset_photon'):
            col_map[col] = 'offset_photon'
        elif c in ('bkgstd [photon]', 'bkgstd_photon'):
            col_map[col] = 'bkgstd_photon'
        elif c == 'chi2':
            col_map[col] = 'chi2'
        elif c in ('uncertainty [nm]', 'uncertainty_nm'):
            col_map[col] = 'uncertainty_nm'
        elif c == 'id':
            col_map[col] = 'id'

    df = df.rename(columns=col_map)
    return df


# --------------------------------------------------------------------------
# Matching logic
# --------------------------------------------------------------------------

def match_localizations(df_ref, df_test, match_radius_nm=200.0):
    """Match localizations between reference and test DataFrames.

    Uses per-frame nearest-neighbour matching within match_radius_nm.

    Parameters
    ----------
    df_ref : DataFrame
        Reference (ImageJ) localizations
    df_test : DataFrame
        Test (FLIKA) localizations
    match_radius_nm : float
        Maximum distance (nm) for a valid match

    Returns
    -------
    matches : DataFrame
        Matched pairs with columns from both ref and test, plus distance_nm
    unmatched_ref : DataFrame
        Reference localizations with no match
    unmatched_test : DataFrame
        Test localizations with no match
    """
    from scipy.spatial import cKDTree

    matches = []
    used_test_indices = set()
    unmatched_ref_indices = []

    frames = sorted(set(df_ref['frame'].unique()) | set(df_test['frame'].unique()))

    for frame in frames:
        ref_frame = df_ref[df_ref['frame'] == frame]
        test_frame = df_test[df_test['frame'] == frame]

        if len(ref_frame) == 0 or len(test_frame) == 0:
            unmatched_ref_indices.extend(ref_frame.index.tolist())
            continue

        ref_xy = ref_frame[['x_nm', 'y_nm']].values
        test_xy = test_frame[['x_nm', 'y_nm']].values

        tree = cKDTree(test_xy)
        distances, indices = tree.query(ref_xy, k=1)

        # Track which test indices are already matched in this frame
        frame_used = set()
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            ref_idx = ref_frame.index[i]
            test_idx = test_frame.index[idx]

            if dist <= match_radius_nm and idx not in frame_used:
                frame_used.add(idx)
                used_test_indices.add(test_idx)
                matches.append({
                    'frame': frame,
                    'ref_x': ref_frame.loc[ref_idx, 'x_nm'],
                    'ref_y': ref_frame.loc[ref_idx, 'y_nm'],
                    'test_x': test_frame.loc[test_idx, 'x_nm'],
                    'test_y': test_frame.loc[test_idx, 'y_nm'],
                    'distance_nm': dist,
                    'ref_sigma': ref_frame.loc[ref_idx].get('sigma_nm', np.nan),
                    'test_sigma': test_frame.loc[test_idx].get('sigma_nm', np.nan),
                    'ref_intensity': ref_frame.loc[ref_idx].get('intensity_photon', np.nan),
                    'test_intensity': test_frame.loc[test_idx].get('intensity_photon', np.nan),
                    'ref_uncertainty': ref_frame.loc[ref_idx].get('uncertainty_nm', np.nan),
                    'test_uncertainty': test_frame.loc[test_idx].get('uncertainty_nm', np.nan),
                    'ref_chi2': ref_frame.loc[ref_idx].get('chi2', np.nan),
                    'test_chi2': test_frame.loc[test_idx].get('chi2', np.nan),
                    'ref_offset': ref_frame.loc[ref_idx].get('offset_photon', np.nan),
                    'test_offset': test_frame.loc[test_idx].get('offset_photon', np.nan),
                })
            else:
                unmatched_ref_indices.append(ref_idx)

    matches_df = pd.DataFrame(matches) if matches else pd.DataFrame()

    unmatched_ref = df_ref.loc[
        [i for i in df_ref.index if i in unmatched_ref_indices]
    ] if unmatched_ref_indices else pd.DataFrame()

    unmatched_test = df_test.loc[
        [i for i in df_test.index if i not in used_test_indices]
    ]

    return matches_df, unmatched_ref, unmatched_test


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def compute_comparison_stats(matches_df, df_ref, df_test):
    """Compute comparison statistics from matched localizations."""
    stats = {}
    n_ref = len(df_ref)
    n_test = len(df_test)
    n_matched = len(matches_df)

    stats['n_ref'] = n_ref
    stats['n_test'] = n_test
    stats['n_matched'] = n_matched
    stats['precision'] = n_matched / n_test if n_test > 0 else 0  # fraction of test that matched
    stats['recall'] = n_matched / n_ref if n_ref > 0 else 0       # fraction of ref that matched
    stats['f1'] = (2 * stats['precision'] * stats['recall'] /
                   (stats['precision'] + stats['recall'])) if (stats['precision'] + stats['recall']) > 0 else 0
    stats['jaccard'] = n_matched / (n_ref + n_test - n_matched) if (n_ref + n_test - n_matched) > 0 else 0

    if n_matched > 0:
        # Position error (Euclidean distance between matched pairs)
        stats['position_error_mean_nm'] = float(matches_df['distance_nm'].mean())
        stats['position_error_median_nm'] = float(matches_df['distance_nm'].median())
        stats['position_error_std_nm'] = float(matches_df['distance_nm'].std())
        stats['position_error_max_nm'] = float(matches_df['distance_nm'].max())
        stats['position_error_p90_nm'] = float(matches_df['distance_nm'].quantile(0.9))

        # X/Y bias (FLIKA - ImageJ), signed
        dx = matches_df['test_x'] - matches_df['ref_x']
        dy = matches_df['test_y'] - matches_df['ref_y']
        stats['x_bias_mean_nm'] = float(dx.mean())
        stats['y_bias_mean_nm'] = float(dy.mean())
        stats['x_bias_std_nm'] = float(dx.std())
        stats['y_bias_std_nm'] = float(dy.std())

        # Individual position precision (spread of positions for each method)
        # ImageJ position spread (std of x and y)
        stats['ref_x_std_nm'] = float(matches_df['ref_x'].std())
        stats['ref_y_std_nm'] = float(matches_df['ref_y'].std())
        stats['test_x_std_nm'] = float(matches_df['test_x'].std())
        stats['test_y_std_nm'] = float(matches_df['test_y'].std())

        # Sigma comparison
        if 'ref_sigma' in matches_df and 'test_sigma' in matches_df:
            sigma_err = matches_df['test_sigma'] - matches_df['ref_sigma']
            valid = sigma_err.dropna()
            if len(valid) > 0:
                stats['sigma_error_mean_nm'] = float(valid.mean())
                stats['sigma_error_std_nm'] = float(valid.std())
                rel = valid / matches_df.loc[valid.index, 'ref_sigma']
                stats['sigma_relative_error_mean'] = float(rel.mean())

        # Intensity comparison
        if 'ref_intensity' in matches_df and 'test_intensity' in matches_df:
            valid_mask = matches_df['ref_intensity'].notna() & matches_df['test_intensity'].notna()
            if valid_mask.sum() > 0:
                ref_i = matches_df.loc[valid_mask, 'ref_intensity']
                test_i = matches_df.loc[valid_mask, 'test_intensity']
                ratio = test_i / ref_i.replace(0, np.nan)
                stats['intensity_ratio_mean'] = float(ratio.mean())
                stats['intensity_ratio_median'] = float(ratio.median())
                stats['intensity_ratio_std'] = float(ratio.std())

        # Uncertainty comparison
        if 'ref_uncertainty' in matches_df and 'test_uncertainty' in matches_df:
            valid_mask = matches_df['ref_uncertainty'].notna() & matches_df['test_uncertainty'].notna()
            if valid_mask.sum() > 0:
                ref_u = matches_df.loc[valid_mask, 'ref_uncertainty']
                test_u = matches_df.loc[valid_mask, 'test_uncertainty']
                ratio = test_u / ref_u.replace(0, np.nan)
                stats['uncertainty_ratio_mean'] = float(ratio.mean())
                stats['uncertainty_ratio_std'] = float(ratio.std())

        # Chi2 comparison
        if 'ref_chi2' in matches_df and 'test_chi2' in matches_df:
            valid_mask = matches_df['ref_chi2'].notna() & matches_df['test_chi2'].notna()
            if valid_mask.sum() > 0:
                ref_c = matches_df.loc[valid_mask, 'ref_chi2']
                test_c = matches_df.loc[valid_mask, 'test_chi2']
                ratio = test_c / ref_c.replace(0, np.nan)
                stats['chi2_ratio_mean'] = float(ratio.mean())
                stats['chi2_ratio_std'] = float(ratio.std())

    # Per-frame counts
    ref_counts = df_ref.groupby('frame').size()
    test_counts = df_test.groupby('frame').size()
    all_frames = sorted(set(ref_counts.index) | set(test_counts.index))
    ref_per_frame = np.array([ref_counts.get(f, 0) for f in all_frames])
    test_per_frame = np.array([test_counts.get(f, 0) for f in all_frames])
    count_diff = test_per_frame - ref_per_frame
    stats['per_frame_count_diff_mean'] = float(count_diff.mean())
    stats['per_frame_count_diff_std'] = float(count_diff.std())
    if ref_per_frame.sum() > 0:
        stats['per_frame_correlation'] = float(np.corrcoef(ref_per_frame, test_per_frame)[0, 1]) if len(all_frames) > 1 else 1.0

    return stats


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_comparison(test_name, matches_df, df_ref, df_test, stats, output_dir):
    """Generate comparison plots for one test."""
    if not HAS_MATPLOTLIB or len(matches_df) == 0:
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f"Comparison: {test_name}\n"
                 f"Ref: {stats['n_ref']} locs | Test: {stats['n_test']} locs | "
                 f"Matched: {stats['n_matched']} | F1: {stats['f1']:.3f}",
                 fontsize=12)

    # 1. Position error histogram
    ax = axes[0, 0]
    ax.hist(matches_df['distance_nm'], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(stats['position_error_median_nm'], color='red', linestyle='--',
               label=f"median={stats['position_error_median_nm']:.1f} nm")
    ax.set_xlabel('Position error (nm)')
    ax.set_ylabel('Count')
    ax.set_title('Position Error Distribution')
    ax.legend()

    # 2. X position scatter
    ax = axes[0, 1]
    ax.scatter(matches_df['ref_x'], matches_df['test_x'], s=1, alpha=0.3)
    lims = [min(matches_df['ref_x'].min(), matches_df['test_x'].min()),
            max(matches_df['ref_x'].max(), matches_df['test_x'].max())]
    ax.plot(lims, lims, 'r-', linewidth=0.5)
    ax.set_xlabel('ImageJ x [nm]')
    ax.set_ylabel('FLIKA x [nm]')
    ax.set_title('X Position')
    ax.set_aspect('equal')

    # 3. Y position scatter
    ax = axes[0, 2]
    ax.scatter(matches_df['ref_y'], matches_df['test_y'], s=1, alpha=0.3)
    lims = [min(matches_df['ref_y'].min(), matches_df['test_y'].min()),
            max(matches_df['ref_y'].max(), matches_df['test_y'].max())]
    ax.plot(lims, lims, 'r-', linewidth=0.5)
    ax.set_xlabel('ImageJ y [nm]')
    ax.set_ylabel('FLIKA y [nm]')
    ax.set_title('Y Position')
    ax.set_aspect('equal')

    # 4. Sigma comparison
    ax = axes[1, 0]
    if 'ref_sigma' in matches_df and 'test_sigma' in matches_df:
        valid = matches_df.dropna(subset=['ref_sigma', 'test_sigma'])
        if len(valid) > 0:
            ax.scatter(valid['ref_sigma'], valid['test_sigma'], s=1, alpha=0.3)
            lims = [0, max(valid['ref_sigma'].max(), valid['test_sigma'].max()) * 1.1]
            ax.plot(lims, lims, 'r-', linewidth=0.5)
            ax.set_xlabel('ImageJ sigma [nm]')
            ax.set_ylabel('FLIKA sigma [nm]')
    ax.set_title('Sigma')

    # 5. Intensity comparison
    ax = axes[1, 1]
    if 'ref_intensity' in matches_df and 'test_intensity' in matches_df:
        valid = matches_df.dropna(subset=['ref_intensity', 'test_intensity'])
        if len(valid) > 0:
            ax.scatter(valid['ref_intensity'], valid['test_intensity'], s=1, alpha=0.3)
            lims = [0, max(valid['ref_intensity'].max(), valid['test_intensity'].max()) * 1.1]
            ax.plot(lims, lims, 'r-', linewidth=0.5)
            ax.set_xlabel('ImageJ intensity [photon]')
            ax.set_ylabel('FLIKA intensity [photon]')
    ax.set_title('Intensity')

    # 6. Per-frame detection count
    ax = axes[1, 2]
    ref_counts = df_ref.groupby('frame').size()
    test_counts = df_test.groupby('frame').size()
    all_frames = sorted(set(ref_counts.index) | set(test_counts.index))
    ax.plot(all_frames, [ref_counts.get(f, 0) for f in all_frames],
            'b-', alpha=0.7, label='ImageJ', linewidth=0.8)
    ax.plot(all_frames, [test_counts.get(f, 0) for f in all_frames],
            'r-', alpha=0.7, label='FLIKA', linewidth=0.8)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Detections per frame')
    ax.set_title('Detections per Frame')
    ax.legend()

    plt.tight_layout()
    plot_path = output_dir / f"comparison_{test_name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved: {plot_path}")


# --------------------------------------------------------------------------
# Timing / speed comparison
# --------------------------------------------------------------------------

def load_timing_data(output_dir):
    """Load timing data from both ImageJ and FLIKA runs.

    ImageJ timing: {test_name}_imagej_timing.txt  (format: "test_name,elapsed_ms")
    FLIKA timing:  flika_run_summary.json          (format: {"test_name": {"time_s": ...}})

    Returns
    -------
    timing : dict
        {test_name: {"imagej_s": float or None, "flika_s": float or None}}
    """
    output_dir = Path(output_dir)
    imagej_dir = output_dir / "imagej_results"
    timing = {}

    # Load ImageJ timing from individual txt files
    if imagej_dir.exists():
        for tf in imagej_dir.glob("*_timing.txt"):
            try:
                text = tf.read_text().strip()
                if ',' in text:
                    name, ms = text.split(',', 1)
                    name = name.strip()
                    timing.setdefault(name, {})['imagej_s'] = float(ms.strip()) / 1000.0
            except (ValueError, IndexError):
                pass

    # Load FLIKA timing from summary JSON
    flika_summary = output_dir / "flika_run_summary.json"
    if flika_summary.exists():
        with open(flika_summary) as f:
            flika_data = json.load(f)
        for name, info in flika_data.items():
            timing.setdefault(name, {})['flika_s'] = info.get('time_s', None)

    return timing


def print_speed_comparison(timing):
    """Print speed comparison table."""
    if not timing:
        return

    # Check if any ImageJ timing exists
    has_imagej = any('imagej_s' in v for v in timing.values())
    has_flika = any('flika_s' in v for v in timing.values())

    if not has_flika:
        return

    print("\n" + "="*90)
    print("SPEED COMPARISON")
    print("="*90)

    if has_imagej:
        header = f"{'Test':30s} {'ImageJ (s)':>12s} {'FLIKA (s)':>12s} {'Speedup':>10s}"
    else:
        header = f"{'Test':30s} {'FLIKA (s)':>12s}"
    print(header)
    print("-"*90)

    speedups = []
    for name in sorted(timing.keys()):
        t = timing[name]
        flika_s = t.get('flika_s')
        imagej_s = t.get('imagej_s')

        flika_str = f"{flika_s:.2f}" if flika_s is not None else "N/A"

        if has_imagej:
            imagej_str = f"{imagej_s:.2f}" if imagej_s is not None else "N/A"
            if imagej_s and flika_s and flika_s > 0:
                speedup = imagej_s / flika_s
                speedups.append(speedup)
                speedup_str = f"{speedup:.1f}x"
                if speedup > 1:
                    speedup_str += " faster"
                else:
                    speedup_str = f"{1/speedup:.1f}x slower"
            else:
                speedup_str = "N/A"
            print(f"{name:30s} {imagej_str:>12s} {flika_str:>12s} {speedup_str:>10s}")
        else:
            print(f"{name:30s} {flika_str:>12s}")

    print("-"*90)

    if speedups:
        mean_speedup = np.mean(speedups)
        print(f"\nAverage speedup: {mean_speedup:.1f}x "
              f"({'FLIKA faster' if mean_speedup > 1 else 'ImageJ faster'})")
    elif has_flika:
        flika_times = [t.get('flika_s', 0) for t in timing.values() if t.get('flika_s')]
        if flika_times:
            print(f"\nFLIKA total time: {sum(flika_times):.1f}s  "
                  f"mean per test: {np.mean(flika_times):.1f}s")
        if not has_imagej:
            print("(Run ImageJ macro to get comparison timing)")


def plot_speed_comparison(timing, output_dir):
    """Generate speed comparison bar chart."""
    if not HAS_MATPLOTLIB or not timing:
        return

    has_imagej = any('imagej_s' in v for v in timing.values())
    names = sorted(timing.keys())
    flika_times = [timing[n].get('flika_s', 0) or 0 for n in names]

    if has_imagej:
        imagej_times = [timing[n].get('imagej_s', 0) or 0 for n in names]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, imagej_times, width, label='ImageJ', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, flika_times, width, label='FLIKA', color='coral', alpha=0.8)
        ax.set_xlabel('Test Configuration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('Speed Comparison: ImageJ vs FLIKA thunderSTORM')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(names))
        ax.bar(x, flika_times, color='coral', alpha=0.8)
        ax.set_xlabel('Test Configuration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title('FLIKA thunderSTORM Processing Time')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')

    plt.tight_layout()
    plot_path = Path(output_dir) / "speed_comparison.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Speed comparison plot saved: {plot_path}")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def print_stats(test_name, stats):
    """Print formatted comparison statistics."""
    print(f"\n{'='*70}")
    print(f"  {test_name}")
    print(f"{'='*70}")
    print(f"  Localizations:  ImageJ={stats['n_ref']}  FLIKA={stats['n_test']}  "
          f"Matched={stats['n_matched']}")
    print(f"  Detection:      Precision={stats['precision']:.3f}  "
          f"Recall={stats['recall']:.3f}  F1={stats['f1']:.3f}  "
          f"Jaccard={stats['jaccard']:.3f}")

    if stats['n_matched'] > 0:
        print(f"  Position error: mean={stats['position_error_mean_nm']:.1f} nm  "
              f"median={stats['position_error_median_nm']:.1f} nm  "
              f"p90={stats['position_error_p90_nm']:.1f} nm  "
              f"max={stats['position_error_max_nm']:.1f} nm")
        if 'x_bias_mean_nm' in stats:
            print(f"  Position bias:  dx={stats['x_bias_mean_nm']:.2f}+/-{stats['x_bias_std_nm']:.2f} nm  "
                  f"dy={stats['y_bias_mean_nm']:.2f}+/-{stats['y_bias_std_nm']:.2f} nm")

        if 'sigma_error_mean_nm' in stats:
            print(f"  Sigma error:    mean={stats['sigma_error_mean_nm']:.2f} nm  "
                  f"std={stats['sigma_error_std_nm']:.2f} nm  "
                  f"relative={stats.get('sigma_relative_error_mean', 0):.3f}")

        if 'intensity_ratio_mean' in stats:
            print(f"  Intensity ratio: mean={stats['intensity_ratio_mean']:.3f}  "
                  f"median={stats['intensity_ratio_median']:.3f}  "
                  f"std={stats['intensity_ratio_std']:.3f}")

        if 'uncertainty_ratio_mean' in stats:
            print(f"  Uncertainty ratio: mean={stats['uncertainty_ratio_mean']:.3f}  "
                  f"std={stats['uncertainty_ratio_std']:.3f}")

        if 'chi2_ratio_mean' in stats:
            print(f"  Chi2 ratio:     mean={stats['chi2_ratio_mean']:.3f}  "
                  f"std={stats['chi2_ratio_std']:.3f}")

    print(f"  Per-frame count: diff_mean={stats['per_frame_count_diff_mean']:.1f}  "
          f"diff_std={stats['per_frame_count_diff_std']:.1f}  "
          f"corr={stats.get('per_frame_correlation', 0):.3f}")

    if 'imagej_time_s' in stats or 'flika_time_s' in stats:
        ij_t = stats.get('imagej_time_s')
        fl_t = stats.get('flika_time_s')
        parts = []
        if ij_t is not None:
            parts.append(f"ImageJ={ij_t:.2f}s")
        if fl_t is not None:
            parts.append(f"FLIKA={fl_t:.2f}s")
        if ij_t and fl_t and fl_t > 0:
            parts.append(f"ratio={ij_t/fl_t:.1f}x")
        print(f"  Timing:         {', '.join(parts)}")


def generate_summary_table(all_stats):
    """Generate a summary table of all tests."""
    print("\n" + "="*110)
    print("SUMMARY TABLE")
    print("="*110)
    header = (f"{'Test':30s} {'Ref':>6s} {'FLIKA':>6s} {'Match':>6s} {'F1':>6s} "
              f"{'PosErr':>8s} {'dX bias':>8s} {'dY bias':>8s} {'SigErr':>8s} {'IntRat':>7s}")
    print(header)
    print("-"*110)

    for name, stats in all_stats.items():
        pos_err = f"{stats.get('position_error_median_nm', 0):.1f}" if stats['n_matched'] > 0 else "N/A"
        dx_bias = f"{stats.get('x_bias_mean_nm', 0):.2f}" if 'x_bias_mean_nm' in stats else "N/A"
        dy_bias = f"{stats.get('y_bias_mean_nm', 0):.2f}" if 'y_bias_mean_nm' in stats else "N/A"
        sig_err = f"{stats.get('sigma_error_mean_nm', 0):.1f}" if 'sigma_error_mean_nm' in stats else "N/A"
        int_rat = f"{stats.get('intensity_ratio_mean', 0):.3f}" if 'intensity_ratio_mean' in stats else "N/A"

        row = (f"{name:30s} {stats['n_ref']:6d} {stats['n_test']:6d} "
               f"{stats['n_matched']:6d} {stats['f1']:6.3f} "
               f"{pos_err:>8s} {dx_bias:>8s} {dy_bias:>8s} {sig_err:>8s} {int_rat:>7s}")
        print(row)

    print("-"*110)

    # Overall scores
    f1_scores = [s['f1'] for s in all_stats.values()]
    pos_errors = [s.get('position_error_median_nm', np.nan)
                  for s in all_stats.values() if s['n_matched'] > 0]

    print(f"\nOverall F1:            mean={np.mean(f1_scores):.3f}  "
          f"min={np.min(f1_scores):.3f}  max={np.max(f1_scores):.3f}")
    if pos_errors:
        print(f"Overall position err:  mean={np.nanmean(pos_errors):.1f} nm  "
              f"max={np.nanmax(pos_errors):.1f} nm")


def plot_summary_chart(all_stats, timing, output_dir):
    """Generate a combined summary chart with F1 scores, detection counts, and timing."""
    if not HAS_MATPLOTLIB:
        return

    names = list(all_stats.keys())
    # Sort by F1 score descending
    names.sort(key=lambda n: all_stats[n]['f1'], reverse=True)

    f1_scores = [all_stats[n]['f1'] for n in names]
    n_ref = [all_stats[n]['n_ref'] for n in names]
    n_test = [all_stats[n]['n_test'] for n in names]
    pos_errors = [all_stats[n].get('position_error_median_nm', 0) for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('FLIKA vs ImageJ ThunderSTORM — Comparison Summary', fontsize=14, fontweight='bold')

    # --- Panel 1: F1 scores ---
    ax = axes[0, 0]
    colors = ['#2ecc71' if f >= 0.99 else '#27ae60' if f >= 0.98
              else '#f39c12' if f >= 0.95 else '#e74c3c' for f in f1_scores]
    bars = ax.barh(range(len(names)), f1_scores, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0.9, 1.005)
    ax.set_xlabel('F1 Score')
    ax.set_title('Detection F1 Score (higher is better)')
    ax.axvline(x=0.99, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
    ax.axvline(x=0.95, color='gray', linestyle=':', alpha=0.4, linewidth=0.8)
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(score - 0.002, i, f'{score:.3f}', va='center', ha='right',
                fontsize=8, fontweight='bold', color='white')
    ax.invert_yaxis()

    # --- Panel 2: Detection counts ---
    ax = axes[0, 1]
    x = np.arange(len(names))
    width = 0.35
    ax.barh(x - width/2, n_ref, width, label='ImageJ', color='steelblue', alpha=0.8)
    ax.barh(x + width/2, n_test, width, label='FLIKA', color='coral', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Number of Localizations')
    ax.set_title('Detection Count Comparison')
    ax.legend(loc='lower right', fontsize=9)
    ax.invert_yaxis()

    # --- Panel 3: Position bias (x/y) ---
    ax = axes[1, 0]
    dx_biases = [all_stats[n].get('x_bias_mean_nm', 0) for n in names]
    dy_biases = [all_stats[n].get('y_bias_mean_nm', 0) for n in names]
    x = np.arange(len(names))
    width = 0.35
    ax.barh(x - width/2, dx_biases, width, label='dX', color='steelblue', alpha=0.8)
    ax.barh(x + width/2, dy_biases, width, label='dY', color='coral', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Position Bias (nm)')
    ax.set_title('Position Bias FLIKA-ImageJ (0 = identical)')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend(loc='lower right', fontsize=9)
    for i, (dx, dy) in enumerate(zip(dx_biases, dy_biases)):
        max_bias = max(abs(dx), abs(dy))
        if max_bias > 0.5:
            ax.text(max(dx, dy) + 0.3, i, f'{max_bias:.1f}nm', va='center', fontsize=7)
    ax.invert_yaxis()

    # --- Panel 4: Speed comparison ---
    ax = axes[1, 1]
    if timing:
        t_names = [n for n in names if n in timing]
        imagej_t = [timing[n].get('imagej_s', 0) or 0 for n in t_names]
        flika_t = [timing[n].get('flika_s', 0) or 0 for n in t_names]
        x = np.arange(len(t_names))
        width = 0.35
        ax.barh(x - width/2, imagej_t, width, label='ImageJ', color='steelblue', alpha=0.8)
        ax.barh(x + width/2, flika_t, width, label='FLIKA', color='coral', alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(t_names, fontsize=9)
        ax.set_xlabel('Time (seconds)')
        ax.set_title('Processing Speed')
        ax.legend(loc='lower right', fontsize=9)
        # Log scale for readability (MLE is 84s vs 1.7s)
        ax.set_xscale('log')
        ax.invert_yaxis()
    else:
        ax.text(0.5, 0.5, 'No timing data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_axis_off()

    plt.tight_layout()
    plot_path = Path(output_dir) / "summary_chart.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Summary chart saved: {plot_path}")


# --------------------------------------------------------------------------
# Main comparison
# --------------------------------------------------------------------------

def compare_test(test_name, imagej_csv, flika_csv, output_dir,
                 match_radius_nm=200.0, do_plot=False):
    """Compare one test's results.

    Returns
    -------
    stats : dict or None
    """
    if not Path(imagej_csv).exists():
        print(f"  SKIP {test_name}: ImageJ results not found ({imagej_csv})")
        return None
    if not Path(flika_csv).exists():
        print(f"  SKIP {test_name}: FLIKA results not found ({flika_csv})")
        return None

    df_ref = load_thunderstorm_csv(imagej_csv)
    df_test = load_thunderstorm_csv(flika_csv)

    matches_df, unmatched_ref, unmatched_test = match_localizations(
        df_ref, df_test, match_radius_nm=match_radius_nm
    )

    stats = compute_comparison_stats(matches_df, df_ref, df_test)
    print_stats(test_name, stats)

    if do_plot:
        plot_comparison(test_name, matches_df, df_ref, df_test, stats, output_dir)

    # Save matched pairs for detailed analysis
    if len(matches_df) > 0:
        matches_path = output_dir / f"matches_{test_name}.csv"
        matches_df.to_csv(matches_path, index=False)

    return stats


def run_comparison(output_dir, test_names=None, match_radius_nm=200.0, do_plot=False):
    """Run comparison for all tests in the output directory.

    Parameters
    ----------
    output_dir : str or Path
    test_names : list of str, optional
    match_radius_nm : float
    do_plot : bool

    Returns
    -------
    all_stats : dict
    """
    output_dir = Path(output_dir)
    imagej_dir = output_dir / "imagej_results"
    flika_dir = output_dir / "flika_results"

    if not imagej_dir.exists():
        print(f"ImageJ results directory not found: {imagej_dir}")
        print("Please run the ImageJ macro first.")
        return {}

    if not flika_dir.exists():
        print(f"FLIKA results directory not found: {flika_dir}")
        print("Please run generate_comparison_macros.py first.")
        return {}

    # Load metadata
    meta_path = output_dir / "test_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        available_tests = list(meta.get("tests", {}).keys())
    else:
        # Infer from file names
        available_tests = [
            p.stem.replace("_imagej", "")
            for p in imagej_dir.glob("*_imagej.csv")
        ]

    if test_names:
        available_tests = [t for t in available_tests if t in test_names]

    if not available_tests:
        print("No tests found to compare.")
        return {}

    comparison_dir = output_dir / "comparison_results"
    comparison_dir.mkdir(exist_ok=True)

    all_stats = {}
    for test_name in available_tests:
        imagej_csv = imagej_dir / f"{test_name}_imagej.csv"
        # FLIKA save_localizations appends _locsID suffix
        flika_csv = flika_dir / f"{test_name}_flika_locsID.csv"
        if not flika_csv.exists():
            flika_csv = flika_dir / f"{test_name}_flika.csv"

        stats = compare_test(
            test_name, imagej_csv, flika_csv, comparison_dir,
            match_radius_nm=match_radius_nm, do_plot=do_plot
        )
        if stats is not None:
            all_stats[test_name] = stats

    if all_stats:
        generate_summary_table(all_stats)

        # Speed comparison
        timing = load_timing_data(output_dir)
        if timing:
            print_speed_comparison(timing)
            if do_plot:
                plot_speed_comparison(timing, comparison_dir)

        # Save full stats (include timing)
        for name in all_stats:
            if name in timing:
                all_stats[name]['imagej_time_s'] = timing[name].get('imagej_s')
                all_stats[name]['flika_time_s'] = timing[name].get('flika_s')

        stats_path = comparison_dir / "comparison_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"\nFull statistics saved to: {stats_path}")

        # Always generate summary chart
        plot_summary_chart(all_stats, timing, comparison_dir)

    return all_stats


# --------------------------------------------------------------------------
# Quick comparison against existing reference
# --------------------------------------------------------------------------

def compare_against_reference(flika_csv, reference_csv, output_dir=None,
                              match_radius_nm=200.0, do_plot=False):
    """Compare a single FLIKA result against a reference CSV.

    Useful for quick checks without running the full test suite.
    """
    if output_dir is None:
        output_dir = Path(flika_csv).parent
    output_dir = Path(output_dir)

    stats = compare_test(
        "reference_comparison",
        reference_csv, flika_csv, output_dir,
        match_radius_nm=match_radius_nm, do_plot=do_plot
    )
    return stats


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare FLIKA vs ImageJ thunderSTORM results"
    )
    parser.add_argument(
        "-d", "--dir",
        required=True,
        help="Comparison test output directory (from generate_comparison_macros.py)"
    )
    parser.add_argument(
        "--test", "-t",
        nargs="*",
        default=None,
        help="Specific test names to compare (default: all available)"
    )
    parser.add_argument(
        "--radius", "-r",
        type=float,
        default=200.0,
        help="Match radius in nm (default: 200)"
    )
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Generate comparison plots (requires matplotlib)"
    )
    parser.add_argument(
        "--reference",
        help="Compare a single FLIKA CSV against this reference CSV instead of batch mode"
    )
    parser.add_argument(
        "--flika-csv",
        help="FLIKA CSV path (used with --reference)"
    )

    args = parser.parse_args()

    if args.reference and args.flika_csv:
        compare_against_reference(
            args.flika_csv, args.reference,
            output_dir=args.dir,
            match_radius_nm=args.radius,
            do_plot=args.plot
        )
    else:
        run_comparison(
            args.dir,
            test_names=args.test,
            match_radius_nm=args.radius,
            do_plot=args.plot
        )


if __name__ == "__main__":
    main()
