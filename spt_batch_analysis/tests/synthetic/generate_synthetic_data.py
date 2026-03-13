#!/usr/bin/env python3
"""
Generate synthetic SMLM data for algorithm validation.

Creates simulated microscopy image stacks with known ground truth positions,
varying imaging conditions (magnification, density, noise, PSF width).
Runs ALL algorithm configurations (filters, detectors, fitters) from the
comparison test suite on each synthetic dataset.

Usage:
    python generate_synthetic_data.py [--output /path/to/output_dir]
    python generate_synthetic_data.py --list-configs
    python generate_synthetic_data.py --configs sparse_100x dense_60x
    python generate_synthetic_data.py --algorithms wavelet_default lsq_fitting
    python generate_synthetic_data.py --compare
"""

import sys
import os
import json
import argparse
import time
import math
from pathlib import Path

import numpy as np
from scipy.special import erf

# Add project root to path
script_dir = Path(__file__).parent.absolute()
project_root = script_dir.parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import algorithm configurations from comparison tests
sys.path.insert(0, str(script_dir.parent / "comparison"))
from generate_comparison_macros import TEST_CONFIGS, CAMERA_DEFAULTS, build_run_analysis_string


# ===========================================================================
# Magnification / objective presets
# ===========================================================================
# Each preset defines realistic imaging parameters for common SMLM setups.
# pixel_size_nm is the effective pixel size at the sample plane.

MAGNIFICATION_PRESETS = {
    '60x': {
        'description': '60x oil objective (NA 1.4)',
        'magnification': 60,
        'pixel_size_nm': 267.0,   # 16um camera pixel / 60
        'na': 1.4,
        'emission_wavelength_nm': 680.0,
        'psf_sigma_nm': 360.0,    # ~1.35 px — realistic for high-NA 60x
    },
    '100x': {
        'description': '100x oil objective (NA 1.49)',
        'magnification': 100,
        'pixel_size_nm': 160.0,   # 16um camera pixel / 100
        'na': 1.49,
        'emission_wavelength_nm': 680.0,
        'psf_sigma_nm': 220.0,    # ~1.375 px — realistic for 100x TIRF
    },
    '100x_bin2': {
        'description': '100x oil with 2x2 binning',
        'magnification': 100,
        'pixel_size_nm': 320.0,   # 16um * 2 / 100
        'na': 1.49,
        'emission_wavelength_nm': 680.0,
        'psf_sigma_nm': 220.0,    # same PSF, larger pixels
    },
    '150x': {
        'description': '150x TIRF objective (NA 1.49)',
        'magnification': 150,
        'pixel_size_nm': 107.0,   # 16um camera pixel / 150
        'na': 1.49,
        'emission_wavelength_nm': 680.0,
        'psf_sigma_nm': 150.0,    # ~1.4 px
    },
    '108nm': {
        'description': 'Match real test data (108nm pixel)',
        'magnification': 148,
        'pixel_size_nm': 108.0,
        'na': 1.49,
        'emission_wavelength_nm': 680.0,
        'psf_sigma_nm': 173.0,    # 1.6 px — matches ThunderSTORM default
    },
}


# ===========================================================================
# Camera model presets
# ===========================================================================

CAMERA_PRESETS = {
    'emccd': {
        'description': 'EMCCD (Andor iXon)',
        'baseline_adu': 100,
        'readout_noise_e': 1.0,   # electrons (negligible with EM gain)
        'photons_per_adu': 3.6,
        'em_gain': 100,
        'is_emccd': True,
        'quantum_efficiency': 0.9,
        'bit_depth': 16,
    },
    'scmos': {
        'description': 'sCMOS (Hamamatsu Orca)',
        'baseline_adu': 100,
        'readout_noise_e': 1.5,
        'photons_per_adu': 0.47,
        'em_gain': 1,
        'is_emccd': False,
        'quantum_efficiency': 0.82,
        'bit_depth': 16,
    },
    'match_test': {
        'description': 'Match real comparison test camera',
        'baseline_adu': 100,
        'readout_noise_e': 1.0,
        'photons_per_adu': 3.6,
        'em_gain': 100,
        'is_emccd': True,
        'quantum_efficiency': 1.0,
        'bit_depth': 16,
    },
}


# ===========================================================================
# Test configurations
# ===========================================================================

SYNTHETIC_CONFIGS = {
    # --- Varying density ---
    'sparse_108nm': {
        'description': 'Sparse molecules, 108nm pixel (match real data)',
        'optics': '108nm',
        'camera': 'match_test',
        'image_size': (128, 128),
        'n_frames': 100,
        'n_molecules': 500,
        'photons_per_molecule': 1500,
        'background_per_pixel': 20,
        'density_description': '~5 molecules/frame',
    },
    'medium_108nm': {
        'description': 'Medium density, 108nm pixel',
        'optics': '108nm',
        'camera': 'match_test',
        'image_size': (128, 128),
        'n_frames': 100,
        'n_molecules': 2000,
        'photons_per_molecule': 1500,
        'background_per_pixel': 20,
        'density_description': '~20 molecules/frame',
    },
    'dense_108nm': {
        'description': 'Dense molecules, 108nm pixel',
        'optics': '108nm',
        'camera': 'match_test',
        'image_size': (128, 128),
        'n_frames': 100,
        'n_molecules': 5000,
        'photons_per_molecule': 1500,
        'background_per_pixel': 20,
        'density_description': '~50 molecules/frame',
    },
    # --- Varying SNR ---
    'low_snr_108nm': {
        'description': 'Low SNR (few photons, high background)',
        'optics': '108nm',
        'camera': 'match_test',
        'image_size': (128, 128),
        'n_frames': 100,
        'n_molecules': 1000,
        'photons_per_molecule': 500,
        'background_per_pixel': 50,
        'density_description': '~10 molecules/frame, low SNR',
    },
    'high_snr_108nm': {
        'description': 'High SNR (many photons, low background)',
        'optics': '108nm',
        'camera': 'match_test',
        'image_size': (128, 128),
        'n_frames': 100,
        'n_molecules': 1000,
        'photons_per_molecule': 5000,
        'background_per_pixel': 10,
        'density_description': '~10 molecules/frame, high SNR',
    },
    # --- Different magnifications ---
    'sparse_60x': {
        'description': 'Sparse molecules, 60x objective',
        'optics': '60x',
        'camera': 'emccd',
        'image_size': (128, 128),
        'n_frames': 100,
        'n_molecules': 500,
        'photons_per_molecule': 1500,
        'background_per_pixel': 15,
        'density_description': '~5 molecules/frame',
    },
    'sparse_100x': {
        'description': 'Sparse molecules, 100x objective',
        'optics': '100x',
        'camera': 'emccd',
        'image_size': (128, 128),
        'n_frames': 100,
        'n_molecules': 500,
        'photons_per_molecule': 1500,
        'background_per_pixel': 15,
        'density_description': '~5 molecules/frame',
    },
    'sparse_150x': {
        'description': 'Sparse molecules, 150x TIRF',
        'optics': '150x',
        'camera': 'emccd',
        'image_size': (128, 128),
        'n_frames': 100,
        'n_molecules': 500,
        'photons_per_molecule': 1500,
        'background_per_pixel': 15,
        'density_description': '~5 molecules/frame',
    },
    # --- sCMOS camera ---
    'medium_scmos_100x': {
        'description': 'Medium density, sCMOS camera, 100x',
        'optics': '100x',
        'camera': 'scmos',
        'image_size': (128, 128),
        'n_frames': 100,
        'n_molecules': 2000,
        'photons_per_molecule': 2000,
        'background_per_pixel': 20,
        'density_description': '~20 molecules/frame, sCMOS',
    },
}


# ===========================================================================
# Integrated Gaussian PSF rendering
# ===========================================================================

def render_integrated_gaussian(image, x_nm, y_nm, intensity_photons,
                                sigma_nm, pixel_size_nm):
    """Render an integrated Gaussian PSF onto an image.

    Uses erf-based pixel integration matching ThunderSTORM's PSF model.
    This gives proper pixel-averaged values rather than point-sampled.

    Parameters
    ----------
    image : ndarray
        Image to render onto (modified in-place)
    x_nm, y_nm : float
        Molecule position in nm
    intensity_photons : float
        Total intensity in photons
    sigma_nm : float
        PSF sigma in nm
    pixel_size_nm : float
        Pixel size in nm
    """
    sigma_px = sigma_nm / pixel_size_nm
    x_px = x_nm / pixel_size_nm
    y_px = y_nm / pixel_size_nm
    sqrt2_sig = math.sqrt(2.0) * sigma_px

    # Determine rendering window (4 sigma radius)
    radius = int(4 * sigma_px) + 1
    row_c = int(y_px)
    col_c = int(x_px)
    r0 = max(0, row_c - radius)
    r1 = min(image.shape[0], row_c + radius + 1)
    c0 = max(0, col_c - radius)
    c1 = min(image.shape[1], col_c + radius + 1)

    if r0 >= r1 or c0 >= c1:
        return

    rows = np.arange(r0, r1)
    cols = np.arange(c0, c1)

    # Integrated Gaussian: erf-based pixel integration
    ex = 0.5 * (erf((cols + 0.5 - x_px) / sqrt2_sig) -
                erf((cols - 0.5 - x_px) / sqrt2_sig))
    ey = 0.5 * (erf((rows + 0.5 - y_px) / sqrt2_sig) -
                erf((rows - 0.5 - y_px) / sqrt2_sig))

    psf = intensity_photons * np.outer(ey, ex)
    image[r0:r1, c0:c1] += psf


# ===========================================================================
# Synthetic data generation
# ===========================================================================

def generate_synthetic_dataset(config_name, config, output_dir, seed=42):
    """Generate a single synthetic dataset.

    Parameters
    ----------
    config_name : str
        Configuration name
    config : dict
        Configuration parameters
    output_dir : Path
        Output directory
    seed : int
        Random seed for reproducibility

    Returns
    -------
    info : dict
        Dataset information including paths and ground truth stats
    """
    import tifffile

    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    optics = MAGNIFICATION_PRESETS[config['optics']]
    camera = CAMERA_PRESETS[config['camera']]

    pixel_size = optics['pixel_size_nm']
    psf_sigma = optics['psf_sigma_nm']
    h, w = config['image_size']
    n_frames = config['n_frames']
    n_molecules = config['n_molecules']
    photons_mean = config['photons_per_molecule']
    bg_photons = config['background_per_pixel']

    print(f"  Optics: {optics['description']}, pixel={pixel_size:.0f}nm, sigma={psf_sigma:.0f}nm")
    print(f"  Camera: {camera['description']}")
    print(f"  Image: {w}x{h}, {n_frames} frames, {n_molecules} molecules")

    # Generate fixed molecule positions (uniform random in image area)
    border = int(4 * psf_sigma / pixel_size)  # avoid border molecules
    mol_x = rng.uniform(border * pixel_size, (w - border) * pixel_size, n_molecules)
    mol_y = rng.uniform(border * pixel_size, (h - border) * pixel_size, n_molecules)

    # Simulate blinking: each molecule has ~10% chance of being on per frame
    blinking = config.get('blinking', {})
    p_on = blinking.get('p_on', 0.05)
    p_off = blinking.get('p_off', 0.4)
    p_bleach = blinking.get('p_bleach', 0.005)
    states = np.zeros((n_frames, n_molecules), dtype=bool)
    active = np.zeros(n_molecules, dtype=bool)
    bleached = np.zeros(n_molecules, dtype=bool)

    for f in range(n_frames):
        for i in range(n_molecules):
            if bleached[i]:
                continue
            if active[i]:
                if rng.rand() < p_bleach:
                    bleached[i] = True
                    active[i] = False
                elif rng.rand() < p_off:
                    active[i] = False
            else:
                if rng.rand() < p_on:
                    active[i] = True
        states[f] = active.copy()

    # Generate movie and ground truth
    movie = np.zeros((n_frames, h, w), dtype=np.float64)
    ground_truth_rows = []

    for f in range(n_frames):
        # Expected photon rate per pixel (background only, constant)
        frame = np.full((h, w), float(bg_photons), dtype=np.float64)

        active_mask = states[f]
        for i in np.where(active_mask)[0]:
            # Sample photon count (Poisson around mean)
            n_photons = rng.poisson(photons_mean)
            if n_photons <= 0:
                continue

            # Render integrated Gaussian PSF onto the rate image
            render_integrated_gaussian(
                frame, mol_x[i], mol_y[i], n_photons, psf_sigma, pixel_size
            )

            ground_truth_rows.append({
                'frame': f + 1,  # 1-indexed
                'x_nm': float(mol_x[i]),
                'y_nm': float(mol_y[i]),
                'intensity_photons': int(n_photons),
                'sigma_nm': float(psf_sigma),
                'molecule_id': int(i),
            })

        # Single Poisson shot noise on the rate image (background + signal)
        frame = rng.poisson(np.clip(frame, 0, None)).astype(np.float64)

        # Camera model — matching ThunderSTORM DataGenerator exactly:
        #   1. Poisson noise (already applied above)
        #   2. QE: photons -> photoelectrons
        electrons = frame * camera['quantum_efficiency']
        #   3. EM gain: gamma(electrons, gain) — multiplicative stochastic amplification
        if camera['is_emccd'] and camera['em_gain'] > 1:
            electrons = rng.gamma(electrons + 1e-10, camera['em_gain'])
        #   4. Readout noise
        electrons += rng.normal(0, camera['readout_noise_e'], (h, w))
        #   5. Convert to ADU: divide by photons_per_adu, add baseline offset
        adu = electrons / camera['photons_per_adu'] + camera['baseline_adu']
        adu = np.clip(adu, 0, 2**camera['bit_depth'] - 1)

        movie[f] = adu

    # Convert to uint16
    movie_uint16 = movie.astype(np.uint16)

    # Save TIFF
    tiff_path = output_dir / f"{config_name}.tif"
    tifffile.imwrite(str(tiff_path), movie_uint16, imagej=True,
                     metadata={'axes': 'TYX', 'finterval': 1.0})

    # Save ground truth CSV
    import pandas as pd
    gt_df = pd.DataFrame(ground_truth_rows)
    gt_path = output_dir / f"{config_name}_ground_truth.csv"
    gt_df.to_csv(str(gt_path), index=False)

    # Save metadata
    meta = {
        'config_name': config_name,
        'description': config['description'],
        'optics': optics,
        'camera': {k: v for k, v in camera.items()},
        'pixel_size_nm': pixel_size,
        'psf_sigma_nm': psf_sigma,
        'image_size': list(config['image_size']),
        'n_frames': n_frames,
        'n_molecules': n_molecules,
        'photons_per_molecule': photons_mean,
        'background_per_pixel': bg_photons,
        'n_ground_truth_localizations': len(ground_truth_rows),
        'mean_active_per_frame': len(ground_truth_rows) / n_frames,
        'tiff_path': str(tiff_path),
        'ground_truth_path': str(gt_path),
        'seed': seed,
    }
    meta_path = output_dir / f"{config_name}_metadata.json"
    with open(str(meta_path), 'w') as fp:
        json.dump(meta, fp, indent=2)

    n_gt = len(ground_truth_rows)
    print(f"  Ground truth: {n_gt} localizations ({n_gt/n_frames:.1f}/frame)")
    print(f"  Saved: {tiff_path.name}")

    return meta


# ===========================================================================
# ImageJ macro generation for synthetic data
# ===========================================================================

def generate_imagej_macros_for_dataset(dataset_name, dataset_cfg, tiff_path,
                                        output_dir, algorithm_configs):
    """Generate ImageJ macros for all algorithm configs on one synthetic dataset.

    Returns list of (test_name, macro_text) tuples.
    """
    optics = MAGNIFICATION_PRESETS[dataset_cfg['optics']]
    camera = CAMERA_PRESETS[dataset_cfg['camera']]

    pixel_size = optics['pixel_size_nm']
    psf_sigma_px = optics['psf_sigma_nm'] / pixel_size

    # Camera setup macro (run once per dataset)
    camera_setup = (
        f'run("Camera setup", "offset={camera["baseline_adu"]:.1f} '
        f'quantumefficiency={camera["quantum_efficiency"]:.1f} '
        f'isemgain={"true" if camera["is_emccd"] else "false"} '
        f'photons2adu={camera["photons_per_adu"]:.1f} '
        f'gainem={camera["em_gain"]:.1f} '
        f'pixelsize={pixel_size:.1f}");\n'
    )

    macros = []
    for algo_name, algo_cfg in algorithm_configs.items():
        test_name = f"{dataset_name}__{algo_name}"
        results_csv = output_dir / f"{test_name}_imagej.csv"
        timing_file = output_dir / f"{test_name}_imagej_timing.txt"

        # Build the run analysis string, overriding sigma with the correct PSF
        algo_copy = json.loads(json.dumps(algo_cfg))
        algo_copy['imagej']['sigma'] = round(psf_sigma_px, 1)

        run_str = build_run_analysis_string(algo_copy)
        input_escaped = str(tiff_path).replace("\\", "/")
        output_escaped = str(results_csv).replace("\\", "/")
        timing_escaped = str(timing_file).replace("\\", "/")

        macro = f'''{camera_setup}
// === Synthetic test: {test_name} ===
// Dataset: {dataset_cfg['description']}  Algorithm: {algo_cfg['description']}
print("Running synthetic test: {test_name}");
if (isOpen("ThunderSTORM: results")) {{
    run("Show results table", "action=reset");
}}
wait(200);
run("Bio-Formats Importer", "open=[{input_escaped}] color_mode=Default rois_import=[ROI manager] split_channels view=Hyperstack stack_order=XYCZT");
t_start = getTime();
run("Run analysis", "{run_str}");
t_end = getTime();
elapsed_ms = t_end - t_start;
print("  Time: " + elapsed_ms + " ms");
run("Export results", "filepath=[{output_escaped}] fileformat=[CSV (comma separated)] sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true");
f = File.open("{timing_escaped}");
print(f, "{test_name}," + elapsed_ms);
File.close(f);
while (nImages>0) {{
    selectImage(nImages);
    close();
}}
print("Test {test_name} complete.");
'''
        macros.append((test_name, macro))

    return macros


# ===========================================================================
# FLIKA analysis of synthetic data
# ===========================================================================

def run_flika_on_synthetic(test_name, dataset_cfg, algo_flika_params,
                           tiff_path, output_dir, image_stack=None):
    """Run FLIKA ThunderSTORM on synthetic data with specific algorithm config.

    Parameters
    ----------
    test_name : str
        Combined name like "sparse_108nm__wavelet_default"
    dataset_cfg : dict
        Synthetic dataset config (optics, camera, etc.)
    algo_flika_params : dict
        Algorithm parameters from TEST_CONFIGS[algo_name]['flika']
    tiff_path : Path
        Path to the TIFF stack
    output_dir : Path
        Where to save results
    image_stack : ndarray, optional
        Pre-loaded image stack (avoids re-reading for each algorithm)
    """
    from thunderstorm_integration import ThunderSTORMDetector
    import tifffile

    optics = MAGNIFICATION_PRESETS[dataset_cfg['optics']]
    camera = CAMERA_PRESETS[dataset_cfg['camera']]

    pixel_size = optics['pixel_size_nm']
    psf_sigma_px = optics['psf_sigma_nm'] / pixel_size

    # Start with algorithm params, add camera/optics
    params = dict(algo_flika_params)
    params['initial_sigma'] = psf_sigma_px
    params['pixel_size'] = pixel_size
    params['photons_per_adu'] = camera['photons_per_adu']
    params['baseline'] = camera['baseline_adu']
    params['is_em_gain'] = camera['is_emccd']
    params['em_gain'] = camera['em_gain']
    params['quantum_efficiency'] = camera['quantum_efficiency']

    if image_stack is None:
        image_stack = tifffile.imread(str(tiff_path))
        if image_stack.ndim == 2:
            image_stack = image_stack[np.newaxis, ...]

    t0 = time.time()
    detector = ThunderSTORMDetector(parameters=params)
    localizations = detector.detect_and_fit(image_stack, show_progress=False)
    elapsed = time.time() - t0

    n_locs = len(localizations.get('x', []))
    csv_path = output_dir / f"{test_name}_flika.csv"
    actual_path = detector.save_localizations(localizations, csv_path,
                                               image_stack=image_stack)
    return {
        'n_locs': n_locs,
        'time_s': elapsed,
        'csv_path': str(actual_path),
    }


# ===========================================================================
# Ground truth comparison
# ===========================================================================

def compare_to_ground_truth(config_name, results_csv, gt_csv, pixel_size_nm,
                             match_radius_nm=200.0, label=''):
    """Compare detected localizations to ground truth."""
    import pandas as pd
    from scipy.spatial import cKDTree

    gt = pd.read_csv(gt_csv)
    try:
        det = pd.read_csv(results_csv)
    except Exception:
        return None

    # Normalize column names
    x_col = 'x [nm]' if 'x [nm]' in det.columns else 'x_nm'
    y_col = 'y [nm]' if 'y [nm]' in det.columns else 'y_nm'
    frame_col = 'frame'

    if x_col not in det.columns:
        print(f"  Warning: cannot find x column in {results_csv}")
        return None

    # Per-frame matching
    n_tp, n_fp, n_fn = 0, 0, 0
    all_errors = []

    for frame in sorted(gt['frame'].unique()):
        gt_f = gt[gt['frame'] == frame]
        det_f = det[det[frame_col] == frame]

        # Ground truth uses corner-at-origin; FLIKA/ImageJ use center-of-pixel-0
        # at 0.5*pixel_size. Shift ground truth to match detection convention.
        gt_xy = gt_f[['x_nm', 'y_nm']].values + 0.5 * pixel_size_nm
        if len(det_f) == 0:
            n_fn += len(gt_f)
            continue
        det_xy = det_f[[x_col, y_col]].values

        if len(gt_xy) == 0:
            n_fp += len(det_f)
            continue

        tree = cKDTree(det_xy)
        dists, indices = tree.query(gt_xy)
        used = set()
        for i, (d, idx) in enumerate(zip(dists, indices)):
            if d <= match_radius_nm and idx not in used:
                used.add(idx)
                n_tp += 1
                all_errors.append({
                    'dx': det_xy[idx, 0] - gt_xy[i, 0],
                    'dy': det_xy[idx, 1] - gt_xy[i, 1],
                    'distance': d,
                })
            else:
                n_fn += 1
        n_fp += len(det_f) - len(used)

    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    errors = np.array([(e['dx'], e['dy']) for e in all_errors]) if all_errors else np.zeros((0, 2))
    if len(errors) > 0:
        rmse_x = float(np.sqrt(np.mean(errors[:, 0]**2)))
        rmse_y = float(np.sqrt(np.mean(errors[:, 1]**2)))
        rmse = float(np.sqrt(np.mean(errors[:, 0]**2 + errors[:, 1]**2)))
        bias_x = float(np.mean(errors[:, 0]))
        bias_y = float(np.mean(errors[:, 1]))
        median_err = float(np.median(np.sqrt(errors[:, 0]**2 + errors[:, 1]**2)))
    else:
        rmse_x = rmse_y = rmse = bias_x = bias_y = median_err = float('nan')

    return {
        'config': config_name,
        'label': label,
        'n_ground_truth': n_tp + n_fn,
        'n_detected': n_tp + n_fp,
        'n_matched': n_tp,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'rmse_nm': rmse,
        'rmse_x_nm': rmse_x,
        'rmse_y_nm': rmse_y,
        'bias_x_nm': bias_x,
        'bias_y_nm': bias_y,
        'median_error_nm': median_err,
    }


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic SMLM data for testing"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(project_root / "test_data" / "synthetic"),
        help="Output directory for synthetic data"
    )
    parser.add_argument(
        "--results", "-r",
        default=str(script_dir / "results"),
        help="Results directory for FLIKA/ImageJ outputs"
    )
    parser.add_argument(
        "--configs", nargs='*',
        help="Subset of dataset configs to run (default: all)"
    )
    parser.add_argument(
        "--algorithms", nargs='*',
        help="Subset of algorithm configs to run (default: all from comparison tests)"
    )
    parser.add_argument(
        "--list-configs", action='store_true',
        help="List available configurations"
    )
    parser.add_argument(
        "--generate-only", action='store_true',
        help="Only generate synthetic data (no FLIKA analysis)"
    )
    parser.add_argument(
        "--flika-only", action='store_true',
        help="Only run FLIKA (data must exist)"
    )
    parser.add_argument(
        "--macro-only", action='store_true',
        help="Only generate ImageJ macros"
    )
    parser.add_argument(
        "--compare", action='store_true',
        help="Compare FLIKA/ImageJ results to ground truth"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Select algorithm configurations
    algo_configs = TEST_CONFIGS
    if args.algorithms:
        algo_configs = {k: v for k, v in TEST_CONFIGS.items() if k in args.algorithms}

    if args.list_configs:
        print("Available synthetic dataset configurations:")
        for name, cfg in SYNTHETIC_CONFIGS.items():
            optics = MAGNIFICATION_PRESETS[cfg['optics']]
            print(f"  {name:25s} - {cfg['description']} "
                  f"(px={optics['pixel_size_nm']:.0f}nm)")
        print(f"\nAvailable algorithm configurations ({len(TEST_CONFIGS)}):")
        for name, cfg in TEST_CONFIGS.items():
            print(f"  {name:25s} - {cfg['description']}")
        return

    output_dir = Path(args.output)
    results_dir = Path(args.results)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset_configs = SYNTHETIC_CONFIGS
    if args.configs:
        dataset_configs = {k: v for k, v in dataset_configs.items()
                          if k in args.configs}

    # --- Generate synthetic data ---
    if not args.flika_only and not args.macro_only and not args.compare:
        print("="*60)
        print("GENERATING SYNTHETIC DATA")
        print("="*60)
        all_meta = {}
        for name, cfg in dataset_configs.items():
            print(f"\n--- {name}: {cfg['description']} ---")
            meta = generate_synthetic_dataset(name, cfg, output_dir,
                                               seed=args.seed)
            all_meta[name] = meta
        with open(str(output_dir / "all_metadata.json"), 'w') as fp:
            json.dump(all_meta, fp, indent=2)
        print(f"\nSynthetic data saved to: {output_dir}")

    if args.generate_only:
        return

    n_algos = len(algo_configs)
    n_datasets = len(dataset_configs)
    total_tests = n_datasets * n_algos

    # --- Generate ImageJ macros ---
    if not args.flika_only and not args.compare:
        print("\n" + "="*60)
        print(f"GENERATING IMAGEJ MACROS ({n_datasets} datasets x {n_algos} algorithms = {total_tests} tests)")
        print("="*60)
        imagej_dir = results_dir / "imagej_results"
        imagej_dir.mkdir(parents=True, exist_ok=True)
        macros_dir = results_dir / "macros"
        macros_dir.mkdir(parents=True, exist_ok=True)

        all_macros = []
        for ds_name, ds_cfg in dataset_configs.items():
            tiff_path = output_dir / f"{ds_name}.tif"
            if not tiff_path.exists():
                print(f"  Skipping {ds_name}: {tiff_path} not found")
                continue
            macros = generate_imagej_macros_for_dataset(
                ds_name, ds_cfg, tiff_path, imagej_dir, algo_configs)
            for test_name, macro_text in macros:
                all_macros.append(macro_text)
            print(f"  Generated {len(macros)} macros for {ds_name}")

        # Combined macro
        combined = '\n\n'.join(all_macros)
        combined_path = macros_dir / "run_all_synthetic.ijm"
        with open(str(combined_path), 'w') as fp:
            fp.write(combined)
        print(f"\nCombined macro ({total_tests} tests): {combined_path}")

    # --- Run FLIKA analysis ---
    if not args.macro_only and not args.compare:
        print("\n" + "="*60)
        print(f"RUNNING FLIKA ANALYSIS ({n_datasets} datasets x {n_algos} algorithms = {total_tests} tests)")
        print("="*60)
        flika_dir = results_dir / "flika_results"
        flika_dir.mkdir(parents=True, exist_ok=True)
        flika_results = {}
        import tifffile

        test_count = 0
        for ds_name, ds_cfg in dataset_configs.items():
            tiff_path = output_dir / f"{ds_name}.tif"
            if not tiff_path.exists():
                print(f"  Skipping {ds_name}: data not found")
                continue

            # Load image once per dataset
            image_stack = tifffile.imread(str(tiff_path))
            if image_stack.ndim == 2:
                image_stack = image_stack[np.newaxis, ...]

            print(f"\n{'='*40}")
            print(f"Dataset: {ds_name} ({ds_cfg['description']})")
            print(f"{'='*40}")

            for algo_name, algo_cfg in algo_configs.items():
                test_name = f"{ds_name}__{algo_name}"
                test_count += 1
                try:
                    result = run_flika_on_synthetic(
                        test_name, ds_cfg, algo_cfg['flika'],
                        tiff_path, flika_dir, image_stack=image_stack)
                    print(f"  [{test_count}/{total_tests}] {algo_name:25s} "
                          f"{result['n_locs']:6d} locs  {result['time_s']:.1f}s")
                    flika_results[test_name] = result
                except Exception as e:
                    print(f"  [{test_count}/{total_tests}] {algo_name:25s} ERROR: {e}")

        with open(str(results_dir / "flika_synthetic_summary.json"), 'w') as fp:
            json.dump(flika_results, fp, indent=2)

    # --- Compare to ground truth ---
    if args.compare or (not args.generate_only and not args.macro_only):
        print("\n" + "="*60)
        print("COMPARING TO GROUND TRUTH")
        print("="*60)

        flika_dir = results_dir / "flika_results"
        imagej_dir = results_dir / "imagej_results"
        analysis_dir = results_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        all_results = []
        for ds_name, ds_cfg in dataset_configs.items():
            optics = MAGNIFICATION_PRESETS[ds_cfg['optics']]
            gt_csv = output_dir / f"{ds_name}_ground_truth.csv"
            if not gt_csv.exists():
                continue

            print(f"\n--- Dataset: {ds_name} ---")

            for algo_name in algo_configs:
                test_name = f"{ds_name}__{algo_name}"

                # Compare FLIKA results
                for suffix in ['_flika.csv', '_flika_locsID.csv']:
                    flika_csv = flika_dir / f"{test_name}{suffix}"
                    if flika_csv.exists():
                        break
                if flika_csv.exists():
                    stats = compare_to_ground_truth(
                        test_name, flika_csv, gt_csv,
                        optics['pixel_size_nm'], label='FLIKA')
                    if stats:
                        stats['dataset'] = ds_name
                        stats['algorithm'] = algo_name
                        all_results.append(stats)

                # Compare ImageJ results (if available)
                imagej_csv = imagej_dir / f"{test_name}_imagej.csv"
                if imagej_csv.exists():
                    stats = compare_to_ground_truth(
                        test_name, imagej_csv, gt_csv,
                        optics['pixel_size_nm'], label='ImageJ')
                    if stats:
                        stats['dataset'] = ds_name
                        stats['algorithm'] = algo_name
                        all_results.append(stats)

        # Print summary table grouped by dataset
        if all_results:
            print("\n" + "="*120)
            print("GROUND TRUTH COMPARISON SUMMARY")
            print("="*120)
            header = (f"{'Dataset':20s} {'Algorithm':25s} {'Method':6s} "
                      f"{'F1':>6s} {'Prec':>6s} {'Rec':>6s} "
                      f"{'RMSE':>8s} {'BiasX':>7s} {'BiasY':>7s} "
                      f"{'Det':>6s} {'GT':>6s}")
            print(header)
            print("-"*120)

            current_dataset = None
            for r in all_results:
                ds = r.get('dataset', r['config'])
                algo = r.get('algorithm', '')
                if ds != current_dataset:
                    if current_dataset is not None:
                        print()  # blank line between datasets
                    current_dataset = ds
                row = (f"{ds:20s} {algo:25s} {r['label']:6s} "
                       f"{r['f1']:6.3f} {r['precision']:6.3f} "
                       f"{r['recall']:6.3f} {r['rmse_nm']:8.1f} "
                       f"{r['bias_x_nm']:7.1f} {r['bias_y_nm']:7.1f} "
                       f"{r['n_detected']:6d} {r['n_ground_truth']:6d}")
                print(row)
            print("-"*120)

            # Save results
            with open(str(analysis_dir / "ground_truth_comparison.json"), 'w') as fp:
                json.dump(all_results, fp, indent=2)
            print(f"\nResults saved to: {analysis_dir}")


if __name__ == '__main__':
    main()
