#!/usr/bin/env python3
"""
Generate ImageJ macros and run FLIKA thunderSTORM for comparison testing.

This script:
1. Defines a set of test configurations covering different thunderSTORM options
2. Generates ImageJ macro (.ijm) files for each configuration
3. Runs the FLIKA thunderSTORM implementation with matching parameters
4. Saves FLIKA results alongside where ImageJ results will go

Usage:
    python generate_comparison_macros.py [--input /path/to/file.tif] [--output /path/to/output_dir]

After running this script:
    1. Open Fiji/ImageJ
    2. Drag-and-drop the generated 'run_all_tests.ijm' macro onto Fiji
    3. Wait for all tests to complete
    4. Run compare_results.py to analyze differences
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

# Add project root to path for imports
script_dir = Path(__file__).parent.absolute()        # tests/comparison/
project_root = script_dir.parent.parent.absolute()   # spt_batch_analysis/
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np

# --------------------------------------------------------------------------
# Test configuration definitions
# --------------------------------------------------------------------------
# Each config maps to both ImageJ macro parameters AND FLIKA ThunderSTORMDetector
# parameters.  The structure separates them clearly so the macro builder and
# the FLIKA runner each get exactly the right keys.

TEST_CONFIGS = {
    # ----- Filter variations -----
    "wavelet_default": {
        "description": "Wavelet B-Spline filter, default settings",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 2.0,
            "order": 3,
            "detector": "Local maximum",
            "connectivity": "4-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 2.0,
            "filter_order": 3,
            "detector_type": "local_maximum",
            "detector_connectivity": "4-neighbourhood",
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "gaussian_wlsq",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },
    "wavelet_scale4_order5": {
        "description": "Wavelet filter with scale=4, order=5",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 4.0,
            "order": 5,
            "detector": "Local maximum",
            "connectivity": "8-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 4.0,
            "filter_order": 5,
            "detector_type": "local_maximum",
            "detector_connectivity": "8-neighbourhood",
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "gaussian_wlsq",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },
    "dog_filter": {
        "description": "Difference of Gaussians filter",
        "imagej": {
            "filter": "Difference-of-Gaussians filter",
            "sigma1": 1.0,
            "sigma2": 1.6,
            "detector": "Local maximum",
            "connectivity": "8-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "dog",
            "filter_sigma1": 1.0,
            "filter_sigma2": 1.6,
            "detector_type": "local_maximum",
            "detector_connectivity": "8-neighbourhood",
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "gaussian_wlsq",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },
    "gaussian_filter": {
        "description": "Gaussian (lowered) filter",
        "imagej": {
            "filter": "Lowered Gaussian filter",
            "scale": 2.0,
            "order": 3,
            "detector": "Local maximum",
            "connectivity": "8-neighbourhood",
            "threshold": "1.5*std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "lowered_gaussian",
            "filter_sigma": 1.6,
            "detector_type": "local_maximum",
            "detector_connectivity": "8-neighbourhood",
            "detector_threshold": "1.5*std(Wave.F1)",
            "fitter_type": "gaussian_wlsq",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },

    # ----- Detector variations -----
    "nms_detector": {
        "description": "Non-maximum suppression detector",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 2.0,
            "order": 3,
            "detector": "Non-maximum suppression",
            "connectivity": "8-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 2.0,
            "filter_order": 3,
            "detector_type": "non_maximum_suppression",
            "detector_connectivity": "8-neighbourhood",
            "detector_radius": 1,
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "gaussian_wlsq",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },
    "centroid_detector": {
        "description": "Centroid of connected components detector",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 2.0,
            "order": 3,
            "detector": "Centroid of connected components",
            "connectivity": "8-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 2.0,
            "filter_order": 3,
            "detector_type": "centroid",
            "detector_connectivity": "8-neighbourhood",
            "use_watershed": False,
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "gaussian_wlsq",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },

    # ----- Fitting method variations -----
    "lsq_fitting": {
        "description": "Least squares fitting",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 2.0,
            "order": 3,
            "detector": "Local maximum",
            "connectivity": "8-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 2.0,
            "filter_order": 3,
            "detector_type": "local_maximum",
            "detector_connectivity": "8-neighbourhood",
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "gaussian_lsq",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },
    "mle_fitting": {
        "description": "Maximum likelihood estimation fitting",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 2.0,
            "order": 3,
            "detector": "Local maximum",
            "connectivity": "8-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Maximum likelihood",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 2.0,
            "filter_order": 3,
            "detector_type": "local_maximum",
            "detector_connectivity": "8-neighbourhood",
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "gaussian_mle",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },

    # ----- PSF model variations -----
    "psf_gaussian": {
        "description": "PSF: Gaussian (non-integrated) with WLSQ",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 2.0,
            "order": 3,
            "detector": "Local maximum",
            "connectivity": "8-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "PSF: Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 2.0,
            "filter_order": 3,
            "detector_type": "local_maximum",
            "detector_connectivity": "8-neighbourhood",
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "gaussian_wlsq",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },
    "radial_symmetry": {
        "description": "Radial symmetry estimator",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 2.0,
            "order": 3,
            "detector": "Local maximum",
            "connectivity": "8-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "Radial symmetry",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 2.0,
            "filter_order": 3,
            "detector_type": "local_maximum",
            "detector_connectivity": "8-neighbourhood",
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "radial_symmetry",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },

    # ----- Multi-emitter fitting -----
    "mfa_enabled": {
        "description": "Multi-emitter fitting analysis enabled",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 2.0,
            "order": 3,
            "detector": "Local maximum",
            "connectivity": "8-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": True,
            "keep_same_intensity": True,
            "nmax": 5,
            "fixed_intensity": False,
            "expected_intensity": "500:2500",
            "pvalue": "1.0E-6",
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 2.0,
            "filter_order": 3,
            "detector_type": "local_maximum",
            "detector_connectivity": "8-neighbourhood",
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "gaussian_wlsq",
            "fit_radius": 3,
            "initial_sigma": 1.6,
            "multi_emitter_enabled": True,
            "multi_emitter_max": 5,
            "multi_emitter_pvalue": 1e-6,
            "multi_emitter_keep_same_intensity": True,
            "multi_emitter_fixed_intensity": False,
            "multi_emitter_intensity_min": 500,
            "multi_emitter_intensity_max": 2500,
        },
    },

    # ----- Threshold variations -----
    "high_threshold": {
        "description": "Higher detection threshold (2x std)",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 2.0,
            "order": 3,
            "detector": "Local maximum",
            "connectivity": "8-neighbourhood",
            "threshold": "2*std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 3,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 2.0,
            "filter_order": 3,
            "detector_type": "local_maximum",
            "detector_connectivity": "8-neighbourhood",
            "detector_threshold": "2*std(Wave.F1)",
            "fitter_type": "gaussian_wlsq",
            "fit_radius": 3,
            "initial_sigma": 1.6,
        },
    },
    "fitradius_5": {
        "description": "Larger fit radius (5 pixels)",
        "imagej": {
            "filter": "Wavelet filter (B-Spline)",
            "scale": 2.0,
            "order": 3,
            "detector": "Local maximum",
            "connectivity": "8-neighbourhood",
            "threshold": "std(Wave.F1)",
            "estimator": "PSF: Integrated Gaussian",
            "sigma": 1.6,
            "fitradius": 5,
            "method": "Weighted Least squares",
            "full_image_fitting": False,
            "mfaenabled": False,
        },
        "flika": {
            "filter_type": "wavelet",
            "filter_scale": 2.0,
            "filter_order": 3,
            "detector_type": "local_maximum",
            "detector_connectivity": "8-neighbourhood",
            "detector_threshold": "std(Wave.F1)",
            "fitter_type": "gaussian_wlsq",
            "fit_radius": 5,
            "initial_sigma": 1.6,
        },
    },
}

# --------------------------------------------------------------------------
# Camera parameters (shared across all tests)
# --------------------------------------------------------------------------
CAMERA_DEFAULTS = {
    "pixel_size": 108.0,
    "photons_per_adu": 3.6,
    "baseline": 100.0,
    "is_em_gain": True,
    "em_gain": 100.0,
    "quantum_efficiency": 1.0,
}


# --------------------------------------------------------------------------
# ImageJ macro generation
# --------------------------------------------------------------------------

def build_run_analysis_string(cfg):
    """Build the thunderSTORM 'Run analysis' parameter string for ImageJ."""
    p = cfg["imagej"]
    parts = []

    parts.append(f'filter=[{p["filter"]}]')
    if "Wavelet" in p["filter"]:
        parts.append(f'scale={p["scale"]}')
        parts.append(f'order={p["order"]}')
    elif p["filter"] in ("Gaussian filter", "Lowered Gaussian filter"):
        parts.append(f'sigma={p["sigma"]}')
    elif "Difference-of-Gaussians" in p["filter"] or p["filter"] == "Difference of Gaussians":
        parts.append(f'sigma1={p.get("sigma1", p["sigma"] * 0.625)}')
        parts.append(f'sigma2={p.get("sigma2", p["sigma"])}')
    elif p["filter"] in ("Median filter", "Averaging filter (Box)", "Difference of averaging filters"):
        parts.append(f'size={p.get("size", 5)}')

    parts.append(f'detector=[{p["detector"]}]')
    parts.append(f'connectivity={p["connectivity"]}')
    parts.append(f'threshold={p["threshold"]}')

    parts.append(f'estimator=[{p["estimator"]}]')
    parts.append(f'sigma={p["sigma"]}')
    parts.append(f'fitradius={p["fitradius"]}')
    parts.append(f'method=[{p["method"]}]')
    parts.append(f'full_image_fitting={"true" if p.get("full_image_fitting") else "false"}')

    mfa = p.get("mfaenabled", False)
    parts.append(f'mfaenabled={"true" if mfa else "false"}')
    if mfa:
        parts.append(f'keep_same_intensity={"true" if p.get("keep_same_intensity", True) else "false"}')
        parts.append(f'nmax={p.get("nmax", 5)}')
        parts.append(f'fixed_intensity={"true" if p.get("fixed_intensity", False) else "false"}')
        parts.append(f'expected_intensity={p.get("expected_intensity", "500:2500")}')
        parts.append(f'pvalue={p.get("pvalue", "1.0E-6")}')

    parts.append('renderer=[No Renderer]')
    parts.append('magnification=5.0')
    parts.append('colorizez=false')
    parts.append('threed=false')
    parts.append('shifts=2')
    parts.append('repaint=50')

    return " ".join(parts)


EXPORT_COLUMNS = "sigma=true intensity=true chi2=true offset=true saveprotocol=true x=true y=true bkgstd=true uncertainty=true frame=true id=true"


def generate_single_test_macro(test_name, cfg, input_path, output_csv_path):
    """Generate an ImageJ macro block for one test configuration."""
    run_str = build_run_analysis_string(cfg)
    # Escape backslashes for ImageJ macro string (Windows paths)
    input_escaped = str(input_path).replace("\\", "/")
    output_escaped = str(output_csv_path).replace("\\", "/")

    # timing_csv_path sits next to the result CSV
    timing_path = str(output_csv_path).replace('.csv', '_timing.txt').replace("\\", "/")

    macro = f'''// === Test: {test_name} ===
// {cfg["description"]}
print("Running test: {test_name}");
// Clear previous thunderSTORM results to prevent data carryover
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
run("Export results", "filepath=[{output_escaped}] fileformat=[CSV (comma separated)] {EXPORT_COLUMNS}");
// Save timing to file
f = File.open("{timing_path}");
print(f, "{test_name}," + elapsed_ms);
File.close(f);
while (nImages>0) {{
    selectImage(nImages);
    close();
}}
print("Test {test_name} complete.");
'''
    return macro


def generate_all_macros(input_path, output_dir, test_names=None):
    """Generate individual and combined ImageJ macros for all tests.

    Parameters
    ----------
    input_path : str or Path
        Path to input .tif file
    output_dir : str or Path
        Directory to write macro files and results
    test_names : list of str, optional
        Subset of test names to generate. If None, generate all.

    Returns
    -------
    macro_path : Path
        Path to the combined 'run_all_tests.ijm' macro
    """
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    imagej_dir = output_dir / "imagej_results"
    imagej_dir.mkdir(exist_ok=True)

    configs = TEST_CONFIGS
    if test_names:
        configs = {k: v for k, v in configs.items() if k in test_names}

    combined_macro = '// Auto-generated thunderSTORM comparison macros\n'
    combined_macro += f'// Input: {input_path}\n'
    combined_macro += f'// Generated: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n'

    # Set thunderSTORM camera parameters
    # Note: thunderSTORM uses these from its Camera Setup dialog;
    # setting them via macro requires the "Camera setup" command
    cam = CAMERA_DEFAULTS
    combined_macro += f'// Set camera parameters\n'
    combined_macro += (
        f'run("Camera setup", '
        f'"offset={cam["baseline"]} '
        f'quantumefficiency={cam["quantum_efficiency"]} '
        f'isemgain={"true" if cam["is_em_gain"] else "false"} '
        f'photons2adu={cam["photons_per_adu"]} '
        f'gainem={cam["em_gain"]} '
        f'pixelsize={cam["pixel_size"]}");\n\n'
    )

    for test_name, cfg in configs.items():
        csv_path = imagej_dir / f"{test_name}_imagej.csv"
        block = generate_single_test_macro(test_name, cfg, input_path, csv_path)
        combined_macro += block + "\n"

        # Also save individual macro
        individual_path = output_dir / f"macro_{test_name}.ijm"
        # Individual macros need camera setup too
        individual_macro = (
            f'run("Camera setup", '
            f'"offset={cam["baseline"]} '
            f'quantumefficiency={cam["quantum_efficiency"]} '
            f'isemgain={"true" if cam["is_em_gain"] else "false"} '
            f'photons2adu={cam["photons_per_adu"]} '
            f'gainem={cam["em_gain"]} '
            f'pixelsize={cam["pixel_size"]}");\n\n'
        )
        individual_macro += block
        with open(individual_path, 'w') as f:
            f.write(individual_macro)

    combined_macro += '\nprint("All tests complete!");\n'

    macro_path = output_dir / "run_all_tests.ijm"
    with open(macro_path, 'w') as f:
        f.write(combined_macro)

    print(f"Generated {len(configs)} test macros")
    print(f"Combined macro: {macro_path}")
    print(f"ImageJ results will be saved to: {imagej_dir}")

    return macro_path


# --------------------------------------------------------------------------
# FLIKA thunderSTORM execution
# --------------------------------------------------------------------------

def run_flika_tests(input_path, output_dir, test_names=None):
    """Run FLIKA thunderSTORM on all test configurations.

    Parameters
    ----------
    input_path : str or Path
        Path to input .tif file
    output_dir : str or Path
        Directory to write results
    test_names : list of str, optional
        Subset of test names to run. If None, run all.

    Returns
    -------
    results : dict
        {test_name: {"n_locs": int, "time_s": float, "csv_path": str}}
    """
    from thunderstorm_integration import ThunderSTORMDetector
    import tifffile

    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir)
    flika_dir = output_dir / "flika_results"
    flika_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    print(f"Loading image: {input_path}")
    image_stack = tifffile.imread(str(input_path))
    if image_stack.ndim == 2:
        image_stack = image_stack[np.newaxis, ...]
    print(f"  Image shape: {image_stack.shape}")

    configs = TEST_CONFIGS
    if test_names:
        configs = {k: v for k, v in configs.items() if k in test_names}

    cam = CAMERA_DEFAULTS
    results = {}

    for test_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"Running FLIKA test: {test_name}")
        print(f"  {cfg['description']}")
        print(f"{'='*60}")

        # Build parameter dict
        params = dict(cfg["flika"])
        params.update({
            "pixel_size": cam["pixel_size"],
            "photons_per_adu": cam["photons_per_adu"],
            "baseline": cam["baseline"],
            "is_em_gain": cam["is_em_gain"],
            "em_gain": cam["em_gain"],
            "quantum_efficiency": cam["quantum_efficiency"],
        })

        try:
            t0 = time.time()
            detector = ThunderSTORMDetector(parameters=params)
            localizations = detector.detect_and_fit(image_stack, show_progress=True)
            elapsed = time.time() - t0

            n_locs = len(localizations.get('x', []))
            print(f"  Found {n_locs} localizations in {elapsed:.1f}s")

            # Save results
            csv_path = flika_dir / f"{test_name}_flika.csv"
            actual_path = detector.save_localizations(localizations, csv_path, image_stack=image_stack)
            csv_path = actual_path  # save_localizations may add _locsID suffix
            print(f"  Saved to: {csv_path}")

            results[test_name] = {
                "n_locs": n_locs,
                "time_s": elapsed,
                "csv_path": str(csv_path),
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = {
                "n_locs": 0,
                "time_s": 0,
                "csv_path": None,
                "error": str(e),
            }

    # Save results summary
    summary_path = output_dir / "flika_run_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nFLIKA run summary saved to: {summary_path}")

    return results


# --------------------------------------------------------------------------
# Save test metadata
# --------------------------------------------------------------------------

def save_test_metadata(output_dir, input_path):
    """Save metadata about the test run for later comparison."""
    meta = {
        "input_file": str(Path(input_path).resolve()),
        "camera": CAMERA_DEFAULTS,
        "tests": {},
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name, cfg in TEST_CONFIGS.items():
        meta["tests"][name] = {
            "description": cfg["description"],
            "imagej_params": cfg["imagej"],
            "flika_params": cfg["flika"],
        }

    meta_path = Path(output_dir) / "test_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Test metadata saved to: {meta_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate ImageJ macros and run FLIKA thunderSTORM for comparison"
    )
    parser.add_argument(
        "--input", "-i",
        default=str(project_root / "test_data" / "real" / "Endothelial_NonBapta_bin10_crop.tif"),
        help="Input .tif image stack"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(script_dir / "results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--tests", "-t",
        nargs="*",
        default=None,
        help="Specific test names to run (default: all). Available: " +
             ", ".join(TEST_CONFIGS.keys())
    )
    parser.add_argument(
        "--macro-only",
        action="store_true",
        help="Only generate ImageJ macros, don't run FLIKA tests"
    )
    parser.add_argument(
        "--flika-only",
        action="store_true",
        help="Only run FLIKA tests, don't generate macros"
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List available test configurations and exit"
    )

    args = parser.parse_args()

    if args.list_tests:
        print("Available test configurations:")
        for name, cfg in TEST_CONFIGS.items():
            print(f"  {name:30s} - {cfg['description']}")
        return

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input file:  {input_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Tests:       {args.tests or 'all'}")
    print()

    # Save metadata
    save_test_metadata(output_dir, input_path)

    # Generate macros
    if not args.flika_only:
        print("\n" + "="*60)
        print("GENERATING IMAGEJ MACROS")
        print("="*60)
        macro_path = generate_all_macros(input_path, output_dir, args.tests)
        print(f"\nTo run in ImageJ/Fiji:")
        print(f"  1. Open Fiji")
        print(f"  2. Plugins > Macros > Run...")
        print(f"  3. Select: {macro_path}")
        print(f"  (or drag-and-drop the .ijm file onto Fiji)")

    # Run FLIKA tests
    if not args.macro_only:
        print("\n" + "="*60)
        print("RUNNING FLIKA THUNDERSTORM")
        print("="*60)
        results = run_flika_tests(input_path, output_dir, args.tests)

        print("\n" + "="*60)
        print("FLIKA RESULTS SUMMARY")
        print("="*60)
        for name, res in results.items():
            status = f"{res['n_locs']} locs in {res['time_s']:.1f}s" if not res.get('error') else f"ERROR: {res['error']}"
            print(f"  {name:30s} {status}")

    if not args.flika_only:
        print(f"\nNext steps:")
        print(f"  1. Run the ImageJ macro: {output_dir / 'run_all_tests.ijm'}")
        print(f"  2. Compare results:  python compare_results.py -d {output_dir}")


if __name__ == "__main__":
    main()
