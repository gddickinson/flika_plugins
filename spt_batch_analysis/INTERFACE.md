# INTERFACE.md — Navigation Map

FLIKA plugin for batch single-particle tracking (SPT) analysis, with a pure-Python
reimplementation of ThunderSTORM for localization microscopy.

## Root modules (the FLIKA plugin package)

| File | Purpose |
|---|---|
| `__init__.py` | Main plugin: GUI, batch pipeline, dual linking methods, file logging. **14,790 lines — needs splitting into modules** (pre-dates the 500-line rule). |
| `config.py` | Configuration management (detection/tracking/classification/visualization parameters; YAML load, validation). |
| `logging_setup.py` | Console + rotating file logging for all components. |
| `thunderstorm_integration.py` | Bridges `thunderstorm_python` detection into the SPT batch pipeline as an alternative to U-Track detection. |
| `utrack_linking.py` | U-Track-compatible track linking (segment creation, gap closing). |
| `example_usage.py` | Minimal usage example inside FLIKA. |
| `about.html`, `info.xml` | FLIKA plugin metadata. |
| `spt_example_config.json` | Example batch configuration. |

## `thunderstorm_python/` — ThunderSTORM reimplementation

| File | Purpose |
|---|---|
| `filters.py` | Image filtering (wavelet, Gaussian, box, median…). |
| `detection.py` | Molecule detection (local maximum, non-maximum suppression, thresholding). |
| `fitting.py` | PSF fitters (`GaussianLSQFitter`, `GaussianWLSQFitter`, `GaussianMLEFitter`, `EllipticalGaussianMLEFitter`, `MultiEmitterFitter`, `CentroidFitter`, `RadialSymmetryFitter`, `PhasorFitter`), Numba-optimized. Also `compute_localization_precision` (Thompson formula) and `BaseFitter._photons_and_background_sd` (ADU→photons + background SD conversion — see `PATCH_NOTES_uncertainty.md`). |
| `pipeline.py` | End-to-end analysis pipeline tying filters → detection → fitting → post-processing. |
| `postprocessing.py` | Drift correction, merging, filtering of localizations. |
| `simulation.py` | Synthetic SMLM data generation. |
| `visualization.py` | Rendering of localization data. |
| `utils.py` | Shared helpers (I/O, units, conversions). |
| `examples.py` | Worked examples for the package. |
| `VALIDATION_REPORT.md` | Validation against ImageJ ThunderSTORM. |
| `fitting.py.bak` | Pre-patch backup of `fitting.py` (referenced by `PATCH_NOTES_uncertainty.md`). |

## Tests and data

| Path | Purpose |
|---|---|
| `tests/test_uncertainty_units.py` | Pins the uncertainty-units patch (photon conversion, background SD, baseline clamp, uncalibrated no-op). |
| `tests/test_single_frame_passthrough.py` | Single/multi-frame pipeline regression tests. |
| `tests/synthetic/` | Synthetic data generator + results for validation. |
| `tests/comparison/` | Comparison harness against ImageJ ThunderSTORM macros. |
| `test_data/` | Real + synthetic sample stacks (~44 MB). Not required by the test suite. |
| `training_data/` | Training data for classification. |

## How modules connect

FLIKA loads `__init__.py` → plugin GUI configures a batch run → detection is done
either via U-Track-style methods or `thunderstorm_integration.py` (which drives
`thunderstorm_python.pipeline`) → localizations are linked into tracks by
`utrack_linking.py` → results are logged (`logging_setup.py`) and configured via
`config.py`.

Run tests with `python3 -m pytest tests/` from the project root.
