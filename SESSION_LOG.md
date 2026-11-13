# Session log

## 2026-11-12: web catalogue, demos and bug fixes (branch web-interface)
- Added `web/` (FastAPI + static ES modules), `scripts/build_static.py`, `scripts/screenshots.py`,
  `scripts/make_demo_fixture.py`, `tests/`, `run_web.py` (port 8198), `requirements-web.txt`, Pages workflow (master).
- Catalogue: 64 info.xml folders, 530 Python files parsed with ast (4 Python-2 files flagged), READMEs rendered.
- Demos (JS ports, node-tested against the plugin Python): ThunderSTORM localization (spt_batch_analysis
  thunderstorm_python; 12 frame/settings cases, 327 fits), track features + directional autocorrelation
  (spt_batch_analysis), FFT in chunks (fft_Chunker). On the real test frame the WLSQ/4-neighbourhood port matches
  all 41 ImageJ ThunderSTORM localizations within ~0.002 nm RMS.
- Fixes (separate commits, regression tests in `tests/test_fix_*.py`):
  - NumPy 2: np.cross of 2-vectors and complex eig in spt FeatureCalculator and locsAndTracksPlotter JoinTracks.
  - spt AutocorrelationAnalyzer joined tracks that start after the previous one ended; the last lag was always empty.
  - fft_Chunker makeAverage kept only the last ROI trace's averages.
  - flash_remover interpolation stopped one frame short (overwrote the frame after the flash).
  - kymograph_analyzer threshold/zscore events mispaired when a trace starts above threshold; start one frame early.
  - Unused pywt import in thunderstorm_python (failed without PyWavelets, which info.xml does not list).
  - Persistence docstring (random directions give ~0.5, not 0). README plugin list now says where each one lives.
- Needs George's decision: fft_Chunker group stop chunk (exclusive, crashes when start == stop); kymograph velocity
  formula (dx/dt gradient ratio); see reports/flika_plugins.md.
