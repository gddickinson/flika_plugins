"""Regression: spt_batch_analysis/thunderstorm_python imported PyWavelets (pywt) without using it.

pywt is not among the plugin's dependencies in info.xml, so without it `import thunderstorm_python` failed and the
SPT plugin silently fell back to running without its ThunderSTORM detection (thunderstorm_integration's ImportError
path). The unused import is removed.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_filters_import_without_pywt():
    code = ("import sys; sys.modules['pywt'] = None; sys.path.insert(0, 'spt_batch_analysis/thunderstorm_python'); "
            "import filters; print(filters.WaveletFilter().name)")
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "Wavelet", r.stderr
