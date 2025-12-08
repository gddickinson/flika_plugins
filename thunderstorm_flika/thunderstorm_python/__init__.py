"""
ThunderSTORM Python Implementation
====================================

A comprehensive Python clone of the thunderSTORM ImageJ plugin for
Single Molecule Localization Microscopy (SMLM) data analysis.

This package provides:
- Image filtering and enhancement
- Molecule detection and localization
- PSF fitting (Gaussian, MLE, LSQ, WLSQ)
- 3D astigmatism-based localization
- Multiple emitter fitting
- Drift correction
- Post-processing and filtering
- Visualization methods
- Simulation tools

Author: George (based on thunderSTORM by Ovesný et al., 2014)
Reference: Ovesný, M., Křížek, P., Borkovec, J., Švindrych, Z., & Hagen, G. M. (2014).
          ThunderSTORM: a comprehensive ImageJ plugin for PALM and STORM data analysis
          and super-resolution imaging. Bioinformatics, 30(16), 2389-2390.
"""

__version__ = "1.0.0"
__author__ = "George"

from . import filters
from . import detection
from . import fitting
from . import postprocessing
from . import visualization
from . import simulation
from . import utils
from .pipeline import ThunderSTORM, create_default_pipeline, quick_analysis

__all__ = [
    'filters',
    'detection',
    'fitting',
    'postprocessing',
    'visualization',
    'simulation',
    'utils',
    'ThunderSTORM',
    'create_default_pipeline',
    'quick_analysis'
]
