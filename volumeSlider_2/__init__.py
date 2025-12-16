#!/usr/bin/env python3
"""
Volume Slider Plugin - Professional Edition
==========================================

A comprehensive 3D/4D lightsheet microscopy analysis plugin for FLIKA.

This package provides:
- Professional GUI with dockable panels
- Real-time 3D visualization  
- Advanced image processing algorithms
- Synthetic test data generation
- Comprehensive error handling and logging
- Batch processing capabilities
- Export to multiple formats including Imaris

Author: Professional Refactor Team
Version: 2.0.0
License: GPL v3
FLIKA Compatibility: 0.2.23+
"""

__version__ = "2.0.0"
__author__ = "Professional Refactor Team"
__license__ = "GPL v3"
__flika_version_required__ = "0.2.23"

import sys
import logging
from pathlib import Path

# Check Python version
if sys.version_info < (3, 7):
    raise RuntimeError("Volume Slider Plugin requires Python 3.7 or higher")

# Setup package-level logging
logger = logging.getLogger(__name__)

# Check if FLIKA is available
try:
    import flika
    from flika import global_vars as g
    from distutils.version import StrictVersion
    
    # Check FLIKA version compatibility
    flika_version = flika.__version__
    if StrictVersion(flika_version) < StrictVersion(__flika_version_required__):
        logger.warning(f"FLIKA version {flika_version} detected. "
                      f"Recommended version: {__flika_version_required__}+")
    
    HAS_FLIKA = True
    logger.info(f"FLIKA {flika_version} detected - plugin mode available")
    
except ImportError:
    HAS_FLIKA = False
    logger.info("FLIKA not found - running in standalone mode")

# Import version checking
try:
    from distutils.version import StrictVersion
    HAS_VERSION_CHECK = True
except ImportError:
    HAS_VERSION_CHECK = False
    logger.warning("Version checking not available")

# Check for required dependencies
REQUIRED_PACKAGES = {
    'numpy': '1.20.0',
    'scipy': '1.7.0', 
    'scikit-image': '0.18.0',
    'pyqtgraph': '0.12.0',
    'qtpy': '1.9.0'
}

OPTIONAL_PACKAGES = {
    'tifffile': '2020.1.1',
    'h5py': '3.0.0',
    'pandas': '1.2.0',
    'numba': '0.53.0'
}

def check_dependencies(show_warnings=True):
    """
    Check if required dependencies are available.
    
    Args:
        show_warnings: Whether to show warning messages
        
    Returns:
        dict: Status of each dependency
    """
    status = {'required': {}, 'optional': {}, 'all_required_available': True}
    
    # Check required packages
    for package, min_version in REQUIRED_PACKAGES.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            
            if HAS_VERSION_CHECK and min_version and version != 'unknown':
                version_ok = StrictVersion(version) >= StrictVersion(min_version)
            else:
                version_ok = True  # Assume OK if we can't check
            
            status['required'][package] = {
                'available': True,
                'version': version,
                'version_ok': version_ok
            }
            
            if not version_ok:
                status['all_required_available'] = False
                if show_warnings:
                    logger.warning(f"{package} version {version} < required {min_version}")
                    
        except ImportError:
            status['required'][package] = {
                'available': False,
                'version': None,
                'version_ok': False
            }
            status['all_required_available'] = False
            if show_warnings:
                logger.error(f"Required package '{package}' not found")
    
    # Check optional packages
    for package, min_version in OPTIONAL_PACKAGES.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            
            if HAS_VERSION_CHECK and min_version and version != 'unknown':
                version_ok = StrictVersion(version) >= StrictVersion(min_version)
            else:
                version_ok = True
            
            status['optional'][package] = {
                'available': True,
                'version': version,
                'version_ok': version_ok
            }
            
            if not version_ok and show_warnings:
                logger.warning(f"{package} version {version} < recommended {min_version}")
                
        except ImportError:
            status['optional'][package] = {
                'available': False,
                'version': None,
                'version_ok': False
            }
            if show_warnings:
                logger.info(f"Optional package '{package}' not found - some features may be limited")
    
    return status

# Check dependencies on import
_dependency_status = check_dependencies(show_warnings=False)

# Import core components based on availability
if _dependency_status['all_required_available']:
    try:
        # Import main plugin classes
        if HAS_FLIKA:
            from .volumeSlider_Start import volumeSliderBase
            logger.info("FLIKA plugin interface loaded")
        
        # Always make standalone interface available
        from .main import VolumeSliderPlugin
        logger.info("Standalone interface loaded")
        
        # Import core functionality
        from .core.data_manager import DataManager
        from .synthetic.test_data_generator import TestDataGenerator
        
        PLUGIN_LOADED = True
        logger.info("Volume Slider Plugin loaded successfully")
        
    except ImportError as e:
        PLUGIN_LOADED = False
        logger.error(f"Failed to load plugin components: {str(e)}")
        
else:
    PLUGIN_LOADED = False
    logger.error("Cannot load plugin - missing required dependencies")

# Plugin metadata for FLIKA
PLUGIN_INFO = {
    'name': 'Volume Slider Professional',
    'version': __version__,
    'author': __author__,
    'description': 'Professional 3D/4D lightsheet microscopy analysis',
    'license': __license__,
    'url': 'https://github.com/your-org/volume-slider-professional',
    'flika_version_required': __flika_version_required__,
    'python_version_required': '3.7+',
    'dependencies': REQUIRED_PACKAGES,
    'optional_dependencies': OPTIONAL_PACKAGES
}

def get_plugin_info():
    """Get plugin information dictionary."""
    info = PLUGIN_INFO.copy()
    info['loaded'] = PLUGIN_LOADED
    info['has_flika'] = HAS_FLIKA
    info['dependency_status'] = _dependency_status
    return info

def print_plugin_status():
    """Print current plugin status to console."""
    info = get_plugin_info()
    
    print(f"\n{info['name']} v{info['version']}")
    print("=" * 50)
    print(f"Status: {'✓ Loaded' if info['loaded'] else '✗ Failed to load'}")
    print(f"FLIKA Integration: {'✓ Available' if info['has_flika'] else '✗ Not available'}")
    print(f"Python: {sys.version.split()[0]} ({'✓' if sys.version_info >= (3, 7) else '✗'})")
    
    print("\nRequired Dependencies:")
    for pkg, status in info['dependency_status']['required'].items():
        symbol = '✓' if status['available'] and status['version_ok'] else '✗'
        version = f" v{status['version']}" if status['version'] else ""
        print(f"  {symbol} {pkg}{version}")
    
    print("\nOptional Dependencies:")
    for pkg, status in info['dependency_status']['optional'].items():
        symbol = '✓' if status['available'] else '○'
        version = f" v{status['version']}" if status['version'] else ""
        print(f"  {symbol} {pkg}{version}")
    
    print()

# Make key components available at package level
if PLUGIN_LOADED:
    # Core classes
    __all__ = [
        'VolumeSliderPlugin',
        'DataManager', 
        'TestDataGenerator',
        'get_plugin_info',
        'print_plugin_status',
        'check_dependencies'
    ]
    
    # Add FLIKA interface if available
    if HAS_FLIKA:
        __all__.append('volumeSliderBase')
        
else:
    __all__ = ['get_plugin_info', 'print_plugin_status', 'check_dependencies']

# Plugin initialization message
if __name__ != "__main__":
    if PLUGIN_LOADED:
        logger.info(f"Volume Slider Plugin v{__version__} ready")
    else:
        logger.warning("Volume Slider Plugin failed to load - check dependencies")

# Backwards compatibility with original plugin interface
try:
    if HAS_FLIKA and PLUGIN_LOADED:
        # Create global instance for FLIKA compatibility
        volumeSliderBase = volumeSliderBase()
except NameError:
    pass  # FLIKA interface not available
