# tirf_analysis_suite/__init__.py
"""
TIRF Analysis Suite for FLIKA
=============================

A comprehensive collection of advanced analysis tools specifically designed for 
Total Internal Reflection Fluorescence (TIRF) microscopy studies of fluorescently-labeled proteins.

This suite provides state-of-the-art algorithms with publication-ready visualizations 
and seamless integration with FLIKA workflows.

Components:
- Single Molecule Tracker: Subpixel accuracy tracking with advanced linking
- Photobleaching Analyzer: Step counting for oligomerization studies  
- TIRF Background Corrector: Advanced uneven illumination correction
- Colocalization Analyzer: Multi-channel analysis with statistical validation
- Membrane Dynamics Analyzer: Cell edge movement and dynamics analysis
- FRAP Analyzer: Recovery kinetics with multiple models
- Cluster Analyzer: Protein aggregate detection and characterization
- Suite Manager: Help, validation, and utility functions

Author: FLIKA Plugin Development Team
Version: 1.0.0
License: MIT
"""

import sys
import os
import traceback

# Add the suite directory to Python path for relative imports
suite_dir = os.path.dirname(os.path.abspath(__file__))
if suite_dir not in sys.path:
    sys.path.insert(0, suite_dir)

# Suite metadata
__version__ = '1.0.0'
__author__ = 'FLIKA Plugin Development Team'
__email__ = 'support@flika-plugins.org'
__license__ = 'MIT'
__description__ = 'Comprehensive TIRF microscopy analysis suite for FLIKA'

# Import FLIKA essentials
try:
    from flika import global_vars as g
    from flika.window import Window
    from flika.utils.BaseProcess import BaseProcess
    print("TIRF Analysis Suite: FLIKA imports successful")
except ImportError as e:
    print(f"TIRF Analysis Suite: Warning - FLIKA import failed: {e}")
    g = None

# Suite initialization flag
_suite_initialized = False

def initialize_suite():
    """Initialize the TIRF Analysis Suite"""
    global _suite_initialized
    
    if _suite_initialized:
        return True
    
    print("Initializing TIRF Analysis Suite v{}...".format(__version__))
    
    try:
        # Import and register all plugins
        success = True
        
        # Import Single Molecule Tracker
        try:
            from .single_molecule_tracker import SingleMoleculeTracker
            print("✓ Single Molecule Tracker loaded")
        except Exception as e:
            print(f"✗ Single Molecule Tracker failed: {e}")
            success = False
        
        # Import Photobleaching Analyzer
        try:
            from .photobleaching_analyzer import PhotobleachingAnalyzer
            print("✓ Photobleaching Analyzer loaded")
        except Exception as e:
            print(f"✗ Photobleaching Analyzer failed: {e}")
            success = False
        
        # Import TIRF Background Corrector
        try:
            from .tirf_background_corrector import TIRFBackgroundCorrector, create_tirf_preprocessing_workflow
            print("✓ TIRF Background Corrector loaded")
        except Exception as e:
            print(f"✗ TIRF Background Corrector failed: {e}")
            success = False
        
        # Import Colocalization Analyzer
        try:
            from .colocalization_analyzer import ColocalizationAnalyzer
            print("✓ Colocalization Analyzer loaded")
        except Exception as e:
            print(f"✗ Colocalization Analyzer failed: {e}")
            success = False
        
        # Import Membrane Dynamics Analyzer
        try:
            from .membrane_dynamics_analyzer import MembraneDynamicsAnalyzer
            print("✓ Membrane Dynamics Analyzer loaded")
        except Exception as e:
            print(f"✗ Membrane Dynamics Analyzer failed: {e}")
            success = False
        
        # Import FRAP Analyzer
        try:
            from .frap_analyzer import FRAPAnalyzer
            print("✓ FRAP Analyzer loaded")
        except Exception as e:
            print(f"✗ FRAP Analyzer failed: {e}")
            success = False
        
        # Import Cluster Analyzer
        try:
            from .cluster_analyzer import ClusterAnalyzer
            print("✓ Cluster Analyzer loaded")
        except Exception as e:
            print(f"✗ Cluster Analyzer failed: {e}")
            success = False
        
        # Import Suite Manager and Utilities
        try:
            from .tirf_suite_manager import (show_suite_help, validate_installation, 
                                           export_analysis_report)
            print("✓ Suite Manager loaded")
        except Exception as e:
            print(f"✗ Suite Manager failed: {e}")
            success = False
        
        # Import Synthetic Data Generator
        try:
            from .synthetic_data_generator import (generate_single_molecule_test_data,
                                                 generate_photobleaching_test_data,
                                                 generate_membrane_test_data,
                                                 generate_frap_test_data,
                                                 generate_cluster_test_data,
                                                 generate_colocalization_test_data,
                                                 generate_complete_test_suite)
            print("✓ Synthetic Data Generator loaded")
        except Exception as e:
            print(f"✗ Synthetic Data Generator failed: {e}")
            success = False
        
        # Import Integrated Workflows
        try:
            from .integrated_workflows import (run_single_molecule_workflow,
                                             run_photobleaching_workflow,
                                             run_membrane_dynamics_workflow,
                                             run_colocalization_workflow,
                                             show_workflow_help)
            print("✓ Integrated Workflows loaded")
        except Exception as e:
            print(f"✗ Integrated Workflows failed: {e}")
            success = False
        
        if success:
            _suite_initialized = True
            print("🎉 TIRF Analysis Suite initialization complete!")
            print("📍 Access plugins through: Plugins > TIRF Analysis")
            
            # Display welcome message if FLIKA is available
            if g is not None:
                try:
                    welcome_message = """
TIRF Analysis Suite v{} loaded successfully! 🔬

Available tools:
• Single Molecule Analysis
• Photobleaching Analysis  
• Background Correction
• Colocalization Analysis
• Membrane Dynamics
• FRAP Analysis
• Cluster Analysis
• Utilities & Workflows

Access through: Plugins > TIRF Analysis

For help: Plugins > TIRF Analysis > Utilities > Plugin Suite Help
""".format(__version__)
                    print(welcome_message)
                except:
                    pass
            
            return True
        else:
            print("⚠️ TIRF Analysis Suite initialization completed with errors")
            return False
            
    except Exception as e:
        print(f"❌ TIRF Analysis Suite initialization failed: {e}")
        traceback.print_exc()
        return False

def get_suite_info():
    """Get information about the suite"""
    return {
        'name': 'TIRF Analysis Suite',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'license': __license__,
        'description': __description__,
        'initialized': _suite_initialized
    }

def validate_suite():
    """Validate that all suite components are working"""
    if not _suite_initialized:
        print("Suite not initialized. Call initialize_suite() first.")
        return False
    
    print("Validating TIRF Analysis Suite components...")
    
    validation_results = {
        'suite_info': get_suite_info(),
        'component_status': {},
        'overall_status': 'unknown'
    }
    
    # Test imports
    components = [
        ('single_molecule_tracker', 'SingleMoleculeTracker'),
        ('photobleaching_analyzer', 'PhotobleachingAnalyzer'),
        ('tirf_background_corrector', 'TIRFBackgroundCorrector'),
        ('colocalization_analyzer', 'ColocalizationAnalyzer'),
        ('membrane_dynamics_analyzer', 'MembraneDynamicsAnalyzer'),
        ('frap_analyzer', 'FRAPAnalyzer'),
        ('cluster_analyzer', 'ClusterAnalyzer'),
        ('tirf_suite_manager', 'show_suite_help'),
        ('synthetic_data_generator', 'generate_complete_test_suite'),
        ('integrated_workflows', 'run_single_molecule_workflow')
    ]
    
    all_working = True
    
    for module_name, class_or_function in components:
        try:
            module = __import__(module_name, fromlist=[class_or_function])
            getattr(module, class_or_function)
            validation_results['component_status'][module_name] = 'working'
            print(f"✓ {module_name}: OK")
        except Exception as e:
            validation_results['component_status'][module_name] = f'error: {str(e)}'
            print(f"✗ {module_name}: {e}")
            all_working = False
    
    validation_results['overall_status'] = 'working' if all_working else 'errors_detected'
    
    if all_working:
        print("🎉 All suite components validated successfully!")
    else:
        print("⚠️ Some components have issues. Check individual status above.")
    
    return validation_results

# Suite-level convenience functions
def show_suite_status():
    """Display current suite status"""
    info = get_suite_info()
    print("\n" + "="*50)
    print(f"TIRF Analysis Suite v{info['version']}")
    print("="*50)
    print(f"Author: {info['author']}")
    print(f"Status: {'Initialized' if info['initialized'] else 'Not Initialized'}")
    print(f"Description: {info['description']}")
    print("="*50)
    
    if info['initialized']:
        print("\n📋 Available Components:")
        print("• Single Molecule Tracker")
        print("• Photobleaching Analyzer")
        print("• TIRF Background Corrector")
        print("• Colocalization Analyzer")
        print("• Membrane Dynamics Analyzer")
        print("• FRAP Analyzer")
        print("• Cluster Analyzer")
        print("• Suite Manager & Utilities")
        print("• Synthetic Data Generator")
        print("• Integrated Workflows")
        
        print("\n🚀 Quick Start:")
        print("1. Plugins > TIRF Analysis > Utilities > Generate Test Data")
        print("2. Choose an analysis tool from the TIRF Analysis menu")
        print("3. For help: Plugins > TIRF Analysis > Utilities > Plugin Suite Help")
    else:
        print("\n⚠️ Suite not initialized. Some components may not be available.")
    
    print("\n")

def get_citation_info():
    """Get citation information for the suite"""
    citation = """
When using the TIRF Analysis Suite in your research, please cite:

"Advanced TIRF Analysis Suite for FLIKA: Comprehensive tools for fluorescence microscopy analysis"
FLIKA Plugin Development Team (2024)
GitHub: https://github.com/flika-org/tirf_analysis_suite

Also cite the original FLIKA paper:
Ellefsen, K., Settle, B., Parker, I. & Smith, I. 
"An algorithm for automated detection, localization and measurement of local calcium signals from camera-based imaging." 
Cell Calcium. 56:147-156, 2014
DOI: 10.1016/j.ceca.2014.08.003
"""
    return citation

# Auto-initialize when the module is imported
try:
    # Only initialize if FLIKA is available
    if g is not None:
        initialization_success = initialize_suite()
        if not initialization_success:
            print("⚠️ TIRF Analysis Suite: Initialization had errors. Some features may not work.")
    else:
        print("⚠️ TIRF Analysis Suite: FLIKA not detected. Suite will be loaded but may not function properly.")
except Exception as e:
    print(f"❌ TIRF Analysis Suite: Auto-initialization failed: {e}")

# Make key functions available at module level
__all__ = [
    'initialize_suite',
    'validate_suite', 
    'get_suite_info',
    'show_suite_status',
    'get_citation_info',
    '__version__',
    '__author__',
    '__description__'
]

# Suite-level menu functions (for direct menu integration if needed)
def launch_suite_help():
    """Launch the comprehensive help system"""
    try:
        from .tirf_suite_manager import show_suite_help
        show_suite_help()
    except Exception as e:
        if g is not None:
            g.alert(f"Could not launch help system: {e}")

def launch_validation():
    """Launch the validation tool"""
    try:
        from .tirf_suite_manager import validate_installation
        validate_installation()
    except Exception as e:
        if g is not None:
            g.alert(f"Could not launch validation: {e}")

def launch_test_data_generator():
    """Launch complete test data generation"""
    try:
        from .synthetic_data_generator import generate_complete_test_suite
        generate_complete_test_suite()
    except Exception as e:
        if g is not None:
            g.alert(f"Could not generate test data: {e}")

# Register suite-level menu items
if g is not None:
    try:
        launch_suite_help.menu_path = 'Plugins>TIRF Analysis>📖 Suite Help & Documentation'
        launch_validation.menu_path = 'Plugins>TIRF Analysis>🔧 Validate Installation'
        launch_test_data_generator.menu_path = 'Plugins>TIRF Analysis>🧪 Generate All Test Data'
    except:
        pass

# Print initialization status
if _suite_initialized:
    print(f"✅ TIRF Analysis Suite v{__version__} ready for use!")
else:
    print(f"⚠️ TIRF Analysis Suite v{__version__} loaded with warnings.")