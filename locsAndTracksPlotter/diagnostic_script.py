#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyQtGraph Compatibility Diagnostic Script

Run this script on each system to diagnose PyQtGraph and FLIKA version issues.

Usage:
    python diagnostic_script.py
    
Output:
    - System information
    - Python version
    - PyQtGraph version and capabilities
    - FLIKA version
    - Recommended actions

@author: george.dickinson@gmail.com
"""

import sys
import platform


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_section(text):
    """Print a formatted section header."""
    print(f"\n{text}")
    print("-" * 70)


def test_pyqtgraph():
    """Test PyQtGraph installation and capabilities."""
    print_section("PyQtGraph Tests")
    
    try:
        import pyqtgraph as pg
        print(f"✓ PyQtGraph imported successfully")
        
        # Version
        try:
            version = pg.__version__
            print(f"✓ Version: {version}")
        except AttributeError:
            print("✗ Could not determine PyQtGraph version")
            version = "unknown"
        
        # Test ScatterPlotItem
        print("\nScatterPlotItem capabilities:")
        try:
            scatter = pg.ScatterPlotItem()
            
            has_setData = hasattr(scatter, 'setData') and callable(scatter.setData)
            has_setPoints = hasattr(scatter, 'setPoints') and callable(scatter.setPoints)
            has_addPoints = hasattr(scatter, 'addPoints') and callable(scatter.addPoints)
            has_getData = hasattr(scatter, 'getData') and callable(scatter.getData)
            
            print(f"  setData:    {'✓ Available' if has_setData else '✗ Missing'}")
            print(f"  setPoints:  {'✓ Available' if has_setPoints else '✗ Missing'}")
            print(f"  addPoints:  {'✓ Available' if has_addPoints else '✗ Missing'}")
            print(f"  getData:    {'✓ Available' if has_getData else '✗ Missing'}")
            
            # Try each method
            print("\nMethod functionality tests:")
            import numpy as np
            test_x = np.array([1, 2, 3])
            test_y = np.array([1, 2, 3])
            
            if has_setData:
                try:
                    scatter.setData(x=test_x, y=test_y)
                    print("  ✓ setData() works correctly")
                except Exception as e:
                    print(f"  ✗ setData() failed: {e}")
            
            if has_setPoints:
                try:
                    scatter2 = pg.ScatterPlotItem()
                    scatter2.setPoints(x=test_x, y=test_y)
                    print("  ✓ setPoints() works correctly")
                except Exception as e:
                    print(f"  ✗ setPoints() failed: {e}")
            
            if has_addPoints:
                try:
                    scatter3 = pg.ScatterPlotItem()
                    scatter3.setData(x=[], y=[])
                    scatter3.addPoints(x=test_x, y=test_y)
                    print("  ✓ addPoints() works correctly")
                except Exception as e:
                    print(f"  ✗ addPoints() failed: {e}")
            
            return version, has_setData, has_setPoints, has_addPoints
            
        except Exception as e:
            print(f"✗ Error testing ScatterPlotItem: {e}")
            return version, False, False, False
            
    except ImportError as e:
        print(f"✗ PyQtGraph not found: {e}")
        return None, False, False, False


def test_flika():
    """Test FLIKA installation."""
    print_section("FLIKA Tests")
    
    try:
        import flika
        print(f"✓ FLIKA imported successfully")
        
        try:
            version = flika.__version__
            print(f"✓ Version: {version}")
        except AttributeError:
            print("✗ Could not determine FLIKA version")
            version = "unknown"
        
        # Check for global_vars
        try:
            import flika.global_vars as g
            print("✓ FLIKA global_vars accessible")
        except ImportError:
            print("✗ Could not import FLIKA global_vars")
        
        # Check for Window
        try:
            from flika.window import Window
            print("✓ FLIKA Window class accessible")
        except ImportError:
            print("✗ Could not import FLIKA Window")
        
        return version
        
    except ImportError as e:
        print(f"✗ FLIKA not found: {e}")
        return None


def test_qt():
    """Test Qt installation."""
    print_section("Qt Tests")
    
    qt_found = False
    qt_version = None
    qt_backend = None
    
    # Try PyQt5
    try:
        from PyQt5 import QtCore
        qt_version = QtCore.QT_VERSION_STR
        qt_backend = "PyQt5"
        qt_found = True
        print(f"✓ PyQt5 found (Qt {qt_version})")
    except ImportError:
        print("  PyQt5 not found")
    
    # Try PyQt6
    try:
        from PyQt6 import QtCore
        qt_version = QtCore.QT_VERSION_STR
        qt_backend = "PyQt6"
        qt_found = True
        print(f"✓ PyQt6 found (Qt {qt_version})")
    except ImportError:
        print("  PyQt6 not found")
    
    # Try PySide2
    try:
        from PySide2 import QtCore
        qt_version = QtCore.__version__
        qt_backend = "PySide2"
        qt_found = True
        print(f"✓ PySide2 found (Qt {qt_version})")
    except ImportError:
        print("  PySide2 not found")
    
    # Try PySide6
    try:
        from PySide6 import QtCore
        qt_version = QtCore.__version__
        qt_backend = "PySide6"
        qt_found = True
        print(f"✓ PySide6 found (Qt {qt_version})")
    except ImportError:
        print("  PySide6 not found")
    
    if not qt_found:
        print("✗ No Qt backend found")
    
    return qt_found, qt_version, qt_backend


def provide_recommendations(pg_version, has_setData, has_setPoints, has_addPoints, flika_version):
    """Provide recommendations based on test results."""
    print_section("Recommendations")
    
    if pg_version is None:
        print("✗ CRITICAL: PyQtGraph is not installed")
        print("  Action: Install PyQtGraph with: pip install pyqtgraph==0.11.1")
        return
    
    if flika_version is None:
        print("✗ CRITICAL: FLIKA is not installed")
        print("  Action: Install FLIKA according to documentation")
        return
    
    # Check compatibility
    if has_setData and has_addPoints:
        print("✓ PyQtGraph appears to be compatible")
        print(f"  Current version {pg_version} should work with the plugin")
    elif has_setPoints and has_addPoints:
        print("⚠ PyQtGraph is using older API (setPoints)")
        print(f"  Current version {pg_version} may have compatibility issues")
        print("  Action: Consider upgrading to PyQtGraph 0.11.1 or later")
    elif has_setData and not has_addPoints:
        print("⚠ PyQtGraph is using newer API without addPoints")
        print(f"  Current version {pg_version} may have partial compatibility")
        print("  The compatibility fixes in the plugin should handle this")
    else:
        print("✗ PyQtGraph configuration is unusual")
        print(f"  Current version {pg_version} may have issues")
        print("  Action: Reinstall PyQtGraph with: pip install --force-reinstall pyqtgraph==0.11.1")
    
    # Specific version recommendations
    if pg_version != "unknown":
        try:
            from distutils.version import StrictVersion
            if StrictVersion(pg_version) < StrictVersion('0.11.0'):
                print("\n⚠ PyQtGraph version is too old")
                print("  Action: Upgrade with: pip install --upgrade pyqtgraph==0.11.1")
            elif StrictVersion(pg_version) >= StrictVersion('0.13.0'):
                print("\n⚠ PyQtGraph version may be too new")
                print("  Action: Downgrade with: pip install pyqtgraph==0.11.1")
        except:
            pass
    
    print("\n" + "=" * 70)
    print("If problems persist after following recommendations:")
    print("1. Check FLIKA documentation for version compatibility")
    print("2. Try running FLIKA examples to verify base installation")
    print("3. Contact plugin developer with this diagnostic output")
    print("=" * 70)


def main():
    """Run all diagnostic tests."""
    print_header("FLIKA Plugin Compatibility Diagnostic")
    print(f"Running on: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version}")
    
    # Test PyQtGraph
    pg_version, has_setData, has_setPoints, has_addPoints = test_pyqtgraph()
    
    # Test FLIKA
    flika_version = test_flika()
    
    # Test Qt
    qt_found, qt_version, qt_backend = test_qt()
    
    # Provide recommendations
    provide_recommendations(pg_version, has_setData, has_setPoints, has_addPoints, flika_version)
    
    print("\n")


if __name__ == "__main__":
    main()
