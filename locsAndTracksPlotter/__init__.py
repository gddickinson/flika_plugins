#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIKA LocsAndTracksPlotter Plugin Initialization - V4 DEBUG

This module initializes the LocsAndTracksPlotter Plugin and applies necessary patches
for PyQtGraph compatibility across different versions, with enhanced debugging.

@author: george.dickinson@gmail.com
"""

import logging
import sys

# Set up detailed logging - IMPORTANT: This will show debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Apply PyQtGraph compatibility patches BEFORE importing anything else
print("\n" + "=" * 70)
print("FLIKA LocsAndTracksPlotter Plugin V4 DEBUG - Initialization")
print("=" * 70)
logger.info("Initializing FLIKA LocsAndTracksPlotter Plugin with compatibility patches...")

try:
    import pyqtgraph as pg
    import numpy as np
    from qtpy.QtGui import QColor

    pg_version = getattr(pg, '__version__', 'unknown')
    print(f"PyQtGraph version: {pg_version}")
    logger.info(f"PyQtGraph version: {pg_version}")

    # ===================================================================
    # PATCH 1: ScatterPlotItem.setPoints
    # ===================================================================
    test_scatter = pg.ScatterPlotItem()
    needs_scatter_patch = not (hasattr(test_scatter, 'setPoints') and callable(test_scatter.setPoints))

    if needs_scatter_patch:
        print("Applying ScatterPlotItem.setPoints patch...")
        logger.info("Applying ScatterPlotItem.setPoints patch...")

        def setPoints(self, *args, **kwargs):
            """
            Compatibility wrapper: setPoints -> setData

            This allows FLIKA's old API calls to work with new PyQtGraph versions.
            Properly handles FLIKA's format: [[x, y, QColor, size], ...]
            """
            # Handle different parameter formats
            if 'pos' in kwargs:
                pos = kwargs.pop('pos')

                # Check if pos is in FLIKA's format: [[x, y, color, size], ...]
                if len(pos) > 0:
                    try:
                        # Check if it's the old FLIKA format with 4 elements per point
                        if hasattr(pos[0], '__len__') and len(pos[0]) >= 4:
                            # FLIKA format: [x, y, QColor, size]
                            coords = []
                            sizes = []
                            brushes = []

                            for p in pos:
                                # Extract coordinates
                                coords.append([p[0], p[1]])

                                # Extract color (element 2)
                                color = p[2]
                                if isinstance(color, QColor):
                                    brushes.append(pg.mkBrush(color))
                                else:
                                    # Fallback to white if not a QColor
                                    brushes.append(pg.mkBrush(255, 255, 255, 255))

                                # Extract size (element 3)
                                sizes.append(p[3])

                            # Set the data with extracted attributes
                            kwargs['pos'] = np.array(coords)
                            kwargs['size'] = sizes
                            kwargs['brush'] = brushes

                        elif hasattr(pos[0], '__len__') and len(pos[0]) >= 2:
                            # Simple format: [[x, y], ...] or [[x, y, ...], ...]
                            coords = np.array([[p[0], p[1]] for p in pos])
                            kwargs['pos'] = coords

                            # Only extract size/brush if not already specified and available
                            if 'size' not in kwargs and len(pos[0]) > 3:
                                try:
                                    sizes = [p[3] for p in pos]
                                    kwargs['size'] = sizes
                                except (IndexError, TypeError):
                                    pass

                            if 'brush' not in kwargs and len(pos[0]) > 2:
                                try:
                                    brushes = []
                                    for p in pos:
                                        color = p[2]
                                        if isinstance(color, QColor):
                                            brushes.append(pg.mkBrush(color))
                                        else:
                                            brushes.append(pg.mkBrush(255, 255, 255, 255))
                                    kwargs['brush'] = brushes
                                except (IndexError, TypeError):
                                    pass
                        else:
                            # Scalar or unknown format - pass through
                            kwargs['pos'] = pos
                    except Exception as e:
                        logger.warning(f"Error parsing pos format, using as-is: {e}")
                        # Fallback: use pos as-is
                        kwargs['pos'] = pos
                else:
                    # Empty array
                    kwargs['pos'] = pos

            # Call setData with converted parameters
            return self.setData(*args, **kwargs)

        # Add setPoints method to ScatterPlotItem
        pg.ScatterPlotItem.setPoints = setPoints
        print("✓ ScatterPlotItem.setPoints patch applied")
        print("  - Preserves point size and color from FLIKA format")
        logger.info("✓ ScatterPlotItem.setPoints patch applied")
        logger.info("  - Preserves point size and color from FLIKA format")
    else:
        print("✓ ScatterPlotItem already has setPoints method")
        logger.info("✓ ScatterPlotItem already has setPoints method")

    # ===================================================================
    # PATCH 2: PlotItem.setBackground
    # ===================================================================
    test_plotitem = pg.PlotItem()
    needs_background_patch = not hasattr(test_plotitem, 'setBackground')

    if needs_background_patch:
        print("Applying PlotItem.setBackground patch...")
        logger.info("Applying PlotItem.setBackground patch...")

        def setBackground(self, color):
            """
            Compatibility wrapper for setBackground.

            In newer PyQtGraph versions, use getViewBox().setBackgroundColor() instead.
            """
            try:
                viewbox = self.getViewBox()
                if viewbox and hasattr(viewbox, 'setBackgroundColor'):
                    viewbox.setBackgroundColor(color)
                else:
                    logger.warning("ViewBox does not have setBackgroundColor method")
            except Exception as e:
                logger.error(f"Error in setBackground patch: {e}")

        pg.PlotItem.setBackground = setBackground
        print("✓ PlotItem.setBackground patch applied")
        logger.info("✓ PlotItem.setBackground patch applied")
    else:
        print("✓ PlotItem already has setBackground method")
        logger.info("✓ PlotItem already has setBackground method")

    # Check PyQtGraph plotting classes
    print("\nChecking PyQtGraph plotting classes...")
    logger.info("Checking PyQtGraph plotting classes...")

    try:
        test_curve = pg.PlotCurveItem()
        has_setData_curve = hasattr(test_curve, 'setData')
        print(f"  PlotCurveItem.setData: {'✓' if has_setData_curve else '✗'}")
        logger.info(f"  PlotCurveItem.setData: {'✓' if has_setData_curve else '✗'}")

        test_plot = pg.PlotDataItem()
        has_setData_plot = hasattr(test_plot, 'setData')
        print(f"  PlotDataItem.setData: {'✓' if has_setData_plot else '✗'}")
        logger.info(f"  PlotDataItem.setData: {'✓' if has_setData_plot else '✗'}")

    except Exception as e:
        logger.warning(f"  Could not check plotting classes: {e}")

    # PyQtGraph configuration
    print("\nPyQtGraph configuration:")
    print(f"  useOpenGL: {pg.getConfigOption('useOpenGL')}")
    print(f"  antialias: {pg.getConfigOption('antialias')}")
    print(f"  imageAxisOrder: {pg.getConfigOption('imageAxisOrder')}")
    logger.info(f"PyQtGraph config - OpenGL: {pg.getConfigOption('useOpenGL')}, antialias: {pg.getConfigOption('antialias')}")

    # Check Qt backend
    try:
        from qtpy import API_NAME, QT_VERSION
        print(f"\nQt backend: {API_NAME} {QT_VERSION}")
        logger.info(f"Qt backend: {API_NAME} {QT_VERSION}")
    except:
        print("\nQt backend: Could not determine")
        logger.info("Qt backend: Could not determine")

except Exception as e:
    print(f"\n✗ ERROR applying PyQtGraph patches: {e}")
    logger.error(f"Error applying PyQtGraph patches: {e}")
    logger.error("Plugin may not work correctly")
    import traceback
    traceback.print_exc()

# Now import the actual plugin
print("\n" + "=" * 70)
print("Loading LocsAndTracksPlotter Plugin modules...")
print("=" * 70)
logger.info("Loading LocsAndTracksPlotter Plugin modules...")

try:
    from .locsAndTracksPlotter import locsAndTracksPlotter
    print("✓ LocsAndTracksPlotter Plugin loaded successfully")
    print("=" * 70)
    print("Plugin initialization complete!")
    print("=" * 70 + "\n")
    logger.info("✓ LocsAndTracksPlotter Plugin loaded successfully")
except ImportError as e:
    print(f"✗ Failed to load LocsAndTracksPlotter Plugin: {e}")
    logger.error(f"Failed to load LocsAndTracksPlotter Plugin: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ Unexpected error loading plugin: {e}")
    logger.error(f"Unexpected error loading plugin: {e}")
    import traceback
    traceback.print_exc()
