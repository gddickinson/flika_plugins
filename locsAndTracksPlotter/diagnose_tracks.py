#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLIKA Track Display Diagnostic

This script helps diagnose why tracks aren't displaying when points are visible.

Run this FROM WITHIN FLIKA after loading your data:
1. Start FLIKA
2. Load your tracking data
3. In FLIKA's Python console, run:
   exec(open('diagnose_tracks.py').read())

@author: george.dickinson@gmail.com
"""

import sys


def diagnose_track_display():
    """Diagnose track display issues."""
    print("\n" + "=" * 70)
    print("FLIKA Track Display Diagnostic")
    print("=" * 70)
    
    try:
        import flika.global_vars as g
        import numpy as np
        
        # Check if FLIKA is running
        if not hasattr(g, 'm') or g.m is None:
            print("\n✗ FLIKA main window not found")
            print("This script must be run from within FLIKA")
            return False
        
        print("\n✓ FLIKA is running")
        
        # Check current window
        if not hasattr(g, 'win') or g.win is None:
            print("✗ No current window - load a TIFF stack first")
            return False
        
        print(f"✓ Current window: {g.win.name}")
        print(f"  Shape: {g.win.image.shape}")
        
        # Check for tracking plugin
        print("\n" + "-" * 70)
        print("Checking Plugin Status")
        print("-" * 70)
        
        try:
            # Try to access the plugin
            # The plugin is typically in g.m or as a separate object
            plugin_found = False
            plugin_obj = None
            
            # Look for the plugin in different places
            if hasattr(g.m, 'locsAndTracksPlotter'):
                plugin_obj = g.m.locsAndTracksPlotter
                plugin_found = True
                print("✓ Plugin found at g.m.locsAndTracksPlotter")
            
            # Check if plugin has been instantiated
            if not plugin_found:
                # Try importing it
                try:
                    from tracking.locsAndTracksPlotter import locsAndTracksPlotter
                    plugin_obj = locsAndTracksPlotter
                    plugin_found = True
                    print("✓ Plugin imported successfully")
                except ImportError:
                    print("✗ Could not import tracking plugin")
                    return False
            
            if not plugin_found or plugin_obj is None:
                print("✗ Plugin not loaded")
                print("  Action: Load the tracking plugin through FLIKA's menu")
                return False
            
            # Check if plugin has data loaded
            print("\n" + "-" * 70)
            print("Checking Data Status")
            print("-" * 70)
            
            if not hasattr(plugin_obj, 'data') or plugin_obj.data is None:
                print("✗ No tracking data loaded")
                print("  Action: Load CSV/HDF5 data file with tracks")
                return False
            
            print(f"✓ Data loaded: {len(plugin_obj.data)} rows")
            
            # Check data columns
            print(f"  Columns: {list(plugin_obj.data.columns)[:10]}...")
            
            # Check for track_number column
            if 'track_number' not in plugin_obj.data.columns:
                print("✗ No 'track_number' column found")
                print("  Data appears to be localizations only, not tracks")
                return False
            
            print("✓ 'track_number' column exists")
            
            # Check track statistics
            track_numbers = plugin_obj.data['track_number'].unique()
            num_tracks = len(track_numbers)
            print(f"✓ Found {num_tracks} unique tracks")
            
            if num_tracks == 0:
                print("✗ No tracks in data (only unlinked points)")
                print("  This data hasn't been linked/tracked yet")
                return False
            
            # Check track lengths
            track_lengths = plugin_obj.data.groupby('track_number').size()
            print(f"  Track length range: {track_lengths.min()} - {track_lengths.max()} points")
            print(f"  Mean track length: {track_lengths.mean():.1f} points")
            
            # Check plot window
            print("\n" + "-" * 70)
            print("Checking Plot Window")
            print("-" * 70)
            
            if not hasattr(plugin_obj, 'plotWindow') or plugin_obj.plotWindow is None:
                print("✗ Plot window not created")
                print("  Action: The plugin should create this automatically")
                print("  Try reloading the plugin")
                return False
            
            print("✓ Plot window exists")
            
            # Check for pathitems (track paths)
            if not hasattr(plugin_obj, 'pathitems'):
                print("⚠ No 'pathitems' attribute")
                print("  Tracks may not have been plotted yet")
                pathitems_exist = False
            else:
                pathitems_exist = True
                num_pathitems = len(plugin_obj.pathitems) if plugin_obj.pathitems else 0
                print(f"  pathitems count: {num_pathitems}")
                
                if num_pathitems == 0:
                    print("⚠ pathitems list is empty - tracks not drawn")
                else:
                    print(f"✓ {num_pathitems} track paths created")
            
            # Check track display settings
            print("\n" + "-" * 70)
            print("Checking Display Settings")
            print("-" * 70)
            
            # Check if there's a trackPlotOptions
            if hasattr(plugin_obj, 'trackPlotOptions'):
                opts = plugin_obj.trackPlotOptions
                print("✓ Track plot options found")
                
                # Check display track checkbox
                if hasattr(opts, 'displayTrack_checkbox'):
                    is_checked = opts.displayTrack_checkbox.isChecked()
                    print(f"  Display Tracks: {'✓ ENABLED' if is_checked else '✗ DISABLED'}")
                    
                    if not is_checked:
                        print("\n⚠ PROBLEM FOUND: 'Display Tracks' is DISABLED")
                        print("  Action: Enable the 'Display Tracks' checkbox in Track Plot Options")
                        print("\n  To fix:")
                        print("  1. Find 'Track Plot Options' panel")
                        print("  2. Check the 'Display Tracks' checkbox")
                        return False
                
                # Check track width
                if hasattr(opts, 'trackWidth_box'):
                    width = opts.trackWidth_box.value()
                    print(f"  Track Width: {width}")
                    if width == 0:
                        print("  ⚠ Track width is 0 - tracks are invisible!")
                        print("  Action: Set track width to at least 1")
                
                # Check track color
                if hasattr(opts, 'trackColor_selector'):
                    color = opts.trackColor_selector.color()
                    print(f"  Track Color: {color}")
                
                # Check track opacity
                if hasattr(opts, 'trackOpacity_selector'):
                    opacity = opts.trackOpacity_selector.value()
                    print(f"  Track Opacity: {opacity}")
                    if opacity == 0:
                        print("  ⚠ Track opacity is 0 - tracks are invisible!")
                        print("  Action: Set track opacity > 0")
            
            # Check if tracks are being plotted to the right window
            print("\n" + "-" * 70)
            print("Checking Track Rendering")
            print("-" * 70)
            
            if hasattr(plugin_obj.plotWindow, 'imageview'):
                view = plugin_obj.plotWindow.imageview.view
                print("✓ Image view found")
                
                # Check items in the view
                items = view.items()
                print(f"  Total items in view: {len(items)}")
                
                # Count path items
                from qtpy.QtWidgets import QGraphicsPathItem
                path_items = [item for item in items if isinstance(item, QGraphicsPathItem)]
                print(f"  QGraphicsPathItem count: {len(path_items)}")
                
                if len(path_items) == 0:
                    print("  ⚠ No path items in the view - tracks not rendered")
                else:
                    print(f"  ✓ {len(path_items)} path items present")
                    
                    # Check if paths are visible
                    visible_paths = [item for item in path_items if item.isVisible()]
                    print(f"  Visible paths: {len(visible_paths)}")
                    
                    if len(visible_paths) == 0:
                        print("  ⚠ Path items exist but are not visible!")
                        print("  Possible causes:")
                        print("    - Paths are hidden (isVisible() = False)")
                        print("    - Opacity is 0")
                        print("    - Pen width is 0")
            
            # Summary
            print("\n" + "=" * 70)
            print("DIAGNOSTIC SUMMARY")
            print("=" * 70)
            
            issues_found = []
            
            if num_tracks == 0:
                issues_found.append("No tracks in data")
            
            if hasattr(plugin_obj, 'trackPlotOptions'):
                if hasattr(plugin_obj.trackPlotOptions, 'displayTrack_checkbox'):
                    if not plugin_obj.trackPlotOptions.displayTrack_checkbox.isChecked():
                        issues_found.append("'Display Tracks' checkbox is DISABLED")
                
                if hasattr(plugin_obj.trackPlotOptions, 'trackWidth_box'):
                    if plugin_obj.trackPlotOptions.trackWidth_box.value() == 0:
                        issues_found.append("Track width is 0")
                
                if hasattr(plugin_obj.trackPlotOptions, 'trackOpacity_selector'):
                    if plugin_obj.trackPlotOptions.trackOpacity_selector.value() == 0:
                        issues_found.append("Track opacity is 0")
            
            if not pathitems_exist or (pathitems_exist and num_pathitems == 0):
                issues_found.append("Tracks not rendered (no path items)")
            
            if issues_found:
                print("\n⚠ ISSUES FOUND:")
                for i, issue in enumerate(issues_found, 1):
                    print(f"  {i}. {issue}")
                print("\nSee detailed output above for fixes")
            else:
                print("\n✓ No obvious issues found")
                print("Tracks should be displaying...")
                print("\nIf tracks still don't show, there may be a graphics driver issue")
                print("or the tracks are outside the visible area.")
            
            return True
            
        except Exception as e:
            print(f"\n✗ Error during diagnostics: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return False


def quick_fix_suggestions():
    """Print quick fix suggestions."""
    print("\n" + "=" * 70)
    print("QUICK FIX CHECKLIST")
    print("=" * 70)
    print("""
1. Check 'Display Tracks' checkbox
   - Open Track Plot Options panel
   - Make sure 'Display Tracks' is CHECKED

2. Check track width
   - Track Width should be > 0 (try 2-3)
   
3. Check track opacity  
   - Opacity should be > 0 (try 100-255)

4. Check track color
   - Make sure it's not the same as background
   - Try bright colors: red, green, cyan

5. Verify data has tracks
   - Check CSV file has 'track_number' column
   - Tracks should have multiple points

6. Try replotting
   - Uncheck and recheck 'Display Tracks'
   - Or reload the data file

7. Check zoom level
   - Tracks might be outside visible area
   - Try 'View All' or zoom out
""")


if __name__ == "__main__":
    success = diagnose_track_display()
    quick_fix_suggestions()
    
    if not success:
        print("\n" + "=" * 70)
        print("TO RUN THIS FROM WITHIN FLIKA:")
        print("=" * 70)
        print("""
1. Start FLIKA
2. Load your data
3. In FLIKA's Python console, run:
   
   exec(open('diagnose_tracks.py').read())
   
OR save this output and send to george.dickinson@gmail.com
""")
