#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example Usage of Calcium Event Detector FLIKA Plugin
=====================================================

This script demonstrates how to use the ca_event_detector FLIKA plugin
both through the GUI and programmatically.

Author: George Stuyt
Date: 2024-12-25
"""

import numpy as np
from pathlib import Path

# Start FLIKA
import flika
flika.start_flika()

from flika import global_vars as g
from flika.process.file_ import open_file

# ============================================================================
# Example 1: Using the GUI
# ============================================================================

print("Example 1: Using the GUI")
print("-" * 60)

# Load your data
window = open_file('path/to/your/calcium_data.tif')

# Option A: Quick Detection
# Go to: Plugins → Calcium Event Detector → Quick Detection
# Select your model file and click Run

# Option B: Full Detection with parameters
# Go to: Plugins → Calcium Event Detector → Run Detection
# Configure all parameters and click Run

print("Use the FLIKA GUI to run detection (see menu above)")
print()


# ============================================================================
# Example 2: Programmatic Detection
# ============================================================================

print("Example 2: Programmatic Detection")
print("-" * 60)

# Import the detector
from ca_event_detector.inference.detect import CalciumEventDetector

# Set paths
model_path = 'path/to/your/trained_model.pth'
data_path = 'path/to/your/calcium_data.tif'

# Load data
from tifffile import imread
image = imread(data_path)
print(f"Loaded image: shape={image.shape}, dtype={image.dtype}")

# Create detector
detector = CalciumEventDetector(model_path)

# Run detection
print("Running detection...")
results = detector.detect(image)

# Extract results
class_mask = results['class_mask']
instance_mask = results['instance_mask']
probabilities = results['probabilities']

# Print statistics
n_sparks = np.sum(class_mask == 1)
n_puffs = np.sum(class_mask == 2)
n_waves = np.sum(class_mask == 3)
n_events = instance_mask.max()

print(f"\nDetection Results:")
print(f"  Ca²⁺ sparks: {n_sparks:,} pixels")
print(f"  Ca²⁺ puffs: {n_puffs:,} pixels")
print(f"  Ca²⁺ waves: {n_waves:,} pixels")
print(f"  Total events: {n_events}")
print()


# ============================================================================
# Example 3: Accessing Results from Window
# ============================================================================

print("Example 3: Accessing Results from Window")
print("-" * 60)

# After running detection via GUI, results are stored in window
if hasattr(window, 'ca_event_results'):
    results = window.ca_event_results
    
    # Access individual components
    class_mask = results['class_mask']
    instance_mask = results['instance_mask']
    
    # Get event statistics
    if hasattr(window, 'ca_event_display'):
        stats = window.ca_event_display.get_event_stats()
        print(f"Event statistics: {stats}")
    
    # Extract individual event properties
    for event_id in range(1, min(6, instance_mask.max() + 1)):  # First 5 events
        mask = instance_mask == event_id
        if not np.any(mask):
            continue
        
        # Get event class
        event_class_id = int(class_mask[mask][0])
        class_names = ['background', 'spark', 'puff', 'wave']
        event_class = class_names[event_class_id]
        
        # Get event size
        event_size = np.sum(mask)
        
        # Get temporal extent
        coords = np.argwhere(mask)
        t_start = coords[:, 0].min()
        t_end = coords[:, 0].max()
        duration = t_end - t_start + 1
        
        print(f"Event {event_id}: {event_class}, size={event_size}, duration={duration} frames")

print()


# ============================================================================
# Example 4: Batch Processing Multiple Files
# ============================================================================

print("Example 4: Batch Processing")
print("-" * 60)

def batch_process_directory(input_dir, output_dir, model_path):
    """
    Process all TIFF files in a directory.
    
    Parameters
    ----------
    input_dir : str or Path
        Directory containing TIFF files
    output_dir : str or Path
        Directory to save results
    model_path : str or Path
        Path to trained model
    """
    from tifffile import imread, imwrite
    import pandas as pd
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create detector
    detector = CalciumEventDetector(model_path)
    
    # Process each file
    for tiff_file in input_dir.glob('*.tif'):
        print(f"\nProcessing: {tiff_file.name}")
        
        # Load
        image = imread(tiff_file)
        
        # Detect
        results = detector.detect(image)
        
        # Save masks
        prefix = tiff_file.stem
        class_path = output_dir / f"{prefix}_class.tif"
        instance_path = output_dir / f"{prefix}_instance.tif"
        
        imwrite(class_path, results['class_mask'].astype(np.uint8))
        imwrite(instance_path, results['instance_mask'].astype(np.uint16))
        
        # Extract and save event properties
        events = []
        class_names = ['background', 'spark', 'puff', 'wave']
        
        for event_id in range(1, results['instance_mask'].max() + 1):
            mask = results['instance_mask'] == event_id
            if not np.any(mask):
                continue
            
            event_class_id = int(results['class_mask'][mask][0])
            event_class = class_names[event_class_id]
            
            coords = np.argwhere(mask)
            
            events.append({
                'file': tiff_file.name,
                'event_id': event_id,
                'class': event_class,
                'size': np.sum(mask),
                't_center': coords[:, 0].mean(),
                'y_center': coords[:, 1].mean(),
                'x_center': coords[:, 2].mean(),
                'duration': coords[:, 0].max() - coords[:, 0].min() + 1
            })
        
        # Save CSV
        csv_path = output_dir / f"{prefix}_events.csv"
        pd.DataFrame(events).to_csv(csv_path, index=False)
        
        print(f"  Detected {len(events)} events")
        print(f"  Saved: {class_path.name}, {instance_path.name}, {csv_path.name}")

# Example usage:
# batch_process_directory(
#     input_dir='data/raw_videos',
#     output_dir='data/processed',
#     model_path='models/best_model.pth'
# )

print("Batch processing function defined (see code for usage)")
print()


# ============================================================================
# Example 5: Visualizing Results
# ============================================================================

print("Example 5: Visualizing Results")
print("-" * 60)

def visualize_detection_results(window, results):
    """
    Create visualization of detection results on FLIKA window.
    
    Parameters
    ----------
    window : Window
        FLIKA window
    results : dict
        Detection results
    """
    from ca_event_detector_flika.event_display import EventDisplay
    
    # Create display
    display = EventDisplay(window)
    
    # Set events
    display.set_events(results['class_mask'], results['instance_mask'])
    
    # Show class overlay
    display.show_overlay('class')
    
    # Get statistics
    stats = display.get_event_stats()
    print(f"\nVisualization created:")
    print(f"  Sparks: {stats['n_sparks']}")
    print(f"  Puffs: {stats['n_puffs']}")
    print(f"  Waves: {stats['n_waves']}")
    print(f"  Total instances: {stats['n_total_instances']}")
    
    return display

# Usage:
# display = visualize_detection_results(window, results)

print("Visualization function defined (see code for usage)")
print()


# ============================================================================
# Example 6: Opening Results Viewer
# ============================================================================

print("Example 6: Results Viewer")
print("-" * 60)

def open_results_viewer(results):
    """
    Open interactive results viewer.
    
    Parameters
    ----------
    results : dict
        Detection results
    """
    from ca_event_detector_flika.event_results_viewer import EventResultsViewer
    
    viewer = EventResultsViewer()
    viewer.set_data(results)
    viewer.show()
    
    return viewer

# Usage:
# viewer = open_results_viewer(results)

print("Results viewer function defined (see code for usage)")
print()


# ============================================================================
# Example 7: Exporting Results
# ============================================================================

print("Example 7: Exporting Results")
print("-" * 60)

def export_all_results(results, output_dir, prefix):
    """
    Export all detection results to files.
    
    Parameters
    ----------
    results : dict
        Detection results
    output_dir : str or Path
        Output directory
    prefix : str
        File name prefix
    """
    from tifffile import imwrite
    import pandas as pd
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save masks
    class_path = output_dir / f"{prefix}_class.tif"
    instance_path = output_dir / f"{prefix}_instance.tif"
    prob_path = output_dir / f"{prefix}_probabilities.tif"
    
    imwrite(class_path, results['class_mask'].astype(np.uint8))
    imwrite(instance_path, results['instance_mask'].astype(np.uint16))
    imwrite(prob_path, results['probabilities'].astype(np.float32))
    
    # Extract event properties
    events = []
    class_names = ['background', 'spark', 'puff', 'wave']
    
    for event_id in range(1, results['instance_mask'].max() + 1):
        mask = results['instance_mask'] == event_id
        if not np.any(mask):
            continue
        
        event_class_id = int(results['class_mask'][mask][0])
        event_class = class_names[event_class_id]
        
        coords = np.argwhere(mask)
        
        events.append({
            'event_id': event_id,
            'class': event_class,
            'class_id': event_class_id,
            'size_pixels': np.sum(mask),
            't_center': coords[:, 0].mean(),
            'y_center': coords[:, 1].mean(),
            'x_center': coords[:, 2].mean(),
            'frame_start': int(coords[:, 0].min()),
            'frame_end': int(coords[:, 0].max()),
            'duration': int(coords[:, 0].max() - coords[:, 0].min() + 1)
        })
    
    # Save CSV
    csv_path = output_dir / f"{prefix}_events.csv"
    pd.DataFrame(events).to_csv(csv_path, index=False)
    
    print(f"\nExported results to {output_dir}:")
    print(f"  {class_path.name}")
    print(f"  {instance_path.name}")
    print(f"  {prob_path.name}")
    print(f"  {csv_path.name}")

# Usage:
# export_all_results(results, 'output', 'experiment_001')

print("Export function defined (see code for usage)")
print()


print("=" * 60)
print("Examples complete!")
print("=" * 60)
