# tracking_results_plotter/examples.py
"""
Example usage scripts for the Tracking Results Plotter plugin.

This module provides comprehensive examples demonstrating how to use the plugin
programmatically for automated analysis workflows, batch processing, and 
integration with other analysis tools.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Add plugin path for imports
plugin_dir = Path(__file__).parent
sys.path.insert(0, str(plugin_dir))

# Import FLIKA
try:
    import flika
    from flika import global_vars as g
    from flika.window import Window
    from flika.process.file_ import open_file
    FLIKA_AVAILABLE = True
except ImportError:
    print("FLIKA not available. Some examples will not work.")
    FLIKA_AVAILABLE = False

# Import plugin modules
try:
    from utils import generate_sample_data, create_example_csv, ColorManager, ExportManager
    from advanced_plots import AdvancedPlotter
    PLUGIN_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Plugin modules not available: {e}")
    PLUGIN_MODULES_AVAILABLE = False

# Import matplotlib for direct plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

def example_1_basic_data_loading():
    """
    Example 1: Basic data loading and validation
    Demonstrates how to load tracking data and check its validity.
    """
    print("=" * 50)
    print("Example 1: Basic Data Loading and Validation")
    print("=" * 50)
    
    if not PLUGIN_MODULES_AVAILABLE:
        print("Plugin modules not available. Skipping example.")
        return
    
    # Create sample data
    print("Creating sample tracking data...")
    sample_data = generate_sample_data(n_tracks=50, n_frames=200, image_size=(512, 512))
    
    # Save to CSV for demonstration
    sample_file = "example_tracking_data.csv"
    sample_data.to_csv(sample_file, index=False)
    print(f"Sample data saved to: {sample_file}")
    
    # Load and validate data
    from tracking_results_plotter import TrackingDataManager
    
    data_manager = TrackingDataManager()
    success = data_manager.load_data(sample_file)
    
    if success:
        print("✓ Data loaded successfully!")
        print(f"  - Rows: {len(data_manager.data)}")
        print(f"  - Columns: {len(data_manager.data.columns)}")
        print(f"  - Unique tracks: {len(data_manager.data[data_manager.column_mapping['track_id']].unique())}")
        print(f"  - Frame range: {data_manager.data[data_manager.column_mapping['frame']].min()} - {data_manager.data[data_manager.column_mapping['frame']].max()}")
        
        # Get track summary
        summary = data_manager.get_track_summary()
        print(f"  - Track length stats: mean={summary['n_points'].mean():.1f}, std={summary['n_points'].std():.1f}")
    else:
        print("✗ Failed to load data")
    
    # Cleanup
    if os.path.exists(sample_file):
        os.remove(sample_file)
    
    print("Example 1 completed.\n")

def example_2_programmatic_plotting():
    """
    Example 2: Programmatic plotting without GUI
    Shows how to create plots directly from tracking data.
    """
    print("=" * 50)
    print("Example 2: Programmatic Plotting")
    print("=" * 50)
    
    if not (PLUGIN_MODULES_AVAILABLE and MATPLOTLIB_AVAILABLE):
        print("Required modules not available. Skipping example.")
        return
    
    # Generate sample data
    print("Generating sample data for plotting...")
    data = generate_sample_data(n_tracks=10, n_frames=100)
    
    # Create basic plots
    print("Creating basic plots...")
    
    # 1. Track trajectories plot
    plt.figure(figsize=(10, 8))
    
    # Plot first 5 tracks
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    
    for i, track_id in enumerate(data['track_number'].unique()[:5]):
        track_data = data[data['track_number'] == track_id].sort_values('frame')
        plt.plot(track_data['x'], track_data['y'], 
                color=colors[i], label=f'Track {track_id}', linewidth=2, alpha=0.7)
        
        # Mark start and end
        plt.scatter(track_data['x'].iloc[0], track_data['y'].iloc[0], 
                   color=colors[i], marker='o', s=50)
        plt.scatter(track_data['x'].iloc[-1], track_data['y'].iloc[-1], 
                   color=colors[i], marker='s', s=50)
    
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title('Track Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Save plot
    trajectory_plot = "example_trajectories.png"
    plt.savefig(trajectory_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Trajectory plot saved to: {trajectory_plot}")
    
    # 2. Intensity time series
    plt.figure(figsize=(12, 6))
    
    for track_id in data['track_number'].unique()[:3]:
        track_data = data[data['track_number'] == track_id].sort_values('frame')
        plt.plot(track_data['frame'], track_data['intensity'], 
                label=f'Track {track_id}', linewidth=2, alpha=0.7)
    
    plt.xlabel('Frame')
    plt.ylabel('Intensity')
    plt.title('Intensity Time Series')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    intensity_plot = "example_intensity.png"
    plt.savefig(intensity_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Intensity plot saved to: {intensity_plot}")
    
    # 3. Property histogram
    plt.figure(figsize=(10, 6))
    
    # Calculate track lengths
    track_lengths = data.groupby('track_number').size()
    
    plt.hist(track_lengths, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
    plt.axvline(track_lengths.mean(), color='red', linestyle='--', 
               label=f'Mean: {track_lengths.mean():.1f}')
    plt.xlabel('Track Length (frames)')
    plt.ylabel('Frequency')
    plt.title('Track Length Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    histogram_plot = "example_histogram.png"
    plt.savefig(histogram_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Histogram plot saved to: {histogram_plot}")
    
    print("Example 2 completed.\n")

def example_3_advanced_analysis():
    """
    Example 3: Advanced analysis with statistical plots
    Demonstrates advanced plotting capabilities and statistical analysis.
    """
    print("=" * 50)
    print("Example 3: Advanced Analysis")
    print("=" * 50)
    
    if not PLUGIN_MODULES_AVAILABLE:
        print("Plugin modules not available. Skipping example.")
        return
    
    # Generate more complex sample data with multiple conditions
    print("Generating multi-condition sample data...")
    
    conditions = ['Control', 'Treatment_A', 'Treatment_B']
    all_data = []
    
    for condition in conditions:
        # Different parameters for each condition
        if condition == 'Control':
            n_tracks, diffusion = 30, 1.0
        elif condition == 'Treatment_A':
            n_tracks, diffusion = 25, 1.5  # More mobile
        else:
            n_tracks, diffusion = 35, 0.5  # Less mobile
        
        condition_data = generate_sample_data(n_tracks=n_tracks, n_frames=150)
        condition_data['Experiment'] = condition
        
        # Modify velocity based on condition
        condition_data['velocity'] *= diffusion
        
        all_data.append(condition_data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Use advanced plotter
    if MATPLOTLIB_AVAILABLE:
        try:
            plotter = AdvancedPlotter()
            
            # Column mapping
            column_mapping = {
                'track_id': 'track_number',
                'frame': 'frame',
                'x': 'x',
                'y': 'y',
                'intensity': 'intensity'
            }
            
            print("Creating advanced analysis plots...")
            
            # Create comprehensive summary figure
            summary_fig = plotter.create_track_summary_figure(
                combined_data, column_mapping, selected_tracks=None)
            
            summary_plot = "example_advanced_summary.png"
            summary_fig.savefig(summary_plot, dpi=300, bbox_inches='tight')
            plt.close(summary_fig)
            print(f"✓ Advanced summary plot saved to: {summary_plot}")
            
            # Create statistical comparison
            comparison_fig = plotter.create_statistical_comparison_figure(
                combined_data, column_mapping, 'Experiment')
            
            comparison_plot = "example_statistical_comparison.png"
            comparison_fig.savefig(comparison_plot, dpi=300, bbox_inches='tight')
            plt.close(comparison_fig)
            print(f"✓ Statistical comparison saved to: {comparison_plot}")
            
            # Create flower plot grid
            selected_tracks = combined_data['track_number'].unique()[:12]
            flower_fig = plotter.create_flower_plot_grid(
                combined_data, column_mapping, selected_tracks, grid_size=(3, 4))
            
            flower_plot = "example_flower_grid.png"
            flower_fig.savefig(flower_plot, dpi=300, bbox_inches='tight')
            plt.close(flower_fig)
            print(f"✓ Flower plot grid saved to: {flower_plot}")
            
        except Exception as e:
            print(f"Error creating advanced plots: {e}")
    
    # Export data and statistics
    try:
        export_manager = ExportManager()
        
        # Export combined data
        data_file = "example_combined_data.csv"
        export_manager.export_data(combined_data, data_file, include_metadata=True)
        print(f"✓ Data exported to: {data_file}")
        
        # Calculate and export statistics
        stats = {}
        for condition in conditions:
            condition_data = combined_data[combined_data['Experiment'] == condition]
            track_lengths = condition_data.groupby('track_number').size()
            
            stats[condition] = {
                'n_tracks': len(track_lengths),
                'mean_track_length': float(track_lengths.mean()),
                'std_track_length': float(track_lengths.std()),
                'mean_velocity': float(condition_data['velocity'].mean()),
                'std_velocity': float(condition_data['velocity'].std())
            }
        
        stats_file = "example_statistics.json"
        export_manager.export_statistics(stats, stats_file)
        print(f"✓ Statistics exported to: {stats_file}")
        
    except Exception as e:
        print(f"Error exporting data: {e}")
    
    print("Example 3 completed.\n")

def example_4_flika_integration():
    """
    Example 4: Integration with FLIKA
    Shows how to use the plugin with FLIKA windows and data.
    """
    print("=" * 50)
    print("Example 4: FLIKA Integration")
    print("=" * 50)
    
    if not FLIKA_AVAILABLE:
        print("FLIKA not available. Skipping example.")
        return
    
    if not PLUGIN_MODULES_AVAILABLE:
        print("Plugin modules not available. Skipping example.")
        return
    
    try:
        # Start FLIKA if not already running
        if g.m is None:
            flika.start_flika()
        
        print("Creating synthetic image stack...")
        
        # Create a synthetic image stack
        image_stack = np.random.randint(0, 255, (100, 256, 256), dtype=np.uint8)
        
        # Add some "particles" to the images
        for frame in range(100):
            for _ in range(10):  # 10 particles per frame
                x = np.random.randint(20, 236)
                y = np.random.randint(20, 236)
                # Add bright spot
                image_stack[frame, y-2:y+3, x-2:x+3] = 255
        
        # Create FLIKA window
        window = Window(image_stack, name="Synthetic Tracking Data")
        print(f"✓ Created FLIKA window: {window.name}")
        
        # Generate corresponding tracking data
        print("Generating corresponding tracking data...")
        tracking_data = generate_sample_data(n_tracks=20, n_frames=100, image_size=(256, 256))
        
        # Save tracking data
        tracking_file = "flika_example_tracks.csv"
        tracking_data.to_csv(tracking_file, index=False)
        
        print("✓ To use with the plugin:")
        print("  1. The FLIKA window is now open")
        print(f"  2. Load the tracking data from: {tracking_file}")
        print("  3. Launch the plugin: Plugins → Tracking Analysis → Launch Results Plotter")
        print("  4. Set the active window and load the CSV file")
        print("  5. Configure display settings and plot overlays")
        
        # Demonstrate programmatic overlay creation (simplified)
        print("\nCreating simple overlay demonstration...")
        
        from flika.roi import makeROI
        
        # Add a few example ROIs to demonstrate overlay concept
        sample_tracks = tracking_data['track_number'].unique()[:3]
        
        for track_id in sample_tracks:
            track_data = tracking_data[
                tracking_data['track_number'] == track_id
            ].sort_values('frame').head(10)  # First 10 points
            
            for i in range(len(track_data) - 1):
                x1, y1 = track_data.iloc[i][['x', 'y']]
                x2, y2 = track_data.iloc[i+1][['x', 'y']]
                
                try:
                    # Create line ROI between consecutive points
                    line_roi = makeROI('line', 
                                     pts=[[x1, y1], [x2, y2]], 
                                     window=window)
                except Exception as e:
                    print(f"Note: ROI creation requires GUI mode: {e}")
                    break
        
        print("✓ Example track segments added to FLIKA window")
        
    except Exception as e:
        print(f"Error in FLIKA integration example: {e}")
    
    print("Example 4 completed.\n")

def example_5_batch_processing():
    """
    Example 5: Batch processing multiple files
    Demonstrates automated processing of multiple tracking result files.
    """
    print("=" * 50)
    print("Example 5: Batch Processing")
    print("=" * 50)
    
    if not PLUGIN_MODULES_AVAILABLE:
        print("Plugin modules not available. Skipping example.")
        return
    
    # Create multiple sample files
    print("Creating multiple sample tracking files...")
    
    batch_dir = Path("batch_example")
    batch_dir.mkdir(exist_ok=True)
    
    sample_files = []
    conditions = ['Control', 'Drug_1uM', 'Drug_10uM', 'Drug_100uM']
    
    for i, condition in enumerate(conditions):
        # Generate data with different characteristics
        n_tracks = np.random.randint(20, 50)
        data = generate_sample_data(n_tracks=n_tracks, n_frames=150)
        data['Experiment'] = condition
        
        # Modify properties based on condition
        if 'Drug' in condition:
            concentration = float(condition.split('_')[1].replace('uM', ''))
            # Simulate drug effect: higher concentration = lower mobility
            mobility_factor = 1.0 / (1.0 + concentration/50.0)
            data['velocity'] *= mobility_factor
        
        filename = batch_dir / f"tracking_results_{condition}.csv"
        data.to_csv(filename, index=False)
        sample_files.append(filename)
        print(f"  ✓ Created: {filename}")
    
    # Batch processing function
    def process_tracking_file(filepath: Path) -> Dict:
        """Process a single tracking file and return summary statistics."""
        try:
            data = pd.read_csv(filepath)
            
            # Calculate summary statistics
            n_tracks = data['track_number'].nunique()
            n_points = len(data)
            track_lengths = data.groupby('track_number').size()
            
            stats = {
                'filename': filepath.name,
                'condition': data['Experiment'].iloc[0] if 'Experiment' in data.columns else 'Unknown',
                'n_tracks': n_tracks,
                'n_points': n_points,
                'mean_track_length': track_lengths.mean(),
                'std_track_length': track_lengths.std(),
                'mean_velocity': data['velocity'].mean() if 'velocity' in data.columns else None,
                'std_velocity': data['velocity'].std() if 'velocity' in data.columns else None,
                'mean_intensity': data['intensity'].mean() if 'intensity' in data.columns else None
            }
            
            return stats
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            return None
    
    # Process all files
    print("\nProcessing files in batch...")
    batch_results = []
    
    for filepath in sample_files:
        print(f"Processing: {filepath.name}")
        result = process_tracking_file(filepath)
        if result:
            batch_results.append(result)
            print(f"  ✓ Tracks: {result['n_tracks']}, Points: {result['n_points']}")
    
    # Create summary report
    if batch_results:
        summary_df = pd.DataFrame(batch_results)
        
        # Save summary
        summary_file = batch_dir / "batch_processing_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ Batch summary saved to: {summary_file}")
        
        # Display summary statistics
        print("\nBatch Processing Summary:")
        print("-" * 40)
        for _, row in summary_df.iterrows():
            print(f"Condition: {row['condition']}")
            print(f"  Tracks: {row['n_tracks']}")
            print(f"  Mean track length: {row['mean_track_length']:.1f}")
            if row['mean_velocity'] is not None:
                print(f"  Mean velocity: {row['mean_velocity']:.2f}")
            print()
        
        # Create comparison plot if matplotlib available
        if MATPLOTLIB_AVAILABLE:
            plt.figure(figsize=(12, 8))
            
            # Plot 1: Track counts
            plt.subplot(2, 2, 1)
            plt.bar(summary_df['condition'], summary_df['n_tracks'], alpha=0.7, color='skyblue')
            plt.title('Number of Tracks by Condition')
            plt.xlabel('Condition')
            plt.ylabel('Number of Tracks')
            plt.xticks(rotation=45)
            
            # Plot 2: Mean track length
            plt.subplot(2, 2, 2)
            plt.bar(summary_df['condition'], summary_df['mean_track_length'], 
                   alpha=0.7, color='lightgreen')
            plt.title('Mean Track Length by Condition')
            plt.xlabel('Condition')
            plt.ylabel('Mean Track Length (frames)')
            plt.xticks(rotation=45)
            
            # Plot 3: Mean velocity (if available)
            if summary_df['mean_velocity'].notna().any():
                plt.subplot(2, 2, 3)
                valid_data = summary_df.dropna(subset=['mean_velocity'])
                plt.bar(valid_data['condition'], valid_data['mean_velocity'], 
                       alpha=0.7, color='lightcoral')
                plt.title('Mean Velocity by Condition')
                plt.xlabel('Condition')
                plt.ylabel('Mean Velocity')
                plt.xticks(rotation=45)
            
            # Plot 4: Mean intensity (if available)
            if summary_df['mean_intensity'].notna().any():
                plt.subplot(2, 2, 4)
                valid_data = summary_df.dropna(subset=['mean_intensity'])
                plt.bar(valid_data['condition'], valid_data['mean_intensity'], 
                       alpha=0.7, color='gold')
                plt.title('Mean Intensity by Condition')
                plt.xlabel('Condition')
                plt.ylabel('Mean Intensity')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            comparison_plot = batch_dir / "batch_comparison.png"
            plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Comparison plot saved to: {comparison_plot}")
    
    print("Example 5 completed.\n")

def example_6_custom_analysis():
    """
    Example 6: Custom analysis workflow
    Shows how to implement custom analysis functions and integrate them.
    """
    print("=" * 50)
    print("Example 6: Custom Analysis Workflow")
    print("=" * 50)
    
    if not PLUGIN_MODULES_AVAILABLE:
        print("Plugin modules not available. Skipping example.")
        return
    
    # Generate sample data
    data = generate_sample_data(n_tracks=30, n_frames=200)
    
    def calculate_confinement_ratio(track_data: pd.DataFrame) -> float:
        """
        Custom analysis: Calculate confinement ratio.
        Ratio of the actual path length to the diameter of the convex hull.
        """
        x_vals = track_data['x'].values
        y_vals = track_data['y'].values
        
        if len(x_vals) < 3:
            return np.nan
        
        # Calculate path length
        path_length = np.sum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
        
        # Calculate convex hull diameter (simplified as range)
        x_range = np.max(x_vals) - np.min(x_vals)
        y_range = np.max(y_vals) - np.min(y_vals)
        hull_diameter = np.sqrt(x_range**2 + y_range**2)
        
        if hull_diameter == 0:
            return np.nan
        
        return path_length / hull_diameter
    
    def calculate_directional_persistence(track_data: pd.DataFrame) -> float:
        """
        Custom analysis: Calculate directional persistence.
        Measure of how much a track maintains its direction.
        """
        x_vals = track_data['x'].values
        y_vals = track_data['y'].values
        
        if len(x_vals) < 3:
            return np.nan
        
        # Calculate step vectors
        dx = np.diff(x_vals)
        dy = np.diff(y_vals)
        
        # Calculate angles
        angles = np.arctan2(dy, dx)
        
        if len(angles) < 2:
            return np.nan
        
        # Calculate directional persistence as cosine of angle changes
        angle_changes = np.diff(angles)
        
        # Normalize angles to [-π, π]
        angle_changes = np.mod(angle_changes + np.pi, 2*np.pi) - np.pi
        
        # Calculate persistence
        persistence = np.mean(np.cos(angle_changes))
        
        return persistence
    
    def analyze_track_complexity(track_data: pd.DataFrame) -> Dict:
        """
        Custom analysis: Comprehensive track complexity analysis.
        """
        x_vals = track_data['x'].values
        y_vals = track_data['y'].values
        
        if len(x_vals) < 5:
            return {
                'confinement_ratio': np.nan,
                'directional_persistence': np.nan,
                'fractal_dimension_simple': np.nan,
                'exploration_efficiency': np.nan
            }
        
        # Calculate various complexity metrics
        confinement = calculate_confinement_ratio(track_data)
        persistence = calculate_directional_persistence(track_data)
        
        # Simple fractal dimension estimate
        path_length = np.sum(np.sqrt(np.diff(x_vals)**2 + np.diff(y_vals)**2))
        end_to_end = np.sqrt((x_vals[-1] - x_vals[0])**2 + (y_vals[-1] - y_vals[0])**2)
        
        if path_length > 0:
            fractal_dim = np.log(path_length) / np.log(max(end_to_end, 1e-10))
        else:
            fractal_dim = np.nan
        
        # Exploration efficiency
        area_explored = (np.max(x_vals) - np.min(x_vals)) * (np.max(y_vals) - np.min(y_vals))
        exploration_eff = area_explored / max(path_length, 1e-10) if path_length > 0 else np.nan
        
        return {
            'confinement_ratio': confinement,
            'directional_persistence': persistence,
            'fractal_dimension_simple': fractal_dim,
            'exploration_efficiency': exploration_eff
        }
    
    # Apply custom analysis to all tracks
    print("Applying custom analysis to all tracks...")
    
    custom_results = []
    
    for track_id in data['track_number'].unique():
        track_data = data[data['track_number'] == track_id].sort_values('frame')
        
        # Apply custom analysis
        complexity_metrics = analyze_track_complexity(track_data)
        
        result = {
            'track_id': track_id,
            'track_length': len(track_data),
            **complexity_metrics
        }
        
        custom_results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(custom_results)
    
    # Remove tracks with insufficient data
    results_df = results_df.dropna()
    
    print(f"✓ Analyzed {len(results_df)} tracks with custom metrics")
    
    # Display summary statistics
    print("\nCustom Analysis Summary:")
    print("-" * 30)
    for column in ['confinement_ratio', 'directional_persistence', 
                   'fractal_dimension_simple', 'exploration_efficiency']:
        if column in results_df.columns:
            values = results_df[column].dropna()
            if len(values) > 0:
                print(f"{column}:")
                print(f"  Mean: {values.mean():.3f}")
                print(f"  Std:  {values.std():.3f}")
                print(f"  Range: {values.min():.3f} - {values.max():.3f}")
                print()
    
    # Create visualization of custom metrics
    if MATPLOTLIB_AVAILABLE:
        print("Creating custom analysis visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Custom Track Analysis Results', fontsize=16, fontweight='bold')
        
        metrics = ['confinement_ratio', 'directional_persistence', 
                  'fractal_dimension_simple', 'exploration_efficiency']
        titles = ['Confinement Ratio', 'Directional Persistence', 
                 'Fractal Dimension', 'Exploration Efficiency']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if metric in results_df.columns:
                values = results_df[metric].dropna()
                if len(values) > 0:
                    ax.hist(values, bins=20, alpha=0.7, edgecolor='black')
                    ax.axvline(values.mean(), color='red', linestyle='--', 
                              label=f'Mean: {values.mean():.3f}')
                    ax.set_title(title)
                    ax.set_xlabel(metric.replace('_', ' ').title())
                    ax.set_ylabel('Frequency')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', 
                           transform=ax.transAxes)
                    ax.set_title(title)
        
        plt.tight_layout()
        
        custom_plot = "example_custom_analysis.png"
        plt.savefig(custom_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Custom analysis plot saved to: {custom_plot}")
    
    # Export custom results
    try:
        custom_results_file = "example_custom_analysis_results.csv"
        results_df.to_csv(custom_results_file, index=False)
        print(f"✓ Custom analysis results saved to: {custom_results_file}")
    except Exception as e:
        print(f"Error saving custom results: {e}")
    
    print("Example 6 completed.\n")

def run_all_examples():
    """Run all example functions."""
    print("🚀 Running Tracking Results Plotter Examples")
    print("=" * 60)
    print("This script demonstrates various ways to use the plugin programmatically.")
    print("Examples include data loading, plotting, analysis, and FLIKA integration.")
    print("=" * 60)
    print()
    
    examples = [
        example_1_basic_data_loading,
        example_2_programmatic_plotting,
        example_3_advanced_analysis,
        example_4_flika_integration,
        example_5_batch_processing,
        example_6_custom_analysis
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"Error in Example {i}: {e}")
            print()
    
    print("🎉 All examples completed!")
    print()
    print("Generated files:")
    print("- example_*.png: Various plot outputs")
    print("- example_*.csv: Data and results files")
    print("- batch_example/: Batch processing demonstration")
    print()
    print("Next steps:")
    print("1. Start FLIKA and load the plugin")
    print("2. Use the GUI to explore the generated data")
    print("3. Modify these examples for your specific analysis needs")
    print("4. Integrate the plugin into your research workflow")

if __name__ == "__main__":
    # Check if running in interactive mode or script mode
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        example_functions = {
            '1': example_1_basic_data_loading,
            '2': example_2_programmatic_plotting,
            '3': example_3_advanced_analysis,
            '4': example_4_flika_integration,
            '5': example_5_batch_processing,
            '6': example_6_custom_analysis
        }
        
        if example_num in example_functions:
            example_functions[example_num]()
        elif example_num.lower() == 'all':
            run_all_examples()
        else:
            print(f"Unknown example: {example_num}")
            print("Available examples: 1, 2, 3, 4, 5, 6, all")
    else:
        run_all_examples()