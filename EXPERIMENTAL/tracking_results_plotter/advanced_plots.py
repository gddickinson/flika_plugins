# tracking_results_plotter/advanced_plots.py
"""
Advanced plotting functionality for the Tracking Results Plotter plugin.

This module provides sophisticated visualization tools including:
- Multi-panel track analysis plots
- Statistical distribution comparisons
- Correlation matrices and heatmaps
- Publication-ready figure generation
- Interactive plotting widgets
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import matplotlib.gridspec as gridspec
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.widgets import RectangleSelector
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .utils import MathUtils, ColorManager

logger = logging.getLogger(__name__)

class AdvancedPlotter:
    """
    Provides advanced plotting capabilities for particle tracking analysis.
    Creates publication-ready figures with multiple panels and statistical analysis.
    """
    
    def __init__(self, color_manager: Optional[ColorManager] = None):
        self.color_manager = color_manager or ColorManager()
        self.figure_style = {
            'dpi': 300,
            'figsize': (12, 8),
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        
        # Set plotting style
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
            if SEABORN_AVAILABLE:
                sns.set_palette("husl")
    
    def create_track_summary_figure(self, data: pd.DataFrame, 
                                  column_mapping: Dict[str, str],
                                  selected_tracks: Optional[List] = None) -> Figure:
        """
        Create comprehensive track summary figure with multiple analysis panels.
        
        Args:
            data: DataFrame with tracking data
            column_mapping: Column name mappings
            selected_tracks: List of track IDs to include (None for all)
            
        Returns:
            Matplotlib Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for advanced plotting")
        
        # Filter data if specific tracks selected
        if selected_tracks is not None:
            data = data[data[column_mapping['track_id']].isin(selected_tracks)]
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12), dpi=self.figure_style['dpi'])
        gs = gridspec.GridSpec(3, 4, hspace=0.3, wspace=0.3)
        
        # Panel 1: Track trajectories overview
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_trajectory_overview(ax1, data, column_mapping)
        
        # Panel 2: Track length distribution
        ax2 = fig.add_subplot(gs[0, 2])
        self._plot_track_length_distribution(ax2, data, column_mapping)
        
        # Panel 3: Velocity distribution
        ax3 = fig.add_subplot(gs[0, 3])
        self._plot_velocity_distribution(ax3, data, column_mapping)
        
        # Panel 4: Displacement over time
        ax4 = fig.add_subplot(gs[1, :2])
        self._plot_displacement_time_series(ax4, data, column_mapping)
        
        # Panel 5: Radius of gyration analysis
        ax5 = fig.add_subplot(gs[1, 2])
        self._plot_radius_gyration_analysis(ax5, data, column_mapping)
        
        # Panel 6: Turning angle distribution
        ax6 = fig.add_subplot(gs[1, 3])
        self._plot_turning_angle_distribution(ax6, data, column_mapping)
        
        # Panel 7: Correlation matrix
        ax7 = fig.add_subplot(gs[2, :2])
        self._plot_correlation_matrix(ax7, data, column_mapping)
        
        # Panel 8: Property comparison
        ax8 = fig.add_subplot(gs[2, 2:])
        self._plot_property_comparison(ax8, data, column_mapping)
        
        # Add title
        n_tracks = data[column_mapping['track_id']].nunique()
        n_points = len(data)
        fig.suptitle(f'Track Analysis Summary - {n_tracks} tracks, {n_points} points', 
                    fontsize=16, fontweight='bold')
        
        return fig
    
    def _plot_trajectory_overview(self, ax, data: pd.DataFrame, mapping: Dict[str, str]):
        """Plot overview of all trajectories."""
        ax.set_title('Track Trajectories Overview', fontweight='bold')
        
        track_col = mapping['track_id']
        x_col = mapping['x']
        y_col = mapping['y']
        
        # Plot each track with different color
        unique_tracks = data[track_col].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, min(len(unique_tracks), 10)))
        
        for i, track_id in enumerate(unique_tracks[:20]):  # Limit to first 20 for visibility
            track_data = data[data[track_col] == track_id].sort_values(mapping['frame'])
            
            color = colors[i % len(colors)]
            ax.plot(track_data[x_col], track_data[y_col], 
                   color=color, alpha=0.7, linewidth=1, label=f'Track {track_id}')
            
            # Mark start and end points
            ax.scatter(track_data[x_col].iloc[0], track_data[y_col].iloc[0], 
                      color=color, marker='o', s=30, alpha=0.8)
            ax.scatter(track_data[x_col].iloc[-1], track_data[y_col].iloc[-1], 
                      color=color, marker='s', s=30, alpha=0.8)
        
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add legend only if few tracks
        if len(unique_tracks) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    def _plot_track_length_distribution(self, ax, data: pd.DataFrame, mapping: Dict[str, str]):
        """Plot distribution of track lengths."""
        ax.set_title('Track Length Distribution', fontweight='bold')
        
        track_lengths = data.groupby(mapping['track_id']).size()
        
        ax.hist(track_lengths, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax.axvline(track_lengths.mean(), color='red', linestyle='--', 
                  label=f'Mean: {track_lengths.mean():.1f}')
        ax.axvline(track_lengths.median(), color='orange', linestyle='--', 
                  label=f'Median: {track_lengths.median():.1f}')
        
        ax.set_xlabel('Track Length (frames)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_velocity_distribution(self, ax, data: pd.DataFrame, mapping: Dict[str, str]):
        """Plot velocity distribution if velocity data available."""
        ax.set_title('Velocity Distribution', fontweight='bold')
        
        # Check if velocity column exists
        velocity_cols = ['velocity', 'speed', 'meanVelocity']
        velocity_col = None
        
        for col in velocity_cols:
            if col in data.columns:
                velocity_col = col
                break
        
        if velocity_col is not None:
            velocities = data[velocity_col].dropna()
            
            ax.hist(velocities, bins=30, alpha=0.7, edgecolor='black', color='lightgreen')
            ax.axvline(velocities.mean(), color='red', linestyle='--', 
                      label=f'Mean: {velocities.mean():.2f}')
            ax.axvline(velocities.median(), color='orange', linestyle='--', 
                      label=f'Median: {velocities.median():.2f}')
            
            ax.set_xlabel('Velocity (pixels/frame)')
            ax.set_ylabel('Frequency')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Velocity data\nnot available', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_displacement_time_series(self, ax, data: pd.DataFrame, mapping: Dict[str, str]):
        """Plot displacement from origin over time for each track."""
        ax.set_title('Displacement from Origin Over Time', fontweight='bold')
        
        track_col = mapping['track_id']
        frame_col = mapping['frame']
        x_col = mapping['x']
        y_col = mapping['y']
        
        # Calculate displacement from first point for each track
        for track_id in data[track_col].unique()[:10]:  # Limit to first 10 tracks
            track_data = data[data[track_col] == track_id].sort_values(frame_col)
            
            if len(track_data) < 2:
                continue
            
            # Calculate displacement from first point
            first_x, first_y = track_data[x_col].iloc[0], track_data[y_col].iloc[0]
            displacements = np.sqrt((track_data[x_col] - first_x)**2 + 
                                  (track_data[y_col] - first_y)**2)
            
            ax.plot(track_data[frame_col], displacements, alpha=0.7, linewidth=1)
        
        ax.set_xlabel('Frame')
        ax.set_ylabel('Displacement from Origin (pixels)')
        ax.grid(True, alpha=0.3)
    
    def _plot_radius_gyration_analysis(self, ax, data: pd.DataFrame, mapping: Dict[str, str]):
        """Plot radius of gyration analysis."""
        ax.set_title('Radius of Gyration', fontweight='bold')
        
        # Check if radius of gyration column exists
        rg_cols = ['radius_gyration', 'Rg', 'rg', 'radius_of_gyration']
        rg_col = None
        
        for col in rg_cols:
            if col in data.columns:
                rg_col = col
                break
        
        if rg_col is not None:
            rg_values = data[rg_col].dropna()
            
            ax.hist(rg_values, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
            ax.axvline(rg_values.mean(), color='red', linestyle='--', 
                      label=f'Mean: {rg_values.mean():.2f}')
            ax.axvline(rg_values.median(), color='orange', linestyle='--', 
                      label=f'Median: {rg_values.median():.2f}')
            
            ax.set_xlabel('Radius of Gyration (pixels)')
            ax.set_ylabel('Frequency')
            ax.legend()
        else:
            # Calculate radius of gyration for each track
            track_col = mapping['track_id']
            x_col = mapping['x']
            y_col = mapping['y']
            
            rg_values = []
            for track_id in data[track_col].unique():
                track_data = data[data[track_col] == track_id]
                if len(track_data) >= 3:
                    rg = MathUtils.calculate_radius_of_gyration(
                        track_data[x_col].values, track_data[y_col].values)
                    if not np.isnan(rg):
                        rg_values.append(rg)
            
            if rg_values:
                ax.hist(rg_values, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
                ax.axvline(np.mean(rg_values), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(rg_values):.2f}')
                ax.set_xlabel('Radius of Gyration (pixels)')
                ax.set_ylabel('Frequency')
                ax.legend()
            else:
                ax.text(0.5, 0.5, 'Insufficient data\nfor Rg calculation', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_turning_angle_distribution(self, ax, data: pd.DataFrame, mapping: Dict[str, str]):
        """Plot distribution of turning angles."""
        ax.set_title('Turning Angle Distribution', fontweight='bold')
        
        track_col = mapping['track_id']
        x_col = mapping['x']
        y_col = mapping['y']
        frame_col = mapping['frame']
        
        all_angles = []
        
        for track_id in data[track_col].unique():
            track_data = data[data[track_col] == track_id].sort_values(frame_col)
            
            if len(track_data) >= 3:
                angles = MathUtils.calculate_turning_angles(track_data, x_col, y_col)
                all_angles.extend(angles)
        
        if all_angles:
            # Convert to degrees for better interpretation
            angles_deg = np.array(all_angles) * 180 / np.pi
            
            ax.hist(angles_deg, bins=30, alpha=0.7, edgecolor='black', color='gold')
            ax.axvline(0, color='red', linestyle='--', label='Straight motion')
            ax.axvline(np.mean(angles_deg), color='blue', linestyle='--', 
                      label=f'Mean: {np.mean(angles_deg):.1f}°')
            
            ax.set_xlabel('Turning Angle (degrees)')
            ax.set_ylabel('Frequency')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data\nfor turning angle analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_correlation_matrix(self, ax, data: pd.DataFrame, mapping: Dict[str, str]):
        """Plot correlation matrix of numerical properties."""
        ax.set_title('Property Correlation Matrix', fontweight='bold')
        
        # Select numerical columns for correlation analysis
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID and frame columns from correlation
        exclude_cols = [mapping['track_id'], mapping['frame']]
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) >= 2:
            corr_data = data[numeric_cols].corr()
            
            if SEABORN_AVAILABLE:
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0,
                           square=True, ax=ax, cbar_kws={'shrink': 0.8})
            else:
                im = ax.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
                ax.set_xticks(range(len(numeric_cols)))
                ax.set_yticks(range(len(numeric_cols)))
                ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
                ax.set_yticklabels(numeric_cols)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.text(0.5, 0.5, 'Insufficient numerical\ncolumns for correlation', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    def _plot_property_comparison(self, ax, data: pd.DataFrame, mapping: Dict[str, str]):
        """Plot comparison of properties across different conditions."""
        ax.set_title('Property Comparison by Condition', fontweight='bold')
        
        # Check if experiment/condition column exists
        condition_cols = ['Experiment', 'experiment', 'condition', 'sample', 'group']
        condition_col = None
        
        for col in condition_cols:
            if col in data.columns:
                condition_col = col
                break
        
        if condition_col is not None:
            # Find a good numerical property to plot
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = [mapping['track_id'], mapping['frame'], mapping['x'], mapping['y']]
            plot_cols = [col for col in numeric_cols if col not in exclude_cols]
            
            if plot_cols:
                plot_col = plot_cols[0]  # Use first available numerical column
                
                conditions = data[condition_col].unique()
                
                if SEABORN_AVAILABLE:
                    sns.boxplot(data=data, x=condition_col, y=plot_col, ax=ax)
                    ax.tick_params(axis='x', rotation=45)
                else:
                    # Manual box plot
                    box_data = [data[data[condition_col] == cond][plot_col].dropna() 
                               for cond in conditions]
                    ax.boxplot(box_data, labels=conditions)
                    ax.tick_params(axis='x', rotation=45)
                
                ax.set_ylabel(plot_col)
            else:
                ax.text(0.5, 0.5, 'No suitable numerical\nproperties found', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
        else:
            ax.text(0.5, 0.5, 'No condition/experiment\ncolumn found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    def create_flower_plot_grid(self, data: pd.DataFrame, 
                               column_mapping: Dict[str, str],
                               track_ids: List[int],
                               grid_size: Tuple[int, int] = None) -> Figure:
        """
        Create a grid of flower plots for multiple tracks.
        
        Args:
            data: DataFrame with tracking data
            column_mapping: Column name mappings
            track_ids: List of track IDs to plot
            grid_size: Tuple of (rows, cols) for grid layout
            
        Returns:
            Matplotlib Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for flower plots")
        
        n_tracks = len(track_ids)
        
        if grid_size is None:
            # Automatically determine grid size
            cols = int(np.ceil(np.sqrt(n_tracks)))
            rows = int(np.ceil(n_tracks / cols))
        else:
            rows, cols = grid_size
        
        fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows), 
                                subplot_kw={'aspect': 'equal'})
        fig.suptitle('Flower Plot Grid', fontsize=16, fontweight='bold')
        
        if n_tracks == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        track_col = column_mapping['track_id']
        x_col = column_mapping['x']
        y_col = column_mapping['y']
        frame_col = column_mapping['frame']
        
        for i, track_id in enumerate(track_ids):
            if i >= len(axes):
                break
                
            ax = axes[i]
            track_data = data[data[track_col] == track_id].sort_values(frame_col)
            
            if len(track_data) < 2:
                ax.text(0.5, 0.5, f'Track {track_id}\nInsufficient data', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Center the track at origin
            x_vals = track_data[x_col].values - track_data[x_col].iloc[0]
            y_vals = track_data[y_col].values - track_data[y_col].iloc[0]
            
            # Plot trajectory
            ax.plot(x_vals, y_vals, 'b-', alpha=0.7, linewidth=1)
            ax.scatter(x_vals[0], y_vals[0], c='green', s=50, marker='o', label='Start')
            ax.scatter(x_vals[-1], y_vals[-1], c='red', s=50, marker='s', label='End')
            
            # Add points along trajectory
            ax.scatter(x_vals[1:-1], y_vals[1:-1], c='blue', s=10, alpha=0.5)
            
            ax.set_title(f'Track {track_id}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('ΔX (pixels)')
            ax.set_ylabel('ΔY (pixels)')
            
            # Set equal aspect ratio and appropriate limits
            max_range = max(np.ptp(x_vals), np.ptp(y_vals))
            if max_range > 0:
                padding = max_range * 0.1
                ax.set_xlim(np.min(x_vals) - padding, np.max(x_vals) + padding)
                ax.set_ylim(np.min(y_vals) - padding, np.max(y_vals) + padding)
        
        # Hide unused subplots
        for i in range(n_tracks, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def create_temporal_analysis_figure(self, data: pd.DataFrame,
                                      column_mapping: Dict[str, str],
                                      track_id: int) -> Figure:
        """
        Create detailed temporal analysis figure for a single track.
        
        Args:
            data: DataFrame with tracking data
            column_mapping: Column name mappings
            track_id: Specific track ID to analyze
            
        Returns:
            Matplotlib Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for temporal analysis")
        
        track_data = data[data[column_mapping['track_id']] == track_id].sort_values(
            column_mapping['frame'])
        
        if len(track_data) < 3:
            raise ValueError(f"Insufficient data for track {track_id}")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Temporal Analysis - Track {track_id}', fontsize=16, fontweight='bold')
        
        frame_col = column_mapping['frame']
        x_col = column_mapping['x']
        y_col = column_mapping['y']
        
        frames = track_data[frame_col].values
        x_vals = track_data[x_col].values
        y_vals = track_data[y_col].values
        
        # Panel 1: X and Y positions over time
        ax1 = axes[0, 0]
        ax1.plot(frames, x_vals, 'b-', label='X position', linewidth=2)
        ax1.plot(frames, y_vals, 'r-', label='Y position', linewidth=2)
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Position (pixels)')
        ax1.set_title('Position vs Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Instantaneous velocity
        ax2 = axes[0, 1]
        if len(frames) > 1:
            velocities = MathUtils.calculate_track_velocity(track_data, x_col, y_col, frame_col)
            if len(velocities) > 0:
                ax2.plot(frames[1:], velocities, 'g-', linewidth=2)
                ax2.axhline(np.mean(velocities), color='red', linestyle='--', 
                           label=f'Mean: {np.mean(velocities):.2f}')
                ax2.legend()
        
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Velocity (pixels/frame)')
        ax2.set_title('Instantaneous Velocity')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Displacement from origin
        ax3 = axes[1, 0]
        displacements = np.sqrt((x_vals - x_vals[0])**2 + (y_vals - y_vals[0])**2)
        ax3.plot(frames, displacements, 'm-', linewidth=2)
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Displacement from Origin (pixels)')
        ax3.set_title('Displacement from Origin')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Trajectory in space
        ax4 = axes[1, 1]
        ax4.plot(x_vals, y_vals, 'b-', alpha=0.7, linewidth=2)
        ax4.scatter(x_vals[0], y_vals[0], c='green', s=100, marker='o', label='Start')
        ax4.scatter(x_vals[-1], y_vals[-1], c='red', s=100, marker='s', label='End')
        
        # Add arrow to show direction
        if len(x_vals) > 1:
            mid_idx = len(x_vals) // 2
            dx = x_vals[mid_idx+1] - x_vals[mid_idx-1]
            dy = y_vals[mid_idx+1] - y_vals[mid_idx-1]
            ax4.arrow(x_vals[mid_idx], y_vals[mid_idx], dx*0.1, dy*0.1,
                     head_width=2, head_length=2, fc='orange', ec='orange')
        
        ax4.set_xlabel('X Position (pixels)')
        ax4.set_ylabel('Y Position (pixels)')
        ax4.set_title('Spatial Trajectory')
        ax4.set_aspect('equal')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_statistical_comparison_figure(self, data: pd.DataFrame,
                                          column_mapping: Dict[str, str],
                                          group_column: str) -> Figure:
        """
        Create statistical comparison figure across different groups/conditions.
        
        Args:
            data: DataFrame with tracking data
            column_mapping: Column name mappings
            group_column: Column name to group data by
            
        Returns:
            Matplotlib Figure object
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib required for statistical comparison")
        
        if group_column not in data.columns:
            raise ValueError(f"Group column '{group_column}' not found in data")
        
        # Calculate track-level statistics
        track_stats = []
        track_col = column_mapping['track_id']
        x_col = column_mapping['x']
        y_col = column_mapping['y']
        frame_col = column_mapping['frame']
        
        for track_id in data[track_col].unique():
            track_data = data[data[track_col] == track_id].sort_values(frame_col)
            
            if len(track_data) < 3:
                continue
            
            stats_dict = {
                'track_id': track_id,
                'group': track_data[group_column].iloc[0],
                'length': len(track_data),
                'rg': MathUtils.calculate_radius_of_gyration(
                    track_data[x_col].values, track_data[y_col].values),
                'straightness': MathUtils.calculate_track_straightness(
                    track_data, x_col, y_col)
            }
            
            # Add velocity statistics
            velocities = MathUtils.calculate_track_velocity(track_data, x_col, y_col, frame_col)
            if len(velocities) > 0:
                stats_dict['mean_velocity'] = np.mean(velocities)
                stats_dict['velocity_std'] = np.std(velocities)
            
            track_stats.append(stats_dict)
        
        stats_df = pd.DataFrame(track_stats)
        
        if len(stats_df) == 0:
            raise ValueError("No tracks with sufficient data for analysis")
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Statistical Comparison by {group_column}', 
                    fontsize=16, fontweight='bold')
        
        properties = ['length', 'rg', 'straightness', 'mean_velocity']
        
        for i, prop in enumerate(properties):
            if i >= 4:  # Only plot first 4 properties
                break
                
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            if prop in stats_df.columns and not stats_df[prop].isna().all():
                if SEABORN_AVAILABLE:
                    sns.boxplot(data=stats_df, x='group', y=prop, ax=ax)
                else:
                    groups = stats_df['group'].unique()
                    box_data = [stats_df[stats_df['group'] == group][prop].dropna() 
                               for group in groups]
                    ax.boxplot(box_data, labels=groups)
                
                ax.set_title(f'{prop.replace("_", " ").title()}')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{prop}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Statistical summary table
        ax_table = axes[1, 2]
        ax_table.axis('off')
        
        summary_stats = []
        for group in stats_df['group'].unique():
            group_data = stats_df[stats_df['group'] == group]
            summary_stats.append([
                group,
                len(group_data),
                f"{group_data['length'].mean():.1f}",
                f"{group_data['rg'].mean():.2f}" if 'rg' in group_data.columns else 'N/A'
            ])
        
        table = ax_table.table(cellText=summary_stats,
                              colLabels=['Group', 'N Tracks', 'Mean Length', 'Mean Rg'],
                              cellLoc='center',
                              loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax_table.set_title('Summary Statistics')
        
        plt.tight_layout()
        return fig


# Create global instance
advanced_plotter = AdvancedPlotter()