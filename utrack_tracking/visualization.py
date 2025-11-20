#!/usr/bin/env python3
"""
Visualization tools for particle tracking results

Copyright (C) 2025, Danuser Lab - UTSouthwestern 
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class TrackVisualizer:
    """Visualization tools for tracking results"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = plt.cm.tab20(np.linspace(0, 1, 20))
    
    def plot_tracks_overview(self, tracks_final: List[Dict], save_path: Optional[str] = None,
                           show_start_end: bool = True, show_gaps: bool = True,
                           max_tracks: int = 50) -> plt.Figure:
        """
        Create overview plot of all tracks
        
        Args:
            tracks_final: List of track dictionaries
            save_path: Path to save figure (optional)
            show_start_end: Whether to mark track start/end points
            show_gaps: Whether to show gaps in tracks
            max_tracks: Maximum number of tracks to display
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        if not tracks_final:
            ax.text(0.5, 0.5, 'No tracks to display', ha='center', va='center',
                   transform=ax.transAxes, fontsize=16)
            return fig
        
        tracks_to_plot = tracks_final[:max_tracks]
        
        for i, track in enumerate(tracks_to_plot):
            color = self.colors[i % len(self.colors)]
            
            # Extract track segments
            segments = self._extract_track_segments(track)
            
            for segment in segments:
                if len(segment) < 2:
                    continue
                
                x_coords = segment[:, 0]
                y_coords = segment[:, 1]
                
                # Plot track
                ax.plot(x_coords, y_coords, 'o-', color=color, alpha=0.7,
                       markersize=3, linewidth=1.5, label=f'Track {i+1}' if i < 10 else '')
                
                if show_start_end:
                    # Mark start point (circle)
                    ax.plot(x_coords[0], y_coords[0], 'o', color=color, markersize=8,
                           markeredgecolor='black', markeredgewidth=2)
                    # Mark end point (square)
                    ax.plot(x_coords[-1], y_coords[-1], 's', color=color, markersize=8,
                           markeredgecolor='black', markeredgewidth=2)
            
            # Show gaps if requested
            if show_gaps and len(segments) > 1:
                for j in range(len(segments) - 1):
                    end_point = segments[j][-1]
                    start_point = segments[j+1][0]
                    ax.plot([end_point[0], start_point[0]], [end_point[1], start_point[1]],
                           '--', color=color, alpha=0.5, linewidth=2)
        
        ax.set_xlabel('X position (pixels)')
        ax.set_ylabel('Y position (pixels)')
        ax.set_title(f'Particle Tracking Results ({len(tracks_to_plot)} tracks shown)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        if len(tracks_to_plot) <= 10:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Track overview saved to {save_path}")
        
        return fig
    
    def plot_track_statistics(self, tracks_final: List[Dict], save_path: Optional[str] = None) -> plt.Figure:
        """Plot track statistics and distributions"""
        from track_analysis import analyze_tracking_results
        
        # Analyze tracks
        analysis = analyze_tracking_results(tracks_final)
        
        if 'error' in analysis:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f'Analysis error: {analysis["error"]}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Track length distribution
        if analysis['track_lengths']:
            ax1.hist(analysis['track_lengths'], bins=20, alpha=0.7, edgecolor='black')
            ax1.axvline(analysis['mean_track_length'], color='red', linestyle='--', 
                       label=f'Mean: {analysis["mean_track_length"]:.1f}')
            ax1.set_xlabel('Track Length (frames)')
            ax1.set_ylabel('Count')
            ax1.set_title('Track Length Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Speed distribution
        if analysis['mean_speeds']:
            ax2.hist(analysis['mean_speeds'], bins=20, alpha=0.7, edgecolor='black')
            ax2.axvline(analysis['mean_speed'], color='red', linestyle='--',
                       label=f'Mean: {analysis["mean_speed"]:.2f}')
            ax2.set_xlabel('Mean Speed (pixels/frame)')
            ax2.set_ylabel('Count')
            ax2.set_title('Speed Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Directionality distribution
        if analysis['directionalities']:
            ax3.hist(analysis['directionalities'], bins=20, alpha=0.7, edgecolor='black')
            ax3.axvline(analysis['mean_directionality'], color='red', linestyle='--',
                       label=f'Mean: {analysis["mean_directionality"]:.2f}')
            ax3.set_xlabel('Directionality')
            ax3.set_ylabel('Count')
            ax3.set_title('Directionality Distribution')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Motion type pie chart
        if analysis['motion_type_counts']:
            motion_types = list(analysis['motion_type_counts'].keys())
            counts = list(analysis['motion_type_counts'].values())
            ax4.pie(counts, labels=motion_types, autopct='%1.1f%%', startangle=90)
            ax4.set_title('Motion Type Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Track statistics saved to {save_path}")
        
        return fig
    
    def plot_individual_track(self, track: Dict, track_id: int = 0, 
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot detailed view of individual track"""
        from track_analysis import MotionAnalyzer
        
        motion_analyzer = MotionAnalyzer()
        analysis = motion_analyzer.analyze_track_motion(track)
        
        if 'error' in analysis:
            fig, ax = plt.subplots(figsize=self.figsize)
            ax.text(0.5, 0.5, f'Analysis error: {analysis["error"]}', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        coordinates = analysis['coordinates']
        displacements = analysis['displacements']
        
        # 1. Track trajectory
        ax1.plot(coordinates[:, 0], coordinates[:, 1], 'o-', alpha=0.7, linewidth=2)
        ax1.plot(coordinates[0, 0], coordinates[0, 1], 'go', markersize=10, label='Start')
        ax1.plot(coordinates[-1, 0], coordinates[-1, 1], 'ro', markersize=10, label='End')
        ax1.set_xlabel('X position (pixels)')
        ax1.set_ylabel('Y position (pixels)')
        ax1.set_title(f'Track {track_id} Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # 2. Speed over time
        if len(displacements) > 0:
            displacement_mags = np.sqrt(np.sum(displacements**2, axis=1))
            frames = np.arange(1, len(displacement_mags) + 1)
            ax2.plot(frames, displacement_mags, 'o-', alpha=0.7)
            ax2.axhline(analysis['mean_speed'], color='red', linestyle='--', 
                       label=f'Mean: {analysis["mean_speed"]:.2f}')
            ax2.set_xlabel('Frame')
            ax2.set_ylabel('Speed (pixels/frame)')
            ax2.set_title('Speed over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Mean Squared Displacement
        if 'msd_results' in analysis and 'lags' in analysis['msd_results']:
            msd_data = analysis['msd_results']
            ax3.loglog(msd_data['lags'], msd_data['msd_values'], 'o-', alpha=0.7)
            
            if not np.isnan(msd_data.get('alpha', np.nan)):
                # Plot fit line
                fit_msd = msd_data['diffusion_coeff'] * (msd_data['lags'] ** msd_data['alpha'])
                ax3.loglog(msd_data['lags'], fit_msd, '--', 
                          label=f'α = {msd_data["alpha"]:.2f}')
            
            ax3.set_xlabel('Time lag')
            ax3.set_ylabel('MSD')
            ax3.set_title('Mean Squared Displacement')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Track information text
        ax4.axis('off')
        info_text = f"""
Track {track_id} Information:
• Length: {analysis['track_length']} frames
• Motion Type: {analysis['motion_type']}
• Mean Speed: {analysis['mean_speed']:.2f} pixels/frame
• Max Speed: {analysis['max_speed']:.2f} pixels/frame
• Directionality: {analysis['directionality']:.3f}
• Net Distance: {analysis['net_distance']:.1f} pixels
• Total Distance: {analysis['total_distance']:.1f} pixels
        """
        
        if 'msd_results' in analysis:
            msd = analysis['msd_results']
            if not np.isnan(msd.get('alpha', np.nan)):
                info_text += f"• MSD Exponent (α): {msd['alpha']:.2f}\n"
                info_text += f"• Diffusion Coeff: {msd['diffusion_coeff']:.2f}\n"
        
        ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Individual track plot saved to {save_path}")
        
        return fig
    
    def create_track_animation(self, tracks_final: List[Dict], save_path: str,
                             fps: int = 5, show_trails: bool = True,
                             trail_length: int = 10) -> None:
        """Create animated visualization of tracks over time"""
        try:
            # Extract all frame numbers
            all_frames = set()
            track_data = []
            
            for track in tracks_final:
                seq_events = track.get('seq_of_events', [])
                if len(seq_events) >= 2:
                    start_frame = int(seq_events[0][0])
                    end_frame = int(seq_events[-1][0])
                    
                    # Extract coordinates
                    segments = self._extract_track_segments(track)
                    if segments:
                        coordinates = segments[0]  # Take first segment for simplicity
                        frames = list(range(start_frame, start_frame + len(coordinates)))
                        all_frames.update(frames)
                        
                        track_data.append({
                            'coordinates': coordinates,
                            'frames': frames,
                            'start_frame': start_frame
                        })
            
            if not all_frames:
                logger.warning("No frame data found for animation")
                return
            
            frame_range = sorted(all_frames)
            
            # Set up the figure and axis
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Find axis limits
            all_coords = np.vstack([td['coordinates'] for td in track_data if len(td['coordinates']) > 0])
            if len(all_coords) > 0:
                x_margin = (all_coords[:, 0].max() - all_coords[:, 0].min()) * 0.1
                y_margin = (all_coords[:, 1].max() - all_coords[:, 1].min()) * 0.1
                ax.set_xlim(all_coords[:, 0].min() - x_margin, all_coords[:, 0].max() + x_margin)
                ax.set_ylim(all_coords[:, 1].min() - y_margin, all_coords[:, 1].max() + y_margin)
            
            ax.set_xlabel('X position (pixels)')
            ax.set_ylabel('Y position (pixels)')
            ax.set_aspect('equal')
            
            # Initialize plots
            track_lines = []
            track_points = []
            
            for i, _ in enumerate(track_data):
                color = self.colors[i % len(self.colors)]
                line, = ax.plot([], [], '-', color=color, alpha=0.5, linewidth=1)
                point, = ax.plot([], [], 'o', color=color, markersize=6)
                track_lines.append(line)
                track_points.append(point)
            
            def animate(frame_idx):
                current_frame = frame_range[frame_idx]
                ax.set_title(f'Particle Tracking Animation - Frame {current_frame}')
                
                for i, td in enumerate(track_data):
                    if current_frame in td['frames']:
                        coord_idx = td['frames'].index(current_frame)
                        current_pos = td['coordinates'][coord_idx]
                        
                        # Update current position
                        track_points[i].set_data([current_pos[0]], [current_pos[1]])
                        
                        # Update trail
                        if show_trails:
                            start_idx = max(0, coord_idx - trail_length)
                            trail_coords = td['coordinates'][start_idx:coord_idx+1]
                            if len(trail_coords) > 1:
                                track_lines[i].set_data(trail_coords[:, 0], trail_coords[:, 1])
                            else:
                                track_lines[i].set_data([], [])
                    else:
                        # Hide if not active in this frame
                        track_points[i].set_data([], [])
                        if not show_trails:
                            track_lines[i].set_data([], [])
                
                return track_lines + track_points
            
            # Create animation
            anim = FuncAnimation(fig, animate, frames=len(frame_range),
                               interval=1000//fps, blit=True, repeat=True)
            
            # Save animation
            anim.save(save_path, writer='pillow', fps=fps)
            logger.info(f"Track animation saved to {save_path}")
            
            plt.close(fig)
            
        except Exception as e:
            logger.error(f"Error creating track animation: {str(e)}")
    
    def _extract_track_segments(self, track: Dict) -> List[np.ndarray]:
        """Extract coordinate segments from track"""
        coords_amp = track.get('tracks_coord_amp_cg', np.array([]))
        
        if coords_amp.size == 0:
            return []
        
        # Handle multiple segments
        if coords_amp.ndim > 1:
            segments = []
            for segment_idx in range(coords_amp.shape[0]):
                segment_coords = self._extract_coords_from_row(coords_amp[segment_idx, :])
                if len(segment_coords) > 0:
                    segments.append(segment_coords)
            return segments
        else:
            coords = self._extract_coords_from_row(coords_amp)
            return [coords] if len(coords) > 0 else []
    
    def _extract_coords_from_row(self, coords_amp_row: np.ndarray) -> np.ndarray:
        """Extract coordinates from single track row"""
        num_frames = len(coords_amp_row) // 8
        coordinates = []
        
        for i in range(num_frames):
            coord_idx = i * 8
            x = coords_amp_row[coord_idx]
            y = coords_amp_row[coord_idx + 1]
            
            if not np.isnan(x) and not np.isnan(y):
                coordinates.append([x, y])
        
        return np.array(coordinates) if coordinates else np.array([]).reshape(0, 2)


def plot_detection_density(movie_info: List[Dict], save_path: Optional[str] = None) -> plt.Figure:
    """Plot detection density heatmap"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Collect all detections
    all_x = []
    all_y = []
    
    for frame_info in movie_info:
        if frame_info['num'] > 0:
            x_coords = frame_info['x_coord'][:, 0] if 'x_coord' in frame_info else frame_info['all_coord'][:, 0]
            y_coords = frame_info['y_coord'][:, 0] if 'y_coord' in frame_info else frame_info['all_coord'][:, 2]
            all_x.extend(x_coords)
            all_y.extend(y_coords)
    
    if all_x and all_y:
        # Create 2D histogram
        h, xedges, yedges = np.histogram2d(all_x, all_y, bins=50)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        im = ax.imshow(h.T, extent=extent, origin='lower', cmap='viridis', alpha=0.8)
        plt.colorbar(im, ax=ax, label='Detection count')
        
        ax.set_xlabel('X position (pixels)')
        ax.set_ylabel('Y position (pixels)')
        ax.set_title('Detection Density Heatmap')
    else:
        ax.text(0.5, 0.5, 'No detections to plot', ha='center', va='center',
               transform=ax.transAxes, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Detection density plot saved to {save_path}")
    
    return fig
