#!/usr/bin/env python3
"""
Utility functions for particle tracking

Python port of various u-track utility functions

Copyright (C) 2025, Danuser Lab - UTSouthwestern 
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


def lap(cost_matrix: Union[np.ndarray, sp.spmatrix], nonlink_marker: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linear Assignment Problem solver
    
    Args:
        cost_matrix: Cost matrix for assignment
        nonlink_marker: Value indicating disallowed links
        
    Returns:
        Tuple of (link12, link21) assignment arrays
    """
    try:
        # Convert sparse matrix to dense if needed
        if sp.issparse(cost_matrix):
            cost_dense = cost_matrix.toarray()
        else:
            cost_dense = cost_matrix.copy()
        
        # Replace nonlink markers with very large values
        cost_dense[cost_dense == nonlink_marker] = 1e10
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_dense)
        
        # Create assignment arrays (1-indexed like MATLAB)
        num_rows, num_cols = cost_dense.shape
        link12 = np.zeros(num_rows, dtype=int)  # Assignment from row to column
        link21 = np.zeros(num_cols, dtype=int)  # Assignment from column to row
        
        # Fill in assignments
        for i, (row, col) in enumerate(zip(row_ind, col_ind)):
            if cost_matrix[row, col] != nonlink_marker:
                link12[row] = col + 1  # 1-indexed
                link21[col] = row + 1  # 1-indexed
        
        return link12, link21
        
    except Exception as e:
        logger.error(f"Error in LAP solver: {str(e)}")
        # Return empty assignments
        num_rows = cost_matrix.shape[0] if hasattr(cost_matrix, 'shape') else 0
        num_cols = cost_matrix.shape[1] if hasattr(cost_matrix, 'shape') and len(cost_matrix.shape) > 1 else 0
        return np.zeros(num_rows, dtype=int), np.zeros(num_cols, dtype=int)


def progress_text(progress: float, message: str = "Progress"):
    """
    Display progress information
    
    Args:
        progress: Progress value (0 to 1, or special values)
        message: Progress message
    """
    if progress == 0:
        logger.info(f"Starting: {message}")
    elif progress == 1:
        logger.info(f"Completed: {message}")
    else:
        logger.info(f"{message}: {progress:.1%}")


def get_track_sel(tracks_coord_amp: np.ndarray) -> np.ndarray:
    """
    Get track start times, end times, and lengths
    
    Args:
        tracks_coord_amp: Track coordinate and amplitude matrix
        
    Returns:
        Array with columns [start_time, end_time, length]
    """
    num_tracks, num_cols = tracks_coord_amp.shape
    num_frames = num_cols // 8
    
    track_sel = np.zeros((num_tracks, 3))
    
    for i_track in range(num_tracks):
        # Get x-coordinates (every 8th column starting from 0)
        x_coords = tracks_coord_amp[i_track, ::8]
        
        # Handle sparse matrices
        if sp.issparse(tracks_coord_amp):
            x_coords = x_coords.toarray().flatten()
        
        # Find valid (non-zero, non-NaN) frames
        if sp.issparse(tracks_coord_amp):
            valid_frames = x_coords != 0
        else:
            valid_frames = ~np.isnan(x_coords) & (x_coords != 0)
        
        if np.any(valid_frames):
            valid_indices = np.where(valid_frames)[0]
            track_sel[i_track, 0] = valid_indices[0] + 1      # Start time (1-indexed)
            track_sel[i_track, 1] = valid_indices[-1] + 1     # End time (1-indexed)
            track_sel[i_track, 2] = len(valid_indices)        # Length
        else:
            # Empty track
            track_sel[i_track, :] = [1, 1, 0]
    
    return track_sel


def find_track_gaps(tracks_final: List[Dict]) -> np.ndarray:
    """
    Find gaps in tracks
    
    Args:
        tracks_final: List of final track structures
        
    Returns:
        Array with gap information [track_id, start_frame, end_frame, gap_length]
    """
    gap_info = []
    
    for track_id, track in enumerate(tracks_final):
        seq_events = track.get('seq_of_events', [])
        if len(seq_events) < 2:
            continue
        
        # Look for gaps in sequence of events
        start_events = seq_events[seq_events[:, 1] == 1]  # Start events
        end_events = seq_events[seq_events[:, 1] == 2]    # End events
        
        # Check for gaps between segments
        for i in range(len(end_events) - 1):
            end_frame = end_events[i, 0]
            next_start_frame = start_events[i + 1, 0] if i + 1 < len(start_events) else None
            
            if next_start_frame is not None and next_start_frame > end_frame + 1:
                gap_length = next_start_frame - end_frame - 1
                gap_info.append([track_id + 1, end_frame + 1, next_start_frame - 1, gap_length])
    
    return np.array(gap_info) if gap_info else np.array([]).reshape(0, 4)


def convert_mat_2_struct(tracks_coord_amp: np.ndarray, tracks_feat_indx: np.ndarray) -> List[Dict]:
    """
    Convert track matrices to structure format
    
    Args:
        tracks_coord_amp: Track coordinate and amplitude matrix
        tracks_feat_indx: Track feature index matrix
        
    Returns:
        List of track structures
    """
    if tracks_coord_amp.size == 0:
        return []
    
    tracks_final = []
    track_sel = get_track_sel(tracks_coord_amp)
    
    for i_track in range(tracks_coord_amp.shape[0]):
        start_time = int(track_sel[i_track, 0])
        end_time = int(track_sel[i_track, 1])
        
        if start_time == 0 or end_time == 0:
            continue
        
        # Extract track data
        tracks_feat_indx_cg = tracks_feat_indx[i_track:i_track+1, start_time-1:end_time]
        tracks_coord_amp_cg = tracks_coord_amp[i_track:i_track+1, (start_time-1)*8:end_time*8]
        
        # Handle sparse matrices
        if sp.issparse(tracks_coord_amp_cg):
            tracks_coord_amp_cg = tracks_coord_amp_cg.toarray()
        
        # Create sequence of events
        seq_of_events = np.array([
            [start_time, 1, 1, np.nan],  # Start event
            [end_time, 2, 1, np.nan]     # End event
        ])
        
        track_final = {
            'tracks_feat_indx_cg': tracks_feat_indx_cg,
            'tracks_coord_amp_cg': tracks_coord_amp_cg,
            'seq_of_events': seq_of_events
        }
        
        tracks_final.append(track_final)
    
    return tracks_final


def create_distance_matrix(coord1: np.ndarray, coord2: np.ndarray) -> np.ndarray:
    """
    Create distance matrix between two sets of coordinates
    
    Args:
        coord1: First set of coordinates (N x D)
        coord2: Second set of coordinates (M x D)
        
    Returns:
        Distance matrix (N x M)
    """
    if coord1.size == 0 or coord2.size == 0:
        return np.array([])
    
    # Ensure 2D arrays
    if coord1.ndim == 1:
        coord1 = coord1.reshape(1, -1)
    if coord2.ndim == 1:
        coord2 = coord2.reshape(1, -1)
    
    # Calculate pairwise distances
    from scipy.spatial.distance import cdist
    return cdist(coord1, coord2)


def optimal_histogram(data: np.ndarray, bins: Optional[np.ndarray] = None, plot: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create optimal histogram (placeholder for MATLAB's optimalHistogram)
    
    Args:
        data: Data to histogram
        bins: Bin edges (if None, auto-determine)
        plot: Whether to plot histogram
        
    Returns:
        Tuple of (counts, bin_edges)
    """
    if data.size == 0:
        return np.array([]), np.array([])
    
    if bins is None:
        # Auto-determine number of bins using Freedman-Diaconis rule
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        bin_width = 2 * iqr / (len(data) ** (1/3))
        if bin_width > 0:
            num_bins = int(np.ceil((np.max(data) - np.min(data)) / bin_width))
            num_bins = max(1, min(num_bins, 100))  # Reasonable limits
        else:
            num_bins = 10
        bins = num_bins
    
    counts, bin_edges = np.histogram(data, bins=bins)
    
    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.hist(data, bins=bin_edges)
            plt.show()
        except ImportError:
            logger.warning("matplotlib not available for plotting")
    
    return counts, bin_edges


def percentile(data: np.ndarray, pct: float) -> float:
    """
    Calculate percentile (equivalent to MATLAB's prctile)
    
    Args:
        data: Input data
        pct: Percentile to calculate (0-100)
        
    Returns:
        Percentile value
    """
    if data.size == 0:
        return np.nan
    
    # Remove NaN values
    clean_data = data[~np.isnan(data)]
    
    if clean_data.size == 0:
        return np.nan
    
    return np.percentile(clean_data, pct)


def sparse_to_full_safe(sparse_matrix: sp.spmatrix) -> np.ndarray:
    """
    Safely convert sparse matrix to full, handling NaN appropriately
    
    Args:
        sparse_matrix: Sparse matrix to convert
        
    Returns:
        Dense numpy array
    """
    if not sp.issparse(sparse_matrix):
        return sparse_matrix
    
    # Convert to dense
    dense_matrix = sparse_matrix.toarray()
    
    # Replace zeros with NaN where appropriate
    # This logic would need to be customized based on the specific use case
    
    return dense_matrix


def validate_movie_info(movie_info: List[Dict], prob_dim: int = 2) -> List[Dict]:
    """
    Validate and standardize movie_info structure
    
    Args:
        movie_info: Input movie information
        prob_dim: Problem dimensionality
        
    Returns:
        Validated movie information
    """
    validated_info = []
    
    for i, frame_info in enumerate(movie_info):
        validated_frame = frame_info.copy()
        
        # Ensure required fields exist
        if 'num' not in validated_frame:
            if 'x_coord' in validated_frame:
                validated_frame['num'] = len(validated_frame['x_coord'])
            else:
                validated_frame['num'] = 0
        
        # Create all_coord if it doesn't exist
        if 'all_coord' not in validated_frame and validated_frame['num'] > 0:
            coords = []
            
            # Add x coordinates
            if 'x_coord' in validated_frame:
                x_coord = np.array(validated_frame['x_coord'])
                if x_coord.ndim == 1:
                    x_coord = np.column_stack([x_coord, np.zeros(len(x_coord))])
                coords.extend([x_coord[:, 0], x_coord[:, 1]])
            else:
                coords.extend([np.zeros(validated_frame['num']), np.zeros(validated_frame['num'])])
            
            # Add y coordinates
            if 'y_coord' in validated_frame:
                y_coord = np.array(validated_frame['y_coord'])
                if y_coord.ndim == 1:
                    y_coord = np.column_stack([y_coord, np.zeros(len(y_coord))])
                coords.extend([y_coord[:, 0], y_coord[:, 1]])
            else:
                coords.extend([np.zeros(validated_frame['num']), np.zeros(validated_frame['num'])])
            
            # Add z coordinates if 3D
            if prob_dim == 3:
                if 'z_coord' in validated_frame:
                    z_coord = np.array(validated_frame['z_coord'])
                    if z_coord.ndim == 1:
                        z_coord = np.column_stack([z_coord, np.zeros(len(z_coord))])
                    coords.extend([z_coord[:, 0], z_coord[:, 1]])
                else:
                    coords.extend([np.zeros(validated_frame['num']), np.zeros(validated_frame['num'])])
            
            validated_frame['all_coord'] = np.column_stack(coords)
        elif 'all_coord' not in validated_frame:
            validated_frame['all_coord'] = np.array([]).reshape(0, 2 * prob_dim)
        
        # Ensure amp exists
        if 'amp' not in validated_frame and validated_frame['num'] > 0:
            validated_frame['amp'] = np.ones((validated_frame['num'], 2))
        elif 'amp' not in validated_frame:
            validated_frame['amp'] = np.array([]).reshape(0, 2)
        
        validated_info.append(validated_frame)
    
    return validated_info
