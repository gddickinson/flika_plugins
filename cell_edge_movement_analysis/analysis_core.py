"""
Analysis Core Module for Cell Edge Movement Analysis

Contains the core analysis functions for detecting cell edges,
calculating movement, and correlating with PIEZO1 intensity.

Author: George Dickinson
"""

import numpy as np
import pandas as pd
from skimage import measure, draw
from scipy import stats, interpolate
import os
import datetime
import json

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def detect_cell_edge(mask):
    """
    Detect the cell edge from a binary mask.
    
    Parameters
    ----------
    mask : numpy.ndarray
        Binary mask image
        
    Returns
    -------
    contour : numpy.ndarray
        Array of (y, x) coordinates defining the cell edge
    """
    contours = measure.find_contours(mask, 0.5)
    
    if len(contours) > 0:
        contour = max(contours, key=len)
        return contour
    else:
        return np.array([])


def detect_edge_movement_vertical_xaxis(current_contour, previous_contour, 
                                       current_mask, previous_mask, config):
    """
    Detect movement using vertical displacement at every x-position.
    Tracks uppermost (minimum y) point at each x on both frames.
    
    Parameters
    ----------
    current_contour : numpy.ndarray
        Edge contour from current frame
    previous_contour : numpy.ndarray
        Edge contour from previous frame
    current_mask : numpy.ndarray
        Mask from current frame
    previous_mask : numpy.ndarray
        Mask from previous frame
    config : dict
        Configuration parameters
        
    Returns
    -------
    movement_score : float
        Overall movement score (positive = down, negative = up)
    movement_type : str
        'extending', 'retracting', or 'stable'
    movement_map : numpy.ndarray
        2D map of movement values
    displacements : numpy.ndarray
        1D array of displacement values at each x-position
    x_positions : numpy.ndarray
        Array of x-positions
    """
    # Get x-range spanning both frames
    x_min = int(min(np.min(current_contour[:, 1]), np.min(previous_contour[:, 1])))
    x_max = int(max(np.max(current_contour[:, 1]), np.max(previous_contour[:, 1])))
    
    # Create arrays for every x-position
    x_positions = np.arange(x_min, x_max + 1)
    displacements = np.full(len(x_positions), np.nan, dtype=float)
    
    # For each x position, find uppermost (minimum y) points
    for i, x in enumerate(x_positions):
        # Previous frame: find uppermost point at this x
        prev_mask = np.abs(previous_contour[:, 1] - x) < 0.5
        if np.any(prev_mask):
            prev_y = np.min(previous_contour[prev_mask, 0])
        else:
            prev_y = None
        
        # Current frame: find uppermost point at this x
        curr_mask = np.abs(current_contour[:, 1] - x) < 0.5
        if np.any(curr_mask):
            curr_y = np.min(current_contour[curr_mask, 0])
        else:
            curr_y = None
        
        # Calculate displacement if both exist
        if prev_y is not None and curr_y is not None:
            displacements[i] = curr_y - prev_y  # Positive = down, negative = up
    
    # Calculate overall movement score
    valid_displacements = displacements[~np.isnan(displacements)]
    
    if len(valid_displacements) > config['min_movement_pixels']:
        movement_score = np.mean(valid_displacements)
    else:
        movement_score = 0.0
    
    # Classify movement type
    if movement_score < -config['movement_threshold']:
        movement_type = 'extending'
    elif movement_score > config['movement_threshold']:
        movement_type = 'retracting'
    else:
        movement_type = 'stable'
    
    # Create 2D movement map
    movement_map = create_movement_map_from_xaxis_displacement(
        displacements, x_positions, current_mask.shape)
    
    return movement_score, movement_type, movement_map, displacements, x_positions


def create_movement_map_from_xaxis_displacement(displacements, x_positions, image_shape):
    """
    Create 2D movement map where each x-column contains the displacement value.
    
    Parameters
    ----------
    displacements : numpy.ndarray
        1D array of displacement values
    x_positions : numpy.ndarray
        Array of x-positions
    image_shape : tuple
        Shape of output image (height, width)
        
    Returns
    -------
    movement_map : numpy.ndarray
        2D array with displacement values in corresponding columns
    """
    movement_map = np.full(image_shape, np.nan, dtype=float)
    
    for x, displacement in zip(x_positions, displacements):
        if not np.isnan(displacement) and 0 <= x < image_shape[1]:
            movement_map[:, int(x)] = displacement
            
    return movement_map


def sample_along_edge_xaxis(edge_coords, mask, n_points, depth, width, 
                            min_cell_coverage, try_rotation, exclude_endpoints):
    """
    Sample points along the cell edge at regular x-coordinate intervals.
    
    Parameters
    ----------
    edge_coords : numpy.ndarray
        Edge coordinates (y, x)
    mask : numpy.ndarray
        Binary mask
    n_points : int
        Number of points to sample
    depth : int
        Depth of sampling rectangle (pixels)
    width : int
        Width of sampling rectangle (pixels)
    min_cell_coverage : float
        Minimum fraction of rectangle inside cell
    try_rotation : bool
        Whether to try 180° rotation if initial placement fails
    exclude_endpoints : bool
        Whether to exclude first and last edge points
        
    Returns
    -------
    sampling_points : numpy.ndarray
        Array of valid sampling points
    valid_points : numpy.ndarray
        Boolean array indicating which points are valid
    sampling_rects : list
        List of sampling rectangle parameters
    point_status : numpy.ndarray
        Status codes for each point
    """
    if len(edge_coords) == 0:
        return np.array([]), np.array([]), [], np.array([])
    
    # Get x-range of edge
    x_min = np.min(edge_coords[:, 1])
    x_max = np.max(edge_coords[:, 1])
    
    # Create evenly spaced x-coordinates
    if exclude_endpoints and n_points > 2:
        # Exclude the very edges
        margin = (x_max - x_min) * 0.05
        x_sample = np.linspace(x_min + margin, x_max - margin, n_points)
    else:
        x_sample = np.linspace(x_min, x_max, n_points)
    
    sampling_points = []
    valid_points = []
    sampling_rects = []
    point_status = []
    
    for x_target in x_sample:
        # Find edge point(s) near this x-coordinate
        x_mask = np.abs(edge_coords[:, 1] - x_target) < (x_max - x_min) / (2 * n_points)
        
        if not np.any(x_mask):
            valid_points.append(False)
            point_status.append(0)  # No edge point found
            sampling_points.append((0, 0))
            sampling_rects.append(None)
            continue
        
        # Use uppermost (minimum y) point at this x
        y_edge = np.min(edge_coords[x_mask, 0])
        x_edge = x_target
        
        # Calculate perpendicular direction (pointing into cell)
        # For uppermost edge, perpendicular is downward
        dx, dy = 0, 1  # Point downward into cell
        
        # Try to place sampling rectangle
        rect_params = create_sampling_rectangle(
            x_edge, y_edge, dx, dy, depth, width, mask, min_cell_coverage
        )
        
        if rect_params is None and try_rotation:
            # Try 180° rotation
            rect_params = create_sampling_rectangle(
                x_edge, y_edge, -dx, -dy, depth, width, mask, min_cell_coverage
            )
            
        if rect_params is not None:
            valid_points.append(True)
            point_status.append(1)  # Valid
            sampling_points.append((y_edge, x_edge))
            sampling_rects.append(rect_params)
        else:
            valid_points.append(False)
            point_status.append(2)  # Insufficient coverage
            sampling_points.append((y_edge, x_edge))
            sampling_rects.append(None)
    
    return (np.array(sampling_points), np.array(valid_points), 
            sampling_rects, np.array(point_status))


def create_sampling_rectangle(x, y, dx, dy, depth, width, mask, min_coverage):
    """
    Create a sampling rectangle at the specified position and direction.
    
    Parameters
    ----------
    x, y : float
        Center position
    dx, dy : float
        Direction vector (normalized)
    depth : int
        Depth of rectangle
    width : int
        Width of rectangle
    mask : numpy.ndarray
        Binary mask
    min_coverage : float
        Minimum fraction of rectangle inside cell
        
    Returns
    -------
    rect_params : dict or None
        Rectangle parameters if valid, None otherwise
    """
    # Normalize direction
    norm = np.sqrt(dx**2 + dy**2)
    if norm > 0:
        dx, dy = dx / norm, dy / norm
    
    # Perpendicular direction
    px, py = -dy, dx
    
    # Create rectangle corners
    corners = np.array([
        [x - px * width/2, y - py * width/2],  # Start of width
        [x + px * width/2, y + py * width/2],  # End of width
        [x + px * width/2 + dx * depth, y + py * width/2 + dy * depth],  # Far corner
        [x - px * width/2 + dx * depth, y - py * width/2 + dy * depth]   # Far corner
    ])
    
    # Check if rectangle is within image bounds
    if (np.any(corners < 0) or 
        np.any(corners[:, 0] >= mask.shape[1]) or 
        np.any(corners[:, 1] >= mask.shape[0])):
        return None
    
    # Create mask for rectangle
    rr, cc = draw.polygon(corners[:, 1], corners[:, 0], mask.shape)
    
    if len(rr) == 0:
        return None
    
    # Check coverage
    coverage = np.sum(mask[rr, cc] > 0) / len(rr)
    
    if coverage >= min_coverage:
        return {
            'center': (x, y),
            'direction': (dx, dy),
            'corners': corners,
            'rr': rr,
            'cc': cc,
            'coverage': coverage
        }
    
    return None


def calculate_local_movement(rect, movement_map):
    """
    Calculate local movement score from movement map within a sampling rectangle.
    
    Parameters
    ----------
    rect : dict
        Sampling rectangle parameters
    movement_map : numpy.ndarray
        2D movement map
        
    Returns
    -------
    local_score : float
        Average movement within the rectangle
    """
    if rect is None:
        return np.nan
    
    rr, cc = rect['rr'], rect['cc']
    values = movement_map[rr, cc]
    valid_values = values[~np.isnan(values)]
    
    if len(valid_values) > 0:
        return np.mean(valid_values)
    else:
        return np.nan


def calculate_local_intensity(rect, image):
    """
    Calculate average intensity within a sampling rectangle.
    
    Parameters
    ----------
    rect : dict
        Sampling rectangle parameters
    image : numpy.ndarray
        Intensity image
        
    Returns
    -------
    intensity : float
        Average intensity within the rectangle
    """
    if rect is None:
        return np.nan
    
    rr, cc = rect['rr'], rect['cc']
    values = image[rr, cc]
    
    if len(values) > 0:
        return np.mean(values)
    else:
        return np.nan


def analyze_frame_pair(current_image, current_mask, comparison_mask, config, temporal_direction):
    """
    Analyze a pair of frames for movement and intensity correlation.
    
    Parameters
    ----------
    current_image : numpy.ndarray
        PIEZO1 intensity image (current frame)
    current_mask : numpy.ndarray
        Cell mask (current frame)
    comparison_mask : numpy.ndarray
        Cell mask (comparison frame - previous or next depending on temporal_direction)
    config : dict
        Analysis configuration
    temporal_direction : str
        'past' or 'future'
        
    Returns
    -------
    results : dict
        Dictionary containing analysis results
    """
    # Detect edges
    current_contour = detect_cell_edge(current_mask)
    comparison_contour = detect_cell_edge(comparison_mask)
    
    if len(current_contour) == 0 or len(comparison_contour) == 0:
        return {'error': 'No cell edge detected'}
    
    # Calculate movement
    movement_score, movement_type, movement_map, displacements, x_positions = \
        detect_edge_movement_vertical_xaxis(
            current_contour, comparison_contour, current_mask, comparison_mask, config
        )
    
    # Sample along edge
    sampling_points, valid_points, sampling_rects, point_status = \
        sample_along_edge_xaxis(
            current_contour, current_mask,
            config['n_points'],
            config['depth'],
            config['width'],
            config['min_cell_coverage'],
            config['try_rotation'],
            config['exclude_endpoints']
        )
    
    # Calculate local movement scores and intensities
    local_movement_scores = []
    intensities = []
    
    for rect in sampling_rects:
        local_score = calculate_local_movement(rect, movement_map)
        intensity = calculate_local_intensity(rect, current_image)
        
        local_movement_scores.append(local_score)
        intensities.append(intensity)
    
    results = {
        'current_contour': current_contour,
        'comparison_contour': comparison_contour,
        'movement_score': movement_score,
        'movement_type': movement_type,
        'movement_map': movement_map,
        'displacements': displacements,
        'x_positions': x_positions,
        'sampling_points': sampling_points,
        'valid_points': valid_points,
        'sampling_rects': sampling_rects,
        'point_status': point_status,
        'local_movement_scores': np.array(local_movement_scores),
        'intensities': np.array(intensities)
    }
    
    return results


def run_analysis(piezo1_window, mask_window, config, output_config, viz_config, 
                progress_callback=None):
    """
    Run the complete cell edge movement analysis.
    
    Parameters
    ----------
    piezo1_window : flika.window.Window
        FLIKA window containing PIEZO1 signal
    mask_window : flika.window.Window
        FLIKA window containing cell masks
    config : dict
        Analysis configuration
    output_config : dict
        Output configuration
    viz_config : dict
        Visualization configuration
    progress_callback : callable, optional
        Function to call with progress updates (percentage, message)
        
    Returns
    -------
    results : dict
        Dictionary containing all analysis results
    """
    if progress_callback:
        progress_callback(0, "Initializing analysis...")
    
    # Get image arrays from windows
    images = piezo1_window.image
    masks = mask_window.image
    
    # Ensure binary masks
    masks = (masks > 0).astype(np.uint8)
    
    # Ensure 3D arrays
    if images.ndim == 2:
        images = images[np.newaxis, ...]
    if masks.ndim == 2:
        masks = masks[np.newaxis, ...]
    
    n_frames = images.shape[0]
    temporal_direction = config['temporal_direction']
    
    if progress_callback:
        progress_callback(5, f"Processing {n_frames} frames...")
    
    # Initialize storage
    all_results = []
    all_movements = []
    all_intensities = []
    all_valid_points = []
    movement_scores = []
    movement_types = []
    
    # Process frame pairs
    for i in range(n_frames - 1):
        if temporal_direction == 'past':
            current_idx = i + 1
            comparison_idx = i
        else:  # 'future'
            current_idx = i
            comparison_idx = i + 1
        
        current_image = images[current_idx]
        current_mask = masks[current_idx]
        comparison_mask = masks[comparison_idx]
        
        # Analyze frame pair
        results = analyze_frame_pair(
            current_image, current_mask, comparison_mask, config, temporal_direction
        )
        
        if 'error' not in results:
            all_results.append(results)
            all_movements.append(results['local_movement_scores'])
            all_intensities.append(results['intensities'])
            all_valid_points.append(results['valid_points'])
            movement_scores.append(results['movement_score'])
            movement_types.append(results['movement_type'])
        
        # Update progress
        if progress_callback and (i + 1) % 5 == 0:
            percentage = int((i + 1) / (n_frames - 1) * 80) + 10
            progress_callback(percentage, f"Processed {i+1}/{n_frames-1} transitions...")
    
    if progress_callback:
        progress_callback(90, "Calculating statistics...")
    
    # Calculate statistics
    processed_pairs = len(all_results)
    total_points = sum(len(m) for m in all_movements)
    valid_measurements = sum(np.sum(v) for v in all_valid_points)
    
    extending_count = sum(1 for t in movement_types if t == 'extending')
    retracting_count = sum(1 for t in movement_types if t == 'retracting')
    stable_count = sum(1 for t in movement_types if t == 'stable')
    
    # Correlation analysis
    if len(all_movements) > 0:
        # Flatten arrays for correlation
        all_mov_flat = []
        all_int_flat = []
        
        for movements, intensities, valid in zip(all_movements, all_intensities, all_valid_points):
            for mov, intensity, v in zip(movements, intensities, valid):
                if v and not np.isnan(mov) and not np.isnan(intensity):
                    all_mov_flat.append(mov)
                    all_int_flat.append(intensity)
        
        if len(all_mov_flat) > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                all_int_flat, all_mov_flat
            )
            correlation_stats = {
                'r_squared': r_value**2,
                'p_value': p_value,
                'sample_size': len(all_mov_flat),
                'slope': slope,
                'intercept': intercept,
                'std_err': std_err
            }
        else:
            correlation_stats = {
                'r_squared': None,
                'p_value': None,
                'sample_size': len(all_mov_flat),
                'slope': None
            }
    else:
        correlation_stats = {
            'r_squared': None,
            'p_value': None,
            'sample_size': 0,
            'slope': None
        }
    
    # Compile summary statistics
    summary_stats = {
        'total_frames': int(n_frames),
        'total_transitions': int(n_frames - 1),
        'processed_transitions': int(processed_pairs),
        'total_sampled_points': int(total_points),
        'valid_measurements': int(valid_measurements),
        'valid_measurement_percentage': float((valid_measurements / total_points * 100) if total_points > 0 else 0),
        'method': 'vertical_displacement_xaxis_full',
        'sampling': 'x_axis',
        'temporal_direction': temporal_direction,
        'movement_statistics': {
            'extending_transitions': int(extending_count),
            'retracting_transitions': int(retracting_count),
            'stable_transitions': int(stable_count),
            'average_movement_score': float(np.mean(movement_scores)) if movement_scores else 0.0,
            'movement_score_std': float(np.std(movement_scores)) if movement_scores else 0.0
        },
        'correlation_analysis': correlation_stats
    }
    
    if progress_callback:
        progress_callback(95, "Analysis complete!")
    
    results_dict = {
        'all_results': all_results,
        'all_movements': all_movements,
        'all_intensities': all_intensities,
        'all_valid_points': all_valid_points,
        'movement_scores': movement_scores,
        'movement_types': movement_types,
        'summary_stats': summary_stats,
        'config': config
    }
    
    if progress_callback:
        progress_callback(100, "Done!")
    
    return results_dict
