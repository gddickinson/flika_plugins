#!/usr/bin/env python3
"""
Advanced Configuration Examples for Python u-track

This module provides pre-configured parameter sets for common tracking scenarios
and advanced configuration utilities.

Copyright (C) 2025, Danuser Lab - UTSouthwestern 
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from track_general import CostMatrixParameters, GapCloseParameters, KalmanFunctions
import logging

logger = logging.getLogger(__name__)


class TrackingConfigurations:
    """Pre-configured parameter sets for common tracking scenarios"""
    
    @staticmethod
    def fast_particles_high_density():
        """Configuration for fast-moving particles in high-density conditions"""
        
        link_params = CostMatrixParameters(
            linear_motion=1,
            min_search_radius=1.0,
            max_search_radius=8.0,
            brown_std_mult=2.5,
            lin_std_mult=np.array([2.5, 3.0, 3.5, 4.0, 4.5]),
            use_local_density=1,
            max_angle_vv=25.0,
            brown_scaling=[0.5, 0.01],
            lin_scaling=[1.2, 0.01],
            time_reach_conf_b=3,
            time_reach_conf_l=3
        )
        
        gap_params = CostMatrixParameters(
            linear_motion=1,
            min_search_radius=1.0,
            max_search_radius=12.0,
            brown_std_mult=np.array([2.5, 3.0, 3.5, 4.0, 4.5]),
            lin_std_mult=np.array([2.5, 3.0, 3.5, 4.0, 4.5]),
            gap_penalty=1.1,
            use_local_density=1,
            max_angle_vv=35.0
        )
        
        gap_close_params = GapCloseParameters(
            time_window=3,
            merge_split=1,
            min_track_len=4,
            tolerance=0.02
        )
        
        return {
            'cost_matrices': [
                {'func_name': 'cost_mat_random_directed_switching_motion_link', 
                 'parameters': link_params.__dict__},
                {'func_name': 'cost_mat_random_directed_switching_motion_close_gaps',
                 'parameters': gap_params.__dict__}
            ],
            'gap_close_param': gap_close_params,
            'description': 'Optimized for fast particles in crowded environments'
        }
    
    @staticmethod
    def slow_particles_low_density():
        """Configuration for slow-moving particles in sparse conditions"""
        
        link_params = CostMatrixParameters(
            linear_motion=0,  # Primarily Brownian motion
            min_search_radius=3.0,
            max_search_radius=15.0,
            brown_std_mult=4.0,
            use_local_density=1,
            brown_scaling=[0.25, 0.01],
            time_reach_conf_b=8
        )
        
        gap_params = CostMatrixParameters(
            linear_motion=0,
            min_search_radius=3.0,
            max_search_radius=25.0,
            brown_std_mult=np.array([4.0, 5.0, 6.0, 7.0, 8.0]),
            gap_penalty=1.02,
            use_local_density=1
        )
        
        gap_close_params = GapCloseParameters(
            time_window=10,
            merge_split=0,  # Disable merge/split for simple tracking
            min_track_len=8,
            tolerance=0.1
        )
        
        return {
            'cost_matrices': [
                {'func_name': 'cost_mat_random_directed_switching_motion_link',
                 'parameters': link_params.__dict__},
                {'func_name': 'cost_mat_random_directed_switching_motion_close_gaps',
                 'parameters': gap_params.__dict__}
            ],
            'gap_close_param': gap_close_params,
            'description': 'Optimized for slow, diffusive particles'
        }
    
    @staticmethod
    def directed_migration():
        """Configuration for directionally migrating cells/particles"""
        
        link_params = CostMatrixParameters(
            linear_motion=1,
            min_search_radius=2.0,
            max_search_radius=20.0,
            brown_std_mult=2.0,
            lin_std_mult=np.array([3.0, 4.0, 5.0, 6.0, 7.0]),
            use_local_density=1,
            max_angle_vv=20.0,  # Strict directionality
            brown_scaling=[0.2, 0.01],
            lin_scaling=[1.5, 0.05],
            time_reach_conf_b=5,
            time_reach_conf_l=8
        )
        
        gap_params = CostMatrixParameters(
            linear_motion=1,
            min_search_radius=2.0,
            max_search_radius=30.0,
            brown_std_mult=np.array([2.0, 3.0, 4.0, 5.0, 6.0]),
            lin_std_mult=np.array([3.0, 4.0, 5.0, 6.0, 7.0]),
            gap_penalty=1.05,
            use_local_density=1,
            max_angle_vv=30.0
        )
        
        gap_close_params = GapCloseParameters(
            time_window=8,
            merge_split=2,  # Allow merging only (cells coming together)
            min_track_len=10,
            tolerance=0.05
        )
        
        return {
            'cost_matrices': [
                {'func_name': 'cost_mat_random_directed_switching_motion_link',
                 'parameters': link_params.__dict__},
                {'func_name': 'cost_mat_random_directed_switching_motion_close_gaps',
                 'parameters': gap_params.__dict__}
            ],
            'gap_close_param': gap_close_params,
            'description': 'Optimized for directional cell migration'
        }
    
    @staticmethod
    def membrane_proteins():
        """Configuration for membrane protein tracking (confined + directed)"""
        
        link_params = CostMatrixParameters(
            linear_motion=2,  # Allow direction switching
            min_search_radius=1.0,
            max_search_radius=6.0,
            brown_std_mult=3.0,
            lin_std_mult=np.array([2.0, 2.5, 3.0, 3.5, 4.0]),
            use_local_density=1,
            max_angle_vv=60.0,  # Allow more angular variation
            brown_scaling=[0.3, 0.01],
            lin_scaling=[0.8, 0.01],  # Slower scaling for confined motion
            time_reach_conf_b=3,
            time_reach_conf_l=4,
            res_limit=0.5  # Account for localization precision
        )
        
        gap_params = CostMatrixParameters(
            linear_motion=2,
            min_search_radius=1.0,
            max_search_radius=10.0,
            brown_std_mult=np.array([3.0, 3.5, 4.0, 4.5, 5.0]),
            lin_std_mult=np.array([2.0, 2.5, 3.0, 3.5, 4.0]),
            gap_penalty=1.2,  # Higher penalty for gaps
            use_local_density=1,
            max_angle_vv=90.0,
            res_limit=0.5
        )
        
        gap_close_params = GapCloseParameters(
            time_window=5,
            merge_split=1,
            min_track_len=5,
            tolerance=0.05
        )
        
        return {
            'cost_matrices': [
                {'func_name': 'cost_mat_random_directed_switching_motion_link',
                 'parameters': link_params.__dict__},
                {'func_name': 'cost_mat_random_directed_switching_motion_close_gaps',
                 'parameters': gap_params.__dict__}
            ],
            'gap_close_param': gap_close_params,
            'description': 'Optimized for membrane protein dynamics'
        }
    
    @staticmethod
    def virus_tracking():
        """Configuration for virus particle tracking"""
        
        link_params = CostMatrixParameters(
            linear_motion=1,
            min_search_radius=1.5,
            max_search_radius=12.0,
            brown_std_mult=2.5,
            lin_std_mult=np.array([3.0, 3.5, 4.0, 4.5, 5.0]),
            use_local_density=1,
            max_angle_vv=40.0,
            brown_scaling=[0.4, 0.01],
            lin_scaling=[1.0, 0.02],
            time_reach_conf_b=4,
            time_reach_conf_l=6,
            amp_ratio_limit=[0.6, 1.6]  # Account for variable brightness
        )
        
        gap_params = CostMatrixParameters(
            linear_motion=1,
            min_search_radius=1.5,
            max_search_radius=18.0,
            brown_std_mult=np.array([2.5, 3.0, 3.5, 4.0, 4.5]),
            lin_std_mult=np.array([3.0, 3.5, 4.0, 4.5, 5.0]),
            gap_penalty=1.15,
            use_local_density=1,
            max_angle_vv=50.0,
            amp_ratio_limit=[0.6, 1.6]
        )
        
        gap_close_params = GapCloseParameters(
            time_window=6,
            merge_split=3,  # Allow splitting only (virus budding)
            min_track_len=6,
            tolerance=0.05
        )
        
        return {
            'cost_matrices': [
                {'func_name': 'cost_mat_random_directed_switching_motion_link',
                 'parameters': link_params.__dict__},
                {'func_name': 'cost_mat_random_directed_switching_motion_close_gaps',
                 'parameters': gap_params.__dict__}
            ],
            'gap_close_param': gap_close_params,
            'description': 'Optimized for virus particle tracking'
        }


class ParameterOptimizer:
    """Utilities for parameter optimization and validation"""
    
    @staticmethod
    def validate_parameters(cost_matrices: List[Dict], gap_close_param: GapCloseParameters) -> List[str]:
        """Validate parameter consistency and return warnings"""
        warnings = []
        
        # Check cost matrix parameters
        for i, cost_mat in enumerate(cost_matrices):
            params = cost_mat.get('parameters', {})
            
            # Search radius validation
            min_radius = params.get('min_search_radius', 0)
            max_radius = params.get('max_search_radius', 0)
            
            if min_radius >= max_radius:
                warnings.append(f"Cost matrix {i}: min_search_radius >= max_search_radius")
            
            if min_radius < 0.5:
                warnings.append(f"Cost matrix {i}: min_search_radius very small ({min_radius})")
            
            if max_radius > 50:
                warnings.append(f"Cost matrix {i}: max_search_radius very large ({max_radius})")
            
            # Motion model validation
            linear_motion = params.get('linear_motion', 0)
            if linear_motion not in [0, 1, 2]:
                warnings.append(f"Cost matrix {i}: invalid linear_motion value ({linear_motion})")
            
            # Multiplier validation
            brown_mult = params.get('brown_std_mult', 3.0)
            if isinstance(brown_mult, (list, np.ndarray)):
                if np.any(np.array(brown_mult) < 1.0):
                    warnings.append(f"Cost matrix {i}: brown_std_mult contains values < 1.0")
            elif brown_mult < 1.0:
                warnings.append(f"Cost matrix {i}: brown_std_mult < 1.0")
        
        # Gap closing validation
        if gap_close_param.time_window > 20:
            warnings.append(f"Gap closing: time_window very large ({gap_close_param.time_window})")
        
        if gap_close_param.min_track_len < 2:
            warnings.append(f"Gap closing: min_track_len very small ({gap_close_param.min_track_len})")
        
        return warnings
    
    @staticmethod
    def suggest_parameters(movie_info: List[Dict], analysis_type: str = 'auto') -> Dict:
        """Suggest parameters based on detection characteristics"""
        
        # Analyze detection characteristics
        detection_stats = ParameterOptimizer._analyze_detections(movie_info)
        
        if analysis_type == 'auto':
            # Auto-determine best configuration
            if detection_stats['avg_density'] > 20 and detection_stats['avg_displacement'] > 5:
                config_type = 'fast_particles_high_density'
            elif detection_stats['avg_density'] < 5 and detection_stats['avg_displacement'] < 2:
                config_type = 'slow_particles_low_density'
            elif detection_stats['directionality_hint'] > 0.3:
                config_type = 'directed_migration'
            else:
                config_type = 'membrane_proteins'  # Default balanced config
        else:
            config_type = analysis_type
        
        # Get base configuration
        configs = TrackingConfigurations()
        if hasattr(configs, config_type):
            base_config = getattr(configs, config_type)()
        else:
            base_config = configs.membrane_proteins()  # Default
        
        # Adjust based on detection stats
        adjusted_config = ParameterOptimizer._adjust_for_data(base_config, detection_stats)
        
        return {
            'config': adjusted_config,
            'detection_stats': detection_stats,
            'reasoning': f'Selected {config_type} based on detection analysis'
        }
    
    @staticmethod
    def _analyze_detections(movie_info: List[Dict]) -> Dict:
        """Analyze detection characteristics to suggest parameters"""
        
        total_detections = 0
        frame_counts = []
        all_positions = []
        
        for frame_info in movie_info:
            num_det = frame_info.get('num', 0)
            frame_counts.append(num_det)
            total_detections += num_det
            
            if num_det > 0:
                if 'all_coord' in frame_info:
                    coords = frame_info['all_coord'][:, ::2]  # x, y coordinates
                elif 'x_coord' in frame_info and 'y_coord' in frame_info:
                    x_coords = frame_info['x_coord'][:, 0]
                    y_coords = frame_info['y_coord'][:, 0]
                    coords = np.column_stack([x_coords, y_coords])
                else:
                    continue
                
                all_positions.extend(coords.tolist())
        
        all_positions = np.array(all_positions)
        
        # Calculate statistics
        stats = {
            'num_frames': len(movie_info),
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / len(movie_info) if movie_info else 0,
            'avg_density': np.mean(frame_counts),
            'detection_variability': np.std(frame_counts),
        }
        
        if len(all_positions) > 0:
            # Spatial distribution
            x_range = np.ptp(all_positions[:, 0])
            y_range = np.ptp(all_positions[:, 1])
            stats['spatial_range'] = [x_range, y_range]
            
            # Estimate typical displacement by looking at frame-to-frame distances
            if len(movie_info) > 1:
                displacements = []
                for i in range(len(movie_info) - 1):
                    frame1 = movie_info[i]
                    frame2 = movie_info[i + 1]
                    
                    if frame1.get('num', 0) > 0 and frame2.get('num', 0) > 0:
                        # Simple nearest neighbor matching for displacement estimate
                        coords1 = ParameterOptimizer._get_frame_coords(frame1)
                        coords2 = ParameterOptimizer._get_frame_coords(frame2)
                        
                        if len(coords1) > 0 and len(coords2) > 0:
                            from scipy.spatial.distance import cdist
                            dist_mat = cdist(coords1, coords2)
                            min_dists = np.min(dist_mat, axis=1)
                            displacements.extend(min_dists[min_dists < 20])  # Reasonable displacement limit
                
                if displacements:
                    stats['avg_displacement'] = np.mean(displacements)
                    stats['displacement_std'] = np.std(displacements)
                    
                    # Rough directionality hint
                    stats['directionality_hint'] = min(1.0, np.std(displacements) / (np.mean(displacements) + 1e-6))
                else:
                    stats['avg_displacement'] = 3.0  # Default
                    stats['displacement_std'] = 2.0
                    stats['directionality_hint'] = 0.5
            else:
                stats['avg_displacement'] = 3.0
                stats['displacement_std'] = 2.0
                stats['directionality_hint'] = 0.5
        else:
            stats['spatial_range'] = [100, 100]  # Default
            stats['avg_displacement'] = 3.0
            stats['displacement_std'] = 2.0
            stats['directionality_hint'] = 0.5
        
        return stats
    
    @staticmethod
    def _get_frame_coords(frame_info: Dict) -> np.ndarray:
        """Extract coordinates from frame info"""
        if 'all_coord' in frame_info:
            return frame_info['all_coord'][:, ::2]
        elif 'x_coord' in frame_info and 'y_coord' in frame_info:
            x_coords = frame_info['x_coord'][:, 0]
            y_coords = frame_info['y_coord'][:, 0]
            return np.column_stack([x_coords, y_coords])
        else:
            return np.array([])
    
    @staticmethod
    def _adjust_for_data(base_config: Dict, detection_stats: Dict) -> Dict:
        """Adjust configuration based on detection statistics"""
        
        config = base_config.copy()
        
        # Adjust search radius based on typical displacement
        displacement_factor = detection_stats['avg_displacement'] / 3.0  # Normalize to typical value
        
        for cost_mat in config['cost_matrices']:
            params = cost_mat['parameters']
            
            # Scale search radii
            params['min_search_radius'] *= displacement_factor
            params['max_search_radius'] *= displacement_factor
            
            # Adjust for density
            density_factor = min(2.0, detection_stats['avg_density'] / 10.0)
            if density_factor > 1.5:  # High density
                params['min_search_radius'] *= 0.8  # Reduce to avoid over-linking
                if 'use_local_density' in params:
                    params['use_local_density'] = 1
            elif density_factor < 0.5:  # Low density
                params['max_search_radius'] *= 1.2  # Increase to catch more links
        
        # Adjust gap closing based on detection variability
        if detection_stats['detection_variability'] > detection_stats['avg_density'] * 0.5:
            # High variability suggests many gaps
            config['gap_close_param'].time_window = min(12, config['gap_close_param'].time_window + 2)
        
        return config


def create_custom_config(
    motion_type: str = 'mixed',
    typical_speed: float = 3.0,
    search_radius_factor: float = 3.0,
    gap_tolerance: int = 5,
    allow_merge_split: bool = True,
    strict_directionality: bool = False
) -> Dict:
    """
    Create custom configuration with simple parameters
    
    Args:
        motion_type: 'brownian', 'directed', 'mixed', or 'switching'
        typical_speed: Typical particle speed in pixels/frame
        search_radius_factor: Multiplier for search radius (higher = more permissive)
        gap_tolerance: Maximum frames to bridge gaps
        allow_merge_split: Whether to allow merging and splitting
        strict_directionality: Whether to enforce strict directional consistency
        
    Returns:
        Configuration dictionary
    """
    
    # Map motion type to linear_motion parameter
    motion_map = {
        'brownian': 0,
        'directed': 1,
        'mixed': 1,
        'switching': 2
    }
    
    linear_motion = motion_map.get(motion_type, 1)
    
    # Calculate search radii
    min_radius = max(1.0, typical_speed * 0.5)
    max_radius = typical_speed * search_radius_factor
    
    # Angle constraints
    max_angle = 15.0 if strict_directionality else 30.0
    
    # Create parameters
    link_params = CostMatrixParameters(
        linear_motion=linear_motion,
        min_search_radius=min_radius,
        max_search_radius=max_radius,
        brown_std_mult=search_radius_factor,
        lin_std_mult=np.array([search_radius_factor] * 5),
        use_local_density=1,
        max_angle_vv=max_angle,
        brown_scaling=[0.25, 0.01],
        lin_scaling=[1.0, 0.01] if motion_type == 'directed' else [0.5, 0.01],
        time_reach_conf_b=gap_tolerance,
        time_reach_conf_l=gap_tolerance
    )
    
    gap_params = CostMatrixParameters(
        linear_motion=linear_motion,
        min_search_radius=min_radius,
        max_search_radius=max_radius * 1.5,
        brown_std_mult=np.linspace(search_radius_factor, search_radius_factor * 1.5, 5),
        lin_std_mult=np.linspace(search_radius_factor, search_radius_factor * 1.5, 5),
        gap_penalty=1.05,
        use_local_density=1,
        max_angle_vv=max_angle * 1.5
    )
    
    merge_split_mode = 1 if allow_merge_split else 0
    
    gap_close_params = GapCloseParameters(
        time_window=gap_tolerance,
        merge_split=merge_split_mode,
        min_track_len=max(3, gap_tolerance // 2),
        tolerance=0.05
    )
    
    return {
        'cost_matrices': [
            {'func_name': 'cost_mat_random_directed_switching_motion_link',
             'parameters': link_params.__dict__},
            {'func_name': 'cost_mat_random_directed_switching_motion_close_gaps',
             'parameters': gap_params.__dict__}
        ],
        'gap_close_param': gap_close_params,
        'description': f'Custom config: {motion_type} motion, speed={typical_speed}, tolerance={gap_tolerance}'
    }


def get_all_configurations() -> Dict[str, Dict]:
    """Get all available pre-configured parameter sets"""
    
    configs = TrackingConfigurations()
    
    available_configs = {
        'fast_particles_high_density': configs.fast_particles_high_density(),
        'slow_particles_low_density': configs.slow_particles_low_density(),
        'directed_migration': configs.directed_migration(),
        'membrane_proteins': configs.membrane_proteins(),
        'virus_tracking': configs.virus_tracking()
    }
    
    return available_configs


def print_config_summary(config: Dict):
    """Print a summary of configuration parameters"""
    
    print("Configuration Summary")
    print("=" * 40)
    print(f"Description: {config.get('description', 'Custom configuration')}")
    print()
    
    # Link parameters
    link_params = config['cost_matrices'][0]['parameters']
    print("Linking Parameters:")
    print(f"  Motion model: {link_params.get('linear_motion', 'Unknown')}")
    print(f"  Search radius: {link_params.get('min_search_radius', 0):.1f} - {link_params.get('max_search_radius', 0):.1f} pixels")
    print(f"  Brownian multiplier: {link_params.get('brown_std_mult', 0)}")
    print(f"  Use local density: {link_params.get('use_local_density', False)}")
    print(f"  Max angle: {link_params.get('max_angle_vv', 0):.1f}°")
    print()
    
    # Gap closing parameters
    gap_params = config['gap_close_param']
    print("Gap Closing Parameters:")
    print(f"  Time window: {gap_params.time_window} frames")
    print(f"  Merge/split mode: {gap_params.merge_split}")
    print(f"  Min track length: {gap_params.min_track_len} frames")
    print(f"  Tolerance: {gap_params.tolerance}")
    print()
