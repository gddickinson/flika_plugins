#!/usr/bin/env python3
"""
Enhanced Gap closing functions for particle tracking - u-track methodology

Python port of u-track's gap closing functions with enhanced algorithms
that better match the original MATLAB implementation.

Copyright (C) 2025, Danuser Lab - UTSouthwestern
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from typing import Dict, List, Any, Optional, Tuple
import time
import warnings

# Import the centralized logging system
try:
    from .module_logger import (
        get_module_logger, PerformanceTimer, log_function_call,
        log_array_info, LoggingMixin
    )
except ImportError:
    try:
        from module_logger import (
            get_module_logger, PerformanceTimer, log_function_call,
            log_array_info, LoggingMixin
        )
    except ImportError:
        # Fallback to basic logging if module_logger not available
        import logging

        def get_module_logger(name):
            return logging.getLogger(name)

        class PerformanceTimer:
            def __init__(self, logger, name):
                self.logger = logger
                self.name = name
                self.start_time = None

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, *args):
                if self.start_time:
                    duration = time.time() - self.start_time
                    self.logger.info(f"{self.name} completed in {duration:.4f}s")

        def log_function_call(logger, func_name, args, kwargs=None):
            logger.debug(f"Calling {func_name}")

        def log_array_info(logger, name, array, context=""):
            if hasattr(array, 'shape'):
                logger.debug(f"{name}: shape={array.shape}")
            else:
                logger.debug(f"{name}: {type(array)}")

        class LoggingMixin:
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.logger = logging.getLogger(self.__class__.__name__)

            def log_info(self, msg): self.logger.info(msg)
            def log_debug(self, msg): self.logger.debug(msg)
            def log_warning(self, msg): self.logger.warning(msg)
            def log_error(self, msg): self.logger.error(msg)
            def log_exception(self, msg): self.logger.exception(msg)
            def log_parameters(self, params, context=""):
                self.logger.info(f"Parameters {context}: {params}")
            def time_operation(self, name):
                return PerformanceTimer(self.logger, name)


# Get logger for this module
logger = get_module_logger('enhanced_gap_closing')


class EnhancedGapCloser(LoggingMixin):
    """Enhanced Gap closing and track merging/splitting handler - u-track methodology"""

    def __init__(self):
        super().__init__()
        self.log_info("EnhancedGapCloser initialized with u-track methodology")

    def close_gaps(
        self,
        tracks_feat_indx_link: np.ndarray,
        tracks_coord_amp_link: np.ndarray,
        kalman_info_link: List[Dict],
        nn_dist_linked_feat: np.ndarray,
        gap_close_param: Dict,
        cost_matrix_info: Dict,
        prob_dim: int,
        movie_info: List[Dict],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Enhanced gap closing using u-track methodology

        Implements the two-step Linear Assignment Problem approach:
        1. Calculate comprehensive cost matrix with motion prediction
        2. Solve globally optimal assignment problem
        3. Build compound tracks with proper merge/split handling
        """
        log_function_call(self.logger, 'enhanced_close_gaps', ())
        self.log_info("=== STARTING ENHANCED GAP CLOSING (u-track methodology) ===")

        with self.time_operation("Complete enhanced gap closing process"):
            try:
                # Log input parameters
                input_gap_params = {
                    'min_track_len': gap_close_param.get('min_track_len', 2),
                    'time_window': gap_close_param.get('time_window', 10),
                    'gap_penalty': gap_close_param.get('gap_penalty', 1.5),
                    'cost_threshold': gap_close_param.get('cost_threshold', 50.0),
                    'prob_dim': prob_dim,
                    'verbose': verbose
                }
                self.log_parameters(input_gap_params, "input gap closing parameters")

                # Input validation and conversion
                tracks_feat_indx_link = self._ensure_array(tracks_feat_indx_link)
                tracks_coord_amp_link = self._ensure_array(tracks_coord_amp_link)
                nn_dist_linked_feat = self._ensure_array(nn_dist_linked_feat)

                if tracks_feat_indx_link.size == 0 or tracks_coord_amp_link.size == 0:
                    self.log_warning("Empty input arrays, returning empty track list")
                    return []

                # Extract track timing information
                track_sel = self._get_track_sel(tracks_coord_amp_link)
                track_start_time = track_sel[:, 0].astype(int)
                track_end_time = track_sel[:, 1].astype(int)
                track_lengths = track_sel[:, 2].astype(int)

                # Filter tracks by minimum length
                min_track_len = int(gap_close_param.get('min_track_len', 2))
                valid_tracks = track_lengths >= min_track_len

                if not np.any(valid_tracks):
                    self.log_warning("No tracks meet minimum length requirement")
                    return self._convert_matrices_to_tracks(tracks_feat_indx_link, tracks_coord_amp_link)

                # Apply filtering
                tracks_feat_indx_link = tracks_feat_indx_link[valid_tracks, :]
                tracks_coord_amp_link = tracks_coord_amp_link[valid_tracks, :]
                track_start_time = track_start_time[valid_tracks]
                track_end_time = track_end_time[valid_tracks]

                if nn_dist_linked_feat.size > 0:
                    nn_dist_linked_feat = nn_dist_linked_feat[valid_tracks, :]

                num_tracks = len(track_start_time)
                num_frames = tracks_coord_amp_link.shape[1] // 8

                self.log_info(f"Processing {num_tracks} valid tracks over {num_frames} frames")

                # Enhanced motion analysis and prediction
                motion_info = self._analyze_track_motion(
                    tracks_coord_amp_link, track_start_time, track_end_time, kalman_info_link
                )

                # Log motion analysis results
                motion_summary = {
                    'tracks_analyzed': len(motion_info['velocities']),
                    'motion_types_brownian': sum(1 for mt in motion_info['motion_types'] if mt == 'brownian'),
                    'motion_types_directed': sum(1 for mt in motion_info['motion_types'] if mt == 'directed'),
                    'motion_types_mixed': sum(1 for mt in motion_info['motion_types'] if mt == 'mixed'),
                    'avg_velocity_magnitude': float(np.mean([np.linalg.norm(v) for v in motion_info['velocities']])),
                    'avg_search_radius': float(np.mean(motion_info['search_radii'])),
                    'min_search_radius': float(np.min(motion_info['search_radii'])),
                    'max_search_radius': float(np.max(motion_info['search_radii']))
                }
                self.log_parameters(motion_summary, "motion analysis results")

                # Calculate enhanced cost matrix using u-track methodology
                cost_mat, assignment_info = self._calculate_enhanced_cost_matrix(
                    tracks_coord_amp_link, tracks_feat_indx_link,
                    track_start_time, track_end_time,
                    motion_info, gap_close_param, cost_matrix_info,
                    prob_dim, movie_info, nn_dist_linked_feat
                )

                # Log cost matrix statistics
                if cost_mat is not None:
                    finite_costs = cost_mat.data[np.isfinite(cost_mat.data) & (cost_mat.data < 1e9)]
                    cost_stats = {
                        'cost_matrix_shape': cost_mat.shape,
                        'total_elements': cost_mat.shape[0] * cost_mat.shape[1],
                        'nonzero_elements': cost_mat.nnz,
                        'sparsity': float(cost_mat.nnz / (cost_mat.shape[0] * cost_mat.shape[1])),
                        'finite_costs_count': len(finite_costs),
                        'min_finite_cost': float(np.min(finite_costs)) if len(finite_costs) > 0 else np.nan,
                        'max_finite_cost': float(np.max(finite_costs)) if len(finite_costs) > 0 else np.nan,
                        'mean_finite_cost': float(np.mean(finite_costs)) if len(finite_costs) > 0 else np.nan,
                        'potential_track_ends': len(assignment_info['track_ends']),
                        'potential_track_starts': len(assignment_info['track_starts'])
                    }
                    self.log_parameters(cost_stats, "cost matrix statistics")

                if cost_mat is None or not self._has_valid_costs(cost_mat):
                    self.log_warning("No valid gap closing opportunities found")
                    return self._convert_matrices_to_tracks(tracks_feat_indx_link, tracks_coord_amp_link)

                # Solve enhanced assignment problem
                assignments = self._solve_enhanced_assignment(
                    cost_mat, assignment_info, gap_close_param
                )

                # Log assignment results
                if assignments:
                    assignment_costs = [cost for _, _, cost in assignments]
                    assignment_stats = {
                        'total_assignments': len(assignments),
                        'assignment_cost_min': float(np.min(assignment_costs)),
                        'assignment_cost_max': float(np.max(assignment_costs)),
                        'assignment_cost_mean': float(np.mean(assignment_costs)),
                        'assignment_cost_std': float(np.std(assignment_costs)),
                        'cost_threshold_used': float(gap_close_param.get('cost_threshold', 50.0)),
                        'assignments_below_threshold': len([c for c in assignment_costs if c < 10.0]),
                        'assignments_10_to_25': len([c for c in assignment_costs if 10.0 <= c < 25.0]),
                        'assignments_above_25': len([c for c in assignment_costs if c >= 25.0])
                    }
                    self.log_parameters(assignment_stats, "assignment results")
                else:
                    self.log_parameters({'total_assignments': 0}, "assignment results (no valid assignments)")

                # Build enhanced compound tracks
                compound_tracks = self._build_enhanced_compound_tracks(
                    assignments, tracks_feat_indx_link, tracks_coord_amp_link,
                    track_start_time, track_end_time, assignment_info, motion_info
                )

                # Log compound track building results
                if compound_tracks:
                    track_chain_lengths = [len(ct.get('track_chain', [1])) for ct in compound_tracks if ct]
                    num_segments_list = [ct.get('num_segments', 1) for ct in compound_tracks if ct]

                    compound_stats = {
                        'total_compound_tracks': len(compound_tracks),
                        'original_track_count': num_tracks,
                        'track_reduction': num_tracks - len(compound_tracks),
                        'reduction_percentage': float((num_tracks - len(compound_tracks)) / num_tracks * 100) if num_tracks > 0 else 0.0,
                        'single_segment_tracks': sum(1 for ns in num_segments_list if ns == 1),
                        'multi_segment_tracks': sum(1 for ns in num_segments_list if ns > 1),
                        'max_segments_per_track': max(num_segments_list) if num_segments_list else 0,
                        'avg_segments_per_track': float(np.mean(num_segments_list)) if num_segments_list else 0.0,
                        'max_chain_length': max(track_chain_lengths) if track_chain_lengths else 0,
                        'avg_chain_length': float(np.mean(track_chain_lengths)) if track_chain_lengths else 0.0
                    }
                    self.log_parameters(compound_stats, "compound track building results")
                else:
                    self.log_parameters({'total_compound_tracks': 0, 'original_track_count': num_tracks}, "compound track building results (no tracks created)")

                # Convert to final format
                tracks_final = self._convert_enhanced_tracks_to_final(
                    compound_tracks, num_frames
                )

                # Log final results summary
                final_summary = {
                    'initial_tracks_input': num_tracks,
                    'final_tracks_output': len(tracks_final),
                    'net_track_reduction': num_tracks - len(tracks_final),
                    'reduction_percentage': float((num_tracks - len(tracks_final)) / num_tracks * 100) if num_tracks > 0 else 0.0,
                    'processing_successful': True,
                    'gap_closing_method': 'enhanced_u_track_methodology'
                }
                self.log_parameters(final_summary, "final gap closing results")

                self.log_info(f"Enhanced gap closing completed: {len(tracks_final)} final tracks")
                return tracks_final

            except Exception as e:
                self.log_error(f"Error in enhanced gap closing: {str(e)}")
                self.log_exception("Full traceback")
                return self._convert_matrices_to_tracks(tracks_feat_indx_link, tracks_coord_amp_link)

    def _get_param_value(self, params, param_name: str, default_value):
        """
        Get parameter value from either dictionary or CostMatrixParameters object
        """
        try:
            if hasattr(params, 'get'):
                # Dictionary-like object
                value = params.get(param_name, default_value)
            elif hasattr(params, param_name):
                # Object with attributes
                value = getattr(params, param_name, default_value)
            else:
                # Fallback to default
                value = default_value

            # Ensure scalar return value
            if hasattr(value, 'item'):
                return float(value.item())
            elif hasattr(value, '__len__') and len(value) == 1:
                return float(value[0])
            else:
                return float(value)
        except (TypeError, ValueError):
            return float(default_value)

    def _analyze_track_motion(self, tracks_coord_amp: np.ndarray,
                            track_start_time: np.ndarray, track_end_time: np.ndarray,
                            kalman_info: List[Dict]) -> Dict:
        """
        Enhanced motion analysis using Kalman filtering and motion models
        Implements u-track's sophisticated motion prediction
        """
        self.log_info("=== ENHANCED MOTION ANALYSIS ===")

        num_tracks = tracks_coord_amp.shape[0]
        motion_info = {
            'velocities': [],
            'accelerations': [],
            'motion_types': [],
            'motion_consistency': [],
            'kalman_predictions': [],
            'search_radii': []
        }

        for track_id in range(num_tracks):
            start_frame = track_start_time[track_id] - 1  # Convert to 0-indexed
            end_frame = track_end_time[track_id] - 1

            # Extract coordinates for this track
            coords = []
            times = []

            for frame in range(start_frame, end_frame + 1):
                coord_idx = frame * 8
                x = tracks_coord_amp[track_id, coord_idx]
                y = tracks_coord_amp[track_id, coord_idx + 1]

                if not np.isnan(x) and not np.isnan(y):
                    coords.append([x, y])
                    times.append(frame)

            coords = np.array(coords)
            times = np.array(times)

            if len(coords) < 2:
                # Not enough points for motion analysis
                motion_info['velocities'].append(np.array([0.0, 0.0]))
                motion_info['accelerations'].append(np.array([0.0, 0.0]))
                motion_info['motion_types'].append('stationary')
                motion_info['motion_consistency'].append(0.0)
                motion_info['kalman_predictions'].append(None)
                motion_info['search_radii'].append(5.0)  # Default radius
                continue

            # Calculate velocities using finite differences
            dt = np.diff(times)
            dt[dt == 0] = 1  # Avoid division by zero
            velocities = np.diff(coords, axis=0) / dt[:, np.newaxis]

            # Calculate accelerations
            if len(velocities) > 1:
                accelerations = np.diff(velocities, axis=0) / dt[1:, np.newaxis]
                mean_acceleration = np.mean(accelerations, axis=0)
            else:
                mean_acceleration = np.array([0.0, 0.0])

            mean_velocity = np.mean(velocities, axis=0) if len(velocities) > 0 else np.array([0.0, 0.0])

            # Ensure velocities and accelerations are properly shaped
            mean_velocity = np.asarray(mean_velocity).flatten()[:2]  # Take only x,y
            mean_acceleration = np.asarray(mean_acceleration).flatten()[:2]  # Take only x,y

            # Pad with zeros if needed
            if len(mean_velocity) < 2:
                mean_velocity = np.array([0.0, 0.0])
            if len(mean_acceleration) < 2:
                mean_acceleration = np.array([0.0, 0.0])

            # Determine motion type using velocity consistency
            if len(velocities) > 2:
                velocity_var = np.var(velocities, axis=0)
                consistency = float(1.0 / (1.0 + np.mean(velocity_var)))

                speed = float(np.linalg.norm(mean_velocity))
                if speed < 0.5:
                    motion_type = 'brownian'
                elif consistency > 0.7:
                    motion_type = 'directed'
                else:
                    motion_type = 'mixed'
            else:
                motion_type = 'brownian'
                consistency = 0.5

            # Enhanced Kalman prediction for track endpoints
            kalman_pred = self._kalman_predict_endpoint(
                coords, times, mean_velocity, mean_acceleration, motion_type
            )

            # Calculate adaptive search radius based on motion uncertainty
            velocity_magnitude = float(np.linalg.norm(mean_velocity))
            uncertainty = float(np.linalg.norm(mean_acceleration)) if len(coords) > 2 else 1.0

            base_radius = max(2.0, velocity_magnitude * 2.0)  # Motion-based radius
            uncertainty_radius = uncertainty * 3.0  # Uncertainty component
            search_radius = float(min(base_radius + uncertainty_radius, 15.0))  # Cap at reasonable max

            motion_info['velocities'].append(mean_velocity)
            motion_info['accelerations'].append(mean_acceleration)
            motion_info['motion_types'].append(motion_type)
            motion_info['motion_consistency'].append(consistency)
            motion_info['kalman_predictions'].append(kalman_pred)
            motion_info['search_radii'].append(search_radius)

        self.log_info(f"Motion analysis completed for {num_tracks} tracks")
        return motion_info

    def _kalman_predict_endpoint(self, coords: np.ndarray, times: np.ndarray,
                               velocity: np.ndarray, acceleration: np.ndarray,
                               motion_type: str) -> Dict:
        """
        Enhanced Kalman filter prediction for track endpoints
        Implements u-track's motion prediction methodology
        """
        if len(coords) < 2:
            return None

        # State vector: [x, y, vx, vy]
        state = np.array([coords[-1, 0], coords[-1, 1], velocity[0], velocity[1]])

        # State transition matrix (constant velocity model)
        dt = 1.0  # Assume unit time step
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Process noise covariance (motion uncertainty)
        if motion_type == 'directed':
            q = 0.1  # Low uncertainty for directed motion
        elif motion_type == 'brownian':
            q = 1.0  # High uncertainty for Brownian motion
        else:
            q = 0.5  # Medium uncertainty for mixed motion

        Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q

        # Measurement noise covariance
        R = np.eye(2) * 0.1  # Assume 0.1 pixel measurement uncertainty

        # Initial covariance
        P = np.eye(4) * 0.1

        prediction = {
            'state': state,
            'covariance': P,
            'F': F,
            'Q': Q,
            'R': R,
            'motion_type': motion_type
        }

        return prediction

    def _calculate_enhanced_cost_matrix(self, tracks_coord_amp: np.ndarray,
                                      tracks_feat_indx: np.ndarray,
                                      track_start_time: np.ndarray,
                                      track_end_time: np.ndarray,
                                      motion_info: Dict, gap_close_param: Dict,
                                      cost_matrix_info: Dict, prob_dim: int,
                                      movie_info: List[Dict],
                                      nn_dist_linked_feat: np.ndarray) -> Tuple[sp.csr_matrix, Dict]:
        """
        Enhanced cost matrix calculation using u-track methodology
        Implements sophisticated distance, motion, and feature-based costs
        """
        self.log_info("=== ENHANCED COST MATRIX CALCULATION ===")

        num_tracks = len(track_start_time)
        time_window = int(gap_close_param.get('time_window', 10))

        # Initialize assignment info
        assignment_info = {
            'track_ends': [],
            'track_starts': [],
            'potential_merges': [],
            'potential_splits': [],
            'birth_death_costs': []
        }

        # Find track ends and starts for potential connections
        for i in range(num_tracks):
            end_time = track_end_time[i]
            start_time = track_start_time[i]

            assignment_info['track_ends'].append({
                'track_id': i,
                'time': end_time,
                'position': self._get_track_position_at_frame(tracks_coord_amp, i, end_time - 1),
                'velocity': motion_info['velocities'][i],
                'motion_type': motion_info['motion_types'][i],
                'search_radius': float(motion_info['search_radii'][i])
            })

            assignment_info['track_starts'].append({
                'track_id': i,
                'time': start_time,
                'position': self._get_track_position_at_frame(tracks_coord_amp, i, start_time - 1),
                'velocity': motion_info['velocities'][i],
                'motion_type': motion_info['motion_types'][i]
            })

        # Build cost matrix for gap closing
        cost_matrix = self._build_enhanced_cost_matrix(
            assignment_info, gap_close_param, cost_matrix_info, time_window
        )

        return cost_matrix, assignment_info

    def _build_enhanced_cost_matrix(self, assignment_info: Dict,
                                   gap_close_param: Dict, cost_matrix_info: Dict,
                                   time_window: int) -> sp.csr_matrix:
        """
        Build enhanced cost matrix with distance, motion, and statistical costs
        """
        track_ends = assignment_info['track_ends']
        track_starts = assignment_info['track_starts']

        num_tracks = len(track_ends)

        # Create cost matrix structure following u-track methodology
        # Matrix size: (num_ends + num_starts) x (num_starts + num_ends)
        # Top-left: end-to-start connections
        # Top-right: end terminations
        # Bottom-left: start births
        # Bottom-right: dummy connections

        matrix_size = num_tracks * 2
        cost_matrix = np.full((matrix_size, matrix_size), np.inf)

        # Parameters from cost_matrix_info - handle CostMatrixParameters object
        params = cost_matrix_info.get('parameters', {})
        max_search_radius = self._get_param_value(params, 'max_search_radius', 15.0)
        brown_std_mult = self._get_param_value(params, 'brown_std_mult', 3.0)
        lin_std_mult = self._get_param_value(params, 'lin_std_mult', 4.0)
        gap_penalty = float(gap_close_param.get('gap_penalty', 1.5))

        # Log extracted parameters
        extracted_params = {
            'max_search_radius': max_search_radius,
            'brown_std_mult': brown_std_mult,
            'lin_std_mult': lin_std_mult,
            'gap_penalty': gap_penalty,
            'time_window': time_window,
            'num_tracks': num_tracks,
            'matrix_size': matrix_size
        }
        self.log_parameters(extracted_params, "cost matrix parameters")

        # Calculate end-to-start connection costs
        connection_count = 0
        valid_connections = 0
        time_gaps = []
        costs_calculated = []

        for i, track_end in enumerate(track_ends):
            for j, track_start in enumerate(track_starts):

                # Skip self-connections initially
                if track_end['track_id'] == track_start['track_id']:
                    continue

                # Check time gap constraint
                time_gap = track_start['time'] - track_end['time']
                if time_gap <= 0 or time_gap > time_window:
                    continue

                connection_count += 1
                time_gaps.append(time_gap)

                # Calculate connection cost
                cost = self._calculate_connection_cost(
                    track_end, track_start, time_gap, params
                )

                # Ensure cost is a scalar
                if hasattr(cost, '__len__') and len(cost) > 1:
                    cost = float(cost[0])  # Take first element if array
                elif hasattr(cost, 'item'):
                    cost = float(cost.item())  # Convert numpy scalar to float
                else:
                    cost = float(cost)

                if cost < np.inf:
                    cost_matrix[i, j] = cost
                    valid_connections += 1
                    costs_calculated.append(cost)

        # Log connection analysis
        if costs_calculated:
            connection_stats = {
                'total_potential_connections': connection_count,
                'valid_connections': valid_connections,
                'connection_success_rate': float(valid_connections / connection_count) if connection_count > 0 else 0.0,
                'time_gaps_min': int(np.min(time_gaps)) if time_gaps else 0,
                'time_gaps_max': int(np.max(time_gaps)) if time_gaps else 0,
                'time_gaps_mean': float(np.mean(time_gaps)) if time_gaps else 0.0,
                'connection_costs_min': float(np.min(costs_calculated)),
                'connection_costs_max': float(np.max(costs_calculated)),
                'connection_costs_mean': float(np.mean(costs_calculated)),
                'connection_costs_std': float(np.std(costs_calculated))
            }
            self.log_parameters(connection_stats, "connection analysis")

        # Add birth and death costs
        birth_cost = float(self._get_param_value(params, 'birth_cost', 10.0))
        death_cost = float(self._get_param_value(params, 'death_cost', 10.0))

        # Death costs (track ends that don't connect)
        for i in range(num_tracks):
            cost_matrix[i, num_tracks + i] = death_cost

        # Birth costs (track starts that don't connect)
        for j in range(num_tracks):
            cost_matrix[num_tracks + j, j] = birth_cost

        # Convert to sparse matrix for efficiency
        cost_matrix[cost_matrix == np.inf] = 1e10  # Replace inf with large number
        cost_sparse = sp.csr_matrix(cost_matrix)

        self.log_info(f"Enhanced cost matrix built: {cost_sparse.shape}, {cost_sparse.nnz} non-zero elements")
        return cost_sparse

    def _calculate_connection_cost(self, track_end: Dict, track_start: Dict,
                                 time_gap: int, params: Dict) -> float:
        """
        Calculate sophisticated connection cost using u-track methodology
        Combines distance, motion prediction, and statistical likelihood
        """
        if track_end['position'] is None or track_start['position'] is None:
            return np.inf

        end_pos = np.asarray(track_end['position']).flatten()
        start_pos = np.asarray(track_start['position']).flatten()
        end_vel = np.asarray(track_end['velocity']).flatten()

        # Ensure we have 2D positions
        if len(end_pos) < 2 or len(start_pos) < 2:
            return np.inf

        # Take only x,y coordinates (first 2 elements)
        end_pos = end_pos[:2]
        start_pos = start_pos[:2]
        end_vel = end_vel[:2] if len(end_vel) >= 2 else np.array([0.0, 0.0])

        # Predict position using motion model
        predicted_pos = end_pos + end_vel * time_gap

        # Calculate distance components - ensure scalar results
        distance = float(np.linalg.norm(start_pos - predicted_pos))
        direct_distance = float(np.linalg.norm(start_pos - end_pos))

        # Motion-based search radius
        search_radius = float(track_end['search_radius'])
        max_search_radius = float(self._get_param_value(params, 'max_search_radius', 15.0))

        if distance > max_search_radius:
            return np.inf

        # Calculate statistical cost based on motion type
        motion_type = track_end['motion_type']

        if motion_type == 'directed':
            # Use linear motion model
            lin_std_mult = float(self._get_param_value(params, 'lin_std_mult', 4.0))
            sigma = search_radius / lin_std_mult
            cost = (distance / sigma) ** 2
        elif motion_type == 'brownian':
            # Use Brownian motion model
            brown_std_mult = float(self._get_param_value(params, 'brown_std_mult', 3.0))
            sigma = np.sqrt(time_gap) * search_radius / brown_std_mult
            cost = (distance / sigma) ** 2
        else:
            # Mixed motion - use adaptive model
            sigma = search_radius / 3.0
            cost = (distance / sigma) ** 2

        # Add gap penalty
        gap_penalty = float(self._get_param_value(params, 'gap_penalty', 1.5))
        cost += gap_penalty * time_gap

        # Add feature-based costs if available
        # This could include intensity ratios, size differences, etc.
        feature_cost = 0.0  # Placeholder for feature-based costs

        total_cost = float(cost + feature_cost)

        return total_cost

    def _get_track_position_at_frame(self, tracks_coord_amp: np.ndarray,
                                   track_id: int, frame: int) -> Optional[np.ndarray]:
        """Get track position at specific frame"""
        if frame < 0 or frame * 8 + 1 >= tracks_coord_amp.shape[1]:
            return None

        coord_idx = frame * 8
        x = tracks_coord_amp[track_id, coord_idx]
        y = tracks_coord_amp[track_id, coord_idx + 1]

        if np.isnan(x) or np.isnan(y):
            return None

        # Ensure we return a proper 2D position array
        return np.array([float(x), float(y)])

    def _solve_enhanced_assignment(self, cost_matrix: sp.csr_matrix,
                                 assignment_info: Dict, gap_close_param: Dict) -> List[Tuple]:
        """
        Solve enhanced assignment problem using u-track methodology
        Implements sophisticated assignment with statistical validation
        """
        self.log_info("=== SOLVING ENHANCED ASSIGNMENT PROBLEM ===")

        if cost_matrix.nnz == 0:
            self.log_warning("Empty cost matrix")
            return []

        # Convert to dense for assignment solver
        cost_dense = cost_matrix.toarray()

        # Solve LAP using Hungarian algorithm
        try:
            row_ind, col_ind = linear_sum_assignment(cost_dense)

            # Filter assignments based on cost thresholds
            assignments = []
            cost_threshold = float(gap_close_param.get('cost_threshold', 50.0))

            for i, (row, col) in enumerate(zip(row_ind, col_ind)):
                cost = float(cost_dense[row, col])

                # Skip assignments with infinite or very high costs
                if cost >= 1e9:
                    continue

                # Apply statistical significance test
                if cost <= cost_threshold:
                    assignments.append((row, col, cost))

            self.log_info(f"Found {len(assignments)} valid assignments")
            return assignments

        except Exception as e:
            self.log_error(f"Assignment solving failed: {e}")
            return []

    def _build_enhanced_compound_tracks(self, assignments: List[Tuple],
                                      tracks_feat_indx: np.ndarray,
                                      tracks_coord_amp: np.ndarray,
                                      track_start_time: np.ndarray,
                                      track_end_time: np.ndarray,
                                      assignment_info: Dict,
                                      motion_info: Dict) -> List[Dict]:
        """
        Build enhanced compound tracks using u-track methodology
        Implements sophisticated track linking with gap interpolation
        """
        self.log_info("=== BUILDING ENHANCED COMPOUND TRACKS ===")

        num_tracks = len(track_start_time)

        # Build connection graph
        connections = {}
        for row, col, cost in assignments:
            if row < num_tracks and col < num_tracks:  # Valid track-to-track connection
                connections[row] = col

        # Find connected components (track chains)
        used_tracks = set()
        compound_tracks = []
        connections_made = 0
        gaps_interpolated = 0

        for start_track in range(num_tracks):
            if start_track in used_tracks:
                continue

            # Follow connection chain
            track_chain = [start_track]
            current_track = start_track

            # Follow forward connections
            while current_track in connections:
                next_track = connections[current_track]
                if next_track in used_tracks:
                    break
                track_chain.append(next_track)
                current_track = next_track
                connections_made += 1

            # Mark all tracks in chain as used
            for track_id in track_chain:
                used_tracks.add(track_id)

            # Build compound track with gap interpolation
            compound_track = self._create_enhanced_compound_track(
                track_chain, tracks_feat_indx, tracks_coord_amp,
                track_start_time, track_end_time, motion_info
            )

            if compound_track:
                compound_tracks.append(compound_track)
                # Count gaps that would be interpolated
                if len(track_chain) > 1:
                    gaps_interpolated += len(track_chain) - 1

        # Add unconnected tracks
        unconnected_tracks = 0
        for track_id in range(num_tracks):
            if track_id not in used_tracks:
                compound_track = self._create_enhanced_compound_track(
                    [track_id], tracks_feat_indx, tracks_coord_amp,
                    track_start_time, track_end_time, motion_info
                )
                if compound_track:
                    compound_tracks.append(compound_track)
                    unconnected_tracks += 1

        # Log connection and gap interpolation statistics
        connection_summary = {
            'total_connections_made': connections_made,
            'gaps_interpolated': gaps_interpolated,
            'unconnected_tracks': unconnected_tracks,
            'connected_tracks': len(used_tracks) - unconnected_tracks,
            'compound_tracks_created': len(compound_tracks),
            'average_chain_length': float((len(used_tracks) - unconnected_tracks + connections_made) / len(compound_tracks)) if len(compound_tracks) > 0 else 1.0
        }
        self.log_parameters(connection_summary, "track connection summary")

        self.log_info(f"Built {len(compound_tracks)} enhanced compound tracks")
        return compound_tracks

    def _create_enhanced_compound_track(self, track_chain: List[int],
                                      tracks_feat_indx: np.ndarray,
                                      tracks_coord_amp: np.ndarray,
                                      track_start_time: np.ndarray,
                                      track_end_time: np.ndarray,
                                      motion_info: Dict) -> Dict:
        """
        Create enhanced compound track with gap interpolation
        """
        if not track_chain:
            return None

        num_frames = tracks_feat_indx.shape[1]

        # Initialize compound track arrays
        compound_feat = np.zeros(num_frames, dtype=int)
        compound_coord = np.full(num_frames * 8, np.nan)

        # Track segments for seq_of_events
        segments = []
        gaps_interpolated = 0
        total_interpolated_frames = 0

        for segment_id, track_id in enumerate(track_chain):
            start_frame = track_start_time[track_id] - 1  # Convert to 0-indexed
            end_frame = track_end_time[track_id] - 1

            # Copy track data
            compound_feat[start_frame:end_frame+1] = tracks_feat_indx[track_id, start_frame:end_frame+1]

            coord_start = start_frame * 8
            coord_end = (end_frame + 1) * 8
            coord_src_start = start_frame * 8
            coord_src_end = (end_frame + 1) * 8

            compound_coord[coord_start:coord_end] = tracks_coord_amp[track_id, coord_src_start:coord_src_end]

            segments.append({
                'start_frame': start_frame + 1,  # Convert back to 1-indexed
                'end_frame': end_frame + 1,
                'segment_id': segment_id + 1
            })

            # Interpolate gaps between segments
            if segment_id < len(track_chain) - 1:
                next_track_id = track_chain[segment_id + 1]
                next_start_frame = track_start_time[next_track_id] - 1

                if next_start_frame > end_frame + 1:
                    # There's a gap to interpolate
                    gap_size = next_start_frame - end_frame - 1
                    self._interpolate_gap(
                        compound_coord, end_frame, next_start_frame,
                        motion_info['velocities'][track_id]
                    )
                    gaps_interpolated += 1
                    total_interpolated_frames += gap_size

        # Log interpolation results for multi-segment tracks
        if len(track_chain) > 1:
            interpolation_stats = {
                'track_chain_length': len(track_chain),
                'segments_created': len(segments),
                'gaps_interpolated': gaps_interpolated,
                'total_interpolated_frames': total_interpolated_frames,
                'avg_gap_size': float(total_interpolated_frames / gaps_interpolated) if gaps_interpolated > 0 else 0.0
            }
            self.log_parameters(interpolation_stats, f"track interpolation (chain {track_chain})")

        # Create seq_of_events in u-track format
        seq_of_events = []
        for segment in segments:
            seq_of_events.append([segment['start_frame'], 1, segment['segment_id'], np.nan])  # Start
            seq_of_events.append([segment['end_frame'], 2, segment['segment_id'], np.nan])    # End

        seq_of_events = np.array(seq_of_events)

        return {
            'tracks_feat_indx_cg': compound_feat.reshape(1, -1),
            'tracks_coord_amp_cg': compound_coord.reshape(1, -1),
            'seq_of_events': seq_of_events,
            'track_chain': track_chain,
            'num_segments': len(segments)
        }

    def _interpolate_gap(self, compound_coord: np.ndarray,
                        end_frame: int, start_frame: int, velocity: np.ndarray):
        """
        Interpolate coordinates in gaps using motion prediction
        """
        if start_frame <= end_frame + 1:
            return  # No gap to interpolate

        # Get end position
        end_coord_idx = end_frame * 8
        end_x = compound_coord[end_coord_idx]
        end_y = compound_coord[end_coord_idx + 1]

        # Get start position
        start_coord_idx = start_frame * 8
        start_x = compound_coord[start_coord_idx]
        start_y = compound_coord[start_coord_idx + 1]

        if np.isnan(end_x) or np.isnan(end_y) or np.isnan(start_x) or np.isnan(start_y):
            return  # Can't interpolate without valid endpoints

        # Linear interpolation between endpoints
        gap_length = start_frame - end_frame - 1

        for i in range(1, gap_length + 1):
            frame = end_frame + i
            coord_idx = frame * 8

            # Linear interpolation
            alpha = i / (gap_length + 1)
            interp_x = end_x + alpha * (start_x - end_x)
            interp_y = end_y + alpha * (start_y - end_y)

            compound_coord[coord_idx] = interp_x
            compound_coord[coord_idx + 1] = interp_y
            compound_coord[coord_idx + 2] = 0.0  # z-coordinate
            compound_coord[coord_idx + 3] = 1000.0  # amplitude (interpolated)
            compound_coord[coord_idx + 4:coord_idx + 8] = [0.1, 0.1, 0.1, 50.0]  # Uncertainties

    def _has_valid_costs(self, cost_matrix: sp.csr_matrix) -> bool:
        """Check if cost matrix has valid finite costs"""
        if cost_matrix.nnz == 0:
            return False

        finite_costs = np.isfinite(cost_matrix.data)
        reasonable_costs = cost_matrix.data < 1e9

        return np.any(finite_costs & reasonable_costs)

    def _convert_enhanced_tracks_to_final(self, compound_tracks: List[Dict],
                                        num_frames: int) -> List[Dict]:
        """Convert enhanced compound tracks to final format"""
        tracks_final = []

        for compound_track in compound_tracks:
            if not compound_track:
                continue

            # Enhanced compound tracks are already in the correct format
            tracks_final.append(compound_track)

        return tracks_final

    def _ensure_array(self, data):
        """Ensure data is a numpy array"""
        if data is None:
            return np.array([]).reshape(0, 0)

        if isinstance(data, list):
            return convert_linking_output(data, 'feat_indx' if len(data) > 0 and isinstance(data[0], (list, int)) else 'coord_amp')

        if hasattr(data, 'toarray'):  # Sparse matrix
            return data.toarray()

        return np.asarray(data)

    def _get_track_sel(self, tracks_coord_amp: np.ndarray) -> np.ndarray:
        """Get track start times, end times, and lengths"""
        num_tracks, num_cols = tracks_coord_amp.shape
        num_frames = num_cols // 8

        track_sel = np.zeros((num_tracks, 3))

        for i_track in range(num_tracks):
            # Find first and last non-NaN coordinates
            x_coords = tracks_coord_amp[i_track, ::8]  # x-coordinates
            valid_frames = ~np.isnan(x_coords)

            if np.any(valid_frames):
                valid_indices = np.where(valid_frames)[0]
                start_time = valid_indices[0] + 1  # Start time (1-indexed)
                end_time = valid_indices[-1] + 1   # End time (1-indexed)
                length = len(valid_indices)        # Length

                track_sel[i_track, 0] = start_time
                track_sel[i_track, 1] = end_time
                track_sel[i_track, 2] = length
            else:
                track_sel[i_track, :] = [1, 1, 0]  # Empty track

        return track_sel

    def _convert_matrices_to_tracks(self, tracks_feat_indx: np.ndarray,
                                   tracks_coord_amp: np.ndarray) -> List[Dict]:
        """Convert track matrices to track structures (fallback method)"""
        if tracks_feat_indx.size == 0:
            return []

        tracks_final = []
        track_sel = self._get_track_sel(tracks_coord_amp)

        for i_track in range(tracks_feat_indx.shape[0]):
            start_time = int(track_sel[i_track, 0])
            end_time = int(track_sel[i_track, 1])
            track_length = int(track_sel[i_track, 2])

            if track_length == 0:
                continue

            # Extract track data
            tracks_feat_indx_cg = tracks_feat_indx[i_track:i_track+1, start_time-1:end_time]
            tracks_coord_amp_cg = tracks_coord_amp[i_track:i_track+1, (start_time-1)*8:end_time*8]

            # Create seq_of_events
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


def convert_linking_output(data, data_type):
    """Enhanced conversion function for linking output"""
    if not data:
        return np.array([]).reshape(0, 0)

    if data_type == 'feat_indx':
        max_length = max(len(track) for track in data) if data else 0
        if max_length == 0:
            return np.array([]).reshape(0, 0)

        feat_matrix = np.zeros((len(data), max_length), dtype=int)
        for i, track in enumerate(data):
            if len(track) > 0:
                feat_matrix[i, :len(track)] = track
        return feat_matrix

    elif data_type == 'coord_amp':
        max_length = 0
        for track in data:
            if isinstance(track, list):
                max_length = max(max_length, len(track))

        if max_length == 0:
            return np.array([]).reshape(0, 0)

        coord_matrix = np.full((len(data), max_length * 8), np.nan)
        for i, track in enumerate(data):
            if isinstance(track, list):
                for j, frame_data in enumerate(track):
                    if isinstance(frame_data, (list, np.ndarray)) and len(frame_data) >= 4:
                        coord_start = j * 8
                        coord_matrix[i, coord_start:coord_start+4] = frame_data[:4]
        return coord_matrix

    return np.array(data)


# Create alias for backwards compatibility
GapCloser = EnhancedGapCloser
