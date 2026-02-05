#!/usr/bin/env python3
"""
Feature Linking for Particle Tracking - ENHANCED WITH COMPREHENSIVE DEBUG LOGGING

Python port of u-track's linking functions for frame-to-frame particle association.

Copyright (C) 2025, Danuser Lab - UTSouthwestern
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment
import time
import sys
import traceback

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

import logging

# Get logger for this module
logger = get_module_logger('linking', log_level=logging.INFO)


class FeatureLinker(LoggingMixin):
    """Frame-to-frame feature linking for particle tracking"""

    def __init__(self):
        """Initialize feature linker"""
        super().__init__()  # This sets up self.logger automatically
        self.log_info("FeatureLinker initialized")

        # NEW DEBUG: Initialize performance tracking
        self.performance_stats = {
            'total_linking_time': 0,
            'cost_matrix_time': 0,
            'assignment_time': 0,
            'conversion_time': 0,
            'frame_pairs_processed': 0,
            'total_assignments_made': 0,
            'total_assignments_rejected': 0
        }
        self.log_debug(f"Performance tracking initialized: {self.performance_stats}")

    def link_features_kalman_sparse(self,
                                   movie_info: List[Dict],
                                   cost_function,
                                   cost_parameters: Dict,
                                   kalman_functions,
                                   prob_dim: int = 2,
                                   kalman_info_prev: Optional[List[Dict]] = None,
                                   linking_costs_prev: Optional[List] = None,
                                   verbose: bool = True) -> Tuple:
        """
        FIXED: Link features between frames using proper cost matrix calculation

        This version actually uses the cost function and respects search radius parameters.
        """
        # NEW DEBUG: System information
        self.log_info(f"=== SYSTEM DEBUG INFO ===")
        self.log_info(f"Python version: {sys.version}")
        self.log_info(f"NumPy version: {np.__version__}")

        # NEW DEBUG: Memory usage tracking
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            self.log_info(f"Initial memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
        except ImportError:
            self.log_debug("psutil not available for memory tracking")

        # Log function call
        log_function_call(self.logger, 'link_features_kalman_sparse',
                         (movie_info, cost_function, cost_parameters, kalman_functions),
                         {'prob_dim': prob_dim, 'verbose': verbose})

        self.log_info("=== STARTING FEATURE LINKING ===")

        # NEW DEBUG: Input validation with detailed checks
        self.log_info("=== INPUT VALIDATION ===")
        input_validation = self._validate_inputs(movie_info, cost_function, cost_parameters, prob_dim)
        self.log_parameters(input_validation, "input validation results")

        if not input_validation['is_valid']:
            self.log_error(f"Input validation failed: {input_validation['errors']}")
            return [], [], [], [], [], 1, None

        # Log input parameters
        self.log_info(f"INPUT PARAMETERS:")
        self.log_info(f"  movie_info: {len(movie_info)} frames")
        self.log_info(f"  cost_function: {cost_function.__name__ if callable(cost_function) else cost_function}")
        self.log_info(f"  prob_dim: {prob_dim}")
        self.log_info(f"  verbose: {verbose}")
        self.log_info(f"  kalman_info_prev: {'Available' if kalman_info_prev else 'None'}")
        self.log_info(f"  linking_costs_prev: {'Available' if linking_costs_prev else 'None'}")

        # Debug cost_parameters in detail
        self.log_info("COST PARAMETERS ANALYSIS:")
        self.log_info(f"  cost_parameters type: {type(cost_parameters)}")

        if hasattr(cost_parameters, '__dict__'):
            self.log_debug("Converting CostMatrixParameters object to dictionary")
            params_dict = cost_parameters.__dict__
            self.log_debug(f"Original object attributes: {list(params_dict.keys())}")
        else:
            params_dict = cost_parameters
            self.log_debug(f"Already dictionary with keys: {list(params_dict.keys()) if isinstance(params_dict, dict) else 'Not a dict'}")

        # Log all parameter values
        if isinstance(params_dict, dict):
            self.log_parameters(params_dict, "detailed parameter values")

        # Extract and validate key parameters
        min_search_radius = params_dict.get('min_search_radius', 2.0)
        max_search_radius = params_dict.get('max_search_radius', 10.0)
        brown_std_mult = params_dict.get('brown_std_mult', 3.0)
        lin_std_mult = params_dict.get('lin_std_mult', [3.0] * 5)
        use_local_density = params_dict.get('use_local_density', False)
        max_angle_vv = params_dict.get('max_angle_vv', 30.0)

        key_params = {
            'min_search_radius': min_search_radius,
            'max_search_radius': max_search_radius,
            'brown_std_mult': brown_std_mult,
            'lin_std_mult': lin_std_mult,
            'use_local_density': use_local_density,
            'max_angle_vv': max_angle_vv
        }
        self.log_parameters(key_params, "key parameter extraction")

        # NEW DEBUG: Enhanced parameter validation
        param_warnings = []
        if min_search_radius >= max_search_radius:
            param_warnings.append(f"min_search_radius ({min_search_radius}) >= max_search_radius ({max_search_radius})")
        if max_search_radius > 100:
            param_warnings.append(f"max_search_radius ({max_search_radius}) is very large")
        if min_search_radius < 0.1:
            param_warnings.append(f"min_search_radius ({min_search_radius}) is very small")
        if not isinstance(brown_std_mult, (int, float)) or brown_std_mult <= 0:
            param_warnings.append(f"Invalid brown_std_mult: {brown_std_mult}")

        for warning in param_warnings:
            self.log_warning(warning)

        try:
            num_frames = len(movie_info)
            self.log_info(f"Processing {num_frames} frames")

            # NEW DEBUG: Enhanced movie_info analysis with data integrity checks
            self.log_info("=== MOVIE_INFO DETAILED ANALYSIS ===")
            movie_analysis = self._analyze_movie_info(movie_info, prob_dim)
            self.log_parameters(movie_analysis, "comprehensive movie analysis")

            # Debug movie_info content
            self.log_info("MOVIE_INFO ANALYSIS:")
            total_detections = 0
            for i, frame_info in enumerate(movie_info):
                frame_detections = frame_info.get('num', 0)
                total_detections += frame_detections
                self.log_debug(f"Frame {i}: {frame_detections} detections")
                if i < 3:  # Show details for first few frames
                    self.log_debug(f"  Keys: {list(frame_info.keys())}")
                    if 'xCoord' in frame_info and len(frame_info['xCoord']) > 0:
                        log_array_info(self.logger, 'xCoord', frame_info['xCoord'], f"frame {i}")
                    if 'yCoord' in frame_info and len(frame_info['yCoord']) > 0:
                        log_array_info(self.logger, 'yCoord', frame_info['yCoord'], f"frame {i}")

            self.log_info(f"Total detections across all frames: {total_detections}")
            self.log_info(f"Average detections per frame: {total_detections/num_frames:.2f}")

            # NEW DEBUG: Detection pattern analysis
            if num_frames > 1:
                detection_changes = []
                for i in range(1, num_frames):
                    prev_count = movie_info[i-1].get('num', 0)
                    curr_count = movie_info[i].get('num', 0)
                    change = curr_count - prev_count
                    detection_changes.append(change)

                detection_stats = {
                    'max_detection_increase': max(detection_changes) if detection_changes else 0,
                    'max_detection_decrease': min(detection_changes) if detection_changes else 0,
                    'avg_detection_change': np.mean(detection_changes) if detection_changes else 0,
                    'detection_stability': np.std(detection_changes) if detection_changes else 0
                }
                self.log_parameters(detection_stats, "detection pattern analysis")

            # Initialize tracking structures
            active_tracks = {}  # track_id -> track_info
            next_track_id = 1
            linking_costs = []
            err_flag = 0

            self.log_info("INITIALIZING TRACKING STRUCTURES")
            self.log_debug(f"Initial active_tracks: {len(active_tracks)}")
            self.log_debug(f"next_track_id: {next_track_id}")

            # NEW DEBUG: Memory check after initialization
            try:
                memory_info = process.memory_info()
                self.log_debug(f"Memory after initialization: {memory_info.rss / 1024 / 1024:.2f} MB")
            except:
                pass

            # Initialize Kalman filter info
            self.log_info("Initializing Kalman filter information")
            kalman_info = []
            for frame_info in movie_info:
                num_features = frame_info.get('num', 0)
                kalman_frame_info = {
                    'num_features': num_features,
                    'state_vec': np.zeros((num_features, 2 * prob_dim)) if num_features > 0 else np.array([]).reshape(0, 2 * prob_dim),
                    'state_noise_var': np.zeros((num_features, 2 * prob_dim, 2 * prob_dim)) if num_features > 0 else np.array([]).reshape(0, 2 * prob_dim, 2 * prob_dim),
                    'observation_vec': np.zeros((num_features, prob_dim)) if num_features > 0 else np.array([]).reshape(0, prob_dim)
                }
                kalman_info.append(kalman_frame_info)

            self.log_info(f"Kalman info initialized for {len(kalman_info)} frames")
            self.log_debug(f"prob_dim used for Kalman: {prob_dim}")
            self.log_debug(f"State vector dimensions: {2 * prob_dim}")

            # NEW DEBUG: Kalman filter memory usage
            kalman_memory_estimate = 0
            for kf_info in kalman_info:
                kalman_memory_estimate += kf_info['state_vec'].nbytes
                kalman_memory_estimate += kf_info['state_noise_var'].nbytes
                kalman_memory_estimate += kf_info['observation_vec'].nbytes
            self.log_debug(f"Kalman filter estimated memory: {kalman_memory_estimate / 1024 / 1024:.2f} MB")

            self.log_info(f"FIXED LINKING: Processing {num_frames} frames using COST MATRIX constraints")

            # FIXED: Convert cost_parameters to dictionary format consistently
            if hasattr(cost_parameters, '__dict__'):
                params = cost_parameters.__dict__
                self.log_debug("Converted CostMatrixParameters object to dictionary")
            else:
                params = cost_parameters
                self.log_debug("Using dictionary parameters directly")

            self.log_parameters(params, "final params for processing")

            # Extract key parameters for validation using the converted params
            min_search_radius = params.get('min_search_radius', 2.0)
            max_search_radius = params.get('max_search_radius', 10.0)
            self.log_info(f"Using search radius constraints: {min_search_radius:.1f} - {max_search_radius:.1f} pixels")

            # Process each frame pair using proper cost matrix
            successful_links = 0
            rejected_links = 0

            # NEW DEBUG: Initialize detailed tracking metrics
            frame_metrics = []
            assignment_quality_metrics = {
                'total_possible_assignments': 0,
                'cost_matrix_assignments': 0,
                'filtered_assignments': 0,
                'distance_rejections': 0,
                'cost_rejections': 0
            }

            with self.time_operation("Complete frame-to-frame linking"):
                for i_frame in range(num_frames - 1):
                    frame_start_time = time.time()
                    self.log_info(f"PROCESSING FRAME PAIR {i_frame} -> {i_frame + 1}")

                    if verbose:
                        self.log_debug(f"Processing frame {i_frame} -> {i_frame + 1} with COST MATRIX")

                    current_frame = movie_info[i_frame]
                    next_frame = movie_info[i_frame + 1]

                    self.log_debug(f"Current frame {i_frame}: {current_frame['num']} detections")
                    self.log_debug(f"Next frame {i_frame + 1}: {next_frame['num']} detections")

                    # NEW DEBUG: Frame pair statistics
                    frame_pair_stats = {
                        'frame_index': i_frame,
                        'current_detections': current_frame['num'],
                        'next_detections': next_frame['num'],
                        'max_possible_links': min(current_frame['num'], next_frame['num']),
                        'total_possible_assignments': current_frame['num'] * next_frame['num']
                    }
                    self.log_parameters(frame_pair_stats, f"frame pair {i_frame} statistics")

                    assignment_quality_metrics['total_possible_assignments'] += frame_pair_stats['total_possible_assignments']

                    if current_frame['num'] == 0 or next_frame['num'] == 0:
                        self.log_warning(f"Skipping frame pair - one or both frames have no detections")

                        # NEW DEBUG: Record empty frame metrics
                        frame_metric = {
                            'frame_pair': f"{i_frame}->{i_frame+1}",
                            'processing_time': time.time() - frame_start_time,
                            'assignments_made': 0,
                            'assignments_rejected': 0,
                            'reason': 'empty_frames'
                        }
                        frame_metrics.append(frame_metric)
                        continue

                    # NEW DEBUG: Coordinate distribution analysis
                    current_coords = self._get_frame_coordinates(current_frame, prob_dim)
                    next_coords = self._get_frame_coordinates(next_frame, prob_dim)

                    if len(current_coords) > 0 and len(next_coords) > 0:
                        coord_analysis = self._analyze_coordinate_distribution(current_coords, next_coords, i_frame)
                        self.log_parameters(coord_analysis, f"coordinate analysis frame {i_frame}")

                    # FIXED: Use actual cost matrix calculation with converted parameters
                    cost_matrix_params = {
                        'prob_dim': prob_dim,
                        'frame_index': i_frame,
                        'min_search_radius': params.get('min_search_radius', 'missing'),
                        'max_search_radius': params.get('max_search_radius', 'missing')
                    }
                    self.log_parameters(cost_matrix_params, "cost matrix calculation")

                    cost_matrix_start_time = time.time()
                    with PerformanceTimer(self.logger, f"Cost matrix calculation for frame {i_frame}"):
                        cost_matrix, assignments = self._calculate_cost_matrix_assignments(
                            [current_frame, next_frame],
                            kalman_info[i_frame],
                            cost_function,
                            params,  # Use converted params instead of original cost_parameters
                            prob_dim,
                            i_frame
                        )
                    cost_matrix_time = time.time() - cost_matrix_start_time
                    self.performance_stats['cost_matrix_time'] += cost_matrix_time

                    # NEW DEBUG: Cost matrix quality analysis
                    if cost_matrix is not None:
                        cost_matrix_analysis = self._analyze_cost_matrix(cost_matrix, i_frame)
                        self.log_parameters(cost_matrix_analysis, f"cost matrix analysis frame {i_frame}")

                    # Process assignments from cost matrix
                    assignment_start_time = time.time()
                    if assignments is not None and len(assignments[0]) > 0:
                        row_ind, col_ind = assignments
                        self.log_info(f"COST MATRIX found {len(row_ind)} valid assignments (respecting search radius)")
                        assignment_quality_metrics['cost_matrix_assignments'] += len(row_ind)

                        if len(row_ind) > 0:
                            sample_pairs = list(zip(row_ind[:5], col_ind[:5]))
                            self.log_debug(f"Assignment pairs sample: {sample_pairs}")

                            # NEW DEBUG: Assignment distance analysis
                            if len(current_coords) > 0 and len(next_coords) > 0:
                                assignment_distances = []
                                for r_idx, c_idx in zip(row_ind, col_ind):
                                    if r_idx < len(current_coords) and c_idx < len(next_coords):
                                        dist = np.linalg.norm(current_coords[r_idx] - next_coords[c_idx])
                                        assignment_distances.append(dist)

                                if assignment_distances:
                                    distance_stats = {
                                        'min_assignment_distance': np.min(assignment_distances),
                                        'max_assignment_distance': np.max(assignment_distances),
                                        'mean_assignment_distance': np.mean(assignment_distances),
                                        'median_assignment_distance': np.median(assignment_distances),
                                        'std_assignment_distance': np.std(assignment_distances)
                                    }
                                    self.log_parameters(distance_stats, f"assignment distance analysis frame {i_frame}")
                    else:
                        row_ind, col_ind = np.array([]), np.array([])
                        self.log_warning("COST MATRIX found NO valid assignments within search radius constraints")

                    assignment_time = time.time() - assignment_start_time
                    self.performance_stats['assignment_time'] += assignment_time

                    # Update active tracks or create new ones
                    used_current = set()
                    used_next = set()
                    frame_successful_links = 0
                    frame_rejected_links = 0

                    self.log_debug(f"Processing {len(row_ind)} assignments for track building")

                    track_update_start_time = time.time()
                    for r_idx, c_idx in zip(row_ind, col_ind):
                        self.log_debug(f"Processing assignment: current[{r_idx}] -> next[{c_idx}]")

                        if r_idx < current_frame['num'] and c_idx < next_frame['num']:
                            # Check if current feature is part of an existing track
                            existing_track = None
                            for track_id, track_info in active_tracks.items():
                                if (track_info['current_frame'] == i_frame and
                                    track_info['current_feature'] == r_idx):
                                    existing_track = track_id
                                    break

                            self.log_debug(f"Existing track for current[{r_idx}]: {existing_track if existing_track else 'None'}")

                            # Get coordinates for validation
                            current_coords = self._get_frame_coordinates(current_frame, prob_dim)
                            next_coords = self._get_frame_coordinates(next_frame, prob_dim)

                            # VALIDATION: Check distance against max search radius
                            if (r_idx < len(current_coords) and c_idx < len(next_coords)):
                                distance = np.linalg.norm(current_coords[r_idx] - next_coords[c_idx])
                                self.log_debug(f"Link distance: {distance:.3f} pixels (max allowed: {max_search_radius:.3f})")

                                if distance > max_search_radius:
                                    self.log_warning(f"REJECTED: Distance {distance:.1f} > max_search_radius {max_search_radius:.1f}")
                                    frame_rejected_links += 1
                                    rejected_links += 1
                                    assignment_quality_metrics['distance_rejections'] += 1
                                    continue
                                else:
                                    self.log_debug("ACCEPTED: Distance within search radius")

                            current_amps = current_frame.get('amp', np.ones((current_frame['num'], 2)))
                            next_amps = next_frame.get('amp', np.ones((next_frame['num'], 2)))

                            current_amp_vals = current_amps[:, 0] if current_amps.ndim > 1 else current_amps
                            next_amp_vals = next_amps[:, 0] if next_amps.ndim > 1 else next_amps

                            current_amp = current_amp_vals[r_idx] if r_idx < len(current_amp_vals) else 1.0
                            next_amp = next_amp_vals[c_idx] if c_idx < len(next_amp_vals) else 1.0

                            self.log_debug(f"Amplitudes: current={current_amp:.1f}, next={next_amp:.1f}")

                            # NEW DEBUG: Amplitude change analysis
                            if current_amp > 0:
                                amp_ratio = next_amp / current_amp
                                if amp_ratio > 3.0 or amp_ratio < 0.33:
                                    self.log_warning(f"Large amplitude change: {current_amp:.1f} -> {next_amp:.1f} (ratio: {amp_ratio:.2f})")

                            if existing_track:
                                # Extend existing track
                                track_info = active_tracks[existing_track]
                                self.log_debug(f"Extending existing track {existing_track}")
                                self.log_debug(f"Previous length: {len(track_info['frames'])} frames")

                                track_info['frames'].append(i_frame + 1)
                                track_info['features'].append(c_idx + 1)  # 1-indexed
                                track_info['coordinates'].append([
                                    next_coords[c_idx, 0],
                                    next_coords[c_idx, 1] if prob_dim > 1 else 0,
                                    next_coords[c_idx, 2] if prob_dim > 2 else 0,
                                    next_amp
                                ])
                                track_info['current_frame'] = i_frame + 1
                                track_info['current_feature'] = c_idx

                                self.log_debug(f"New length: {len(track_info['frames'])} frames")
                                self.log_debug(f"Frame range: {track_info['frames'][0]} - {track_info['frames'][-1]}")

                                # NEW DEBUG: Track velocity analysis
                                if len(track_info['coordinates']) >= 2:
                                    last_coord = np.array(track_info['coordinates'][-2][:prob_dim])
                                    curr_coord = np.array(track_info['coordinates'][-1][:prob_dim])
                                    displacement = np.linalg.norm(curr_coord - last_coord)
                                    self.log_debug(f"Track {existing_track} displacement: {displacement:.3f} pixels")
                            else:
                                # Create new track
                                self.log_debug(f"Creating new track {next_track_id}")

                                new_track = {
                                    'frames': [i_frame, i_frame + 1],
                                    'features': [r_idx + 1, c_idx + 1],  # 1-indexed
                                    'coordinates': [
                                        [current_coords[r_idx, 0],
                                         current_coords[r_idx, 1] if prob_dim > 1 else 0,
                                         current_coords[r_idx, 2] if prob_dim > 2 else 0,
                                         current_amp],
                                        [next_coords[c_idx, 0],
                                         next_coords[c_idx, 1] if prob_dim > 1 else 0,
                                         next_coords[c_idx, 2] if prob_dim > 2 else 0,
                                         next_amp]
                                    ],
                                    'current_frame': i_frame + 1,
                                    'current_feature': c_idx
                                }
                                active_tracks[next_track_id] = new_track
                                self.log_debug(f"Track {next_track_id} spans frames: {new_track['frames']}")
                                self.log_debug(f"Starting coordinates: [{current_coords[r_idx, 0]:.2f}, {current_coords[r_idx, 1] if prob_dim > 1 else 0:.2f}]")
                                next_track_id += 1

                            used_current.add(r_idx)
                            used_next.add(c_idx)
                            frame_successful_links += 1
                            successful_links += 1
                            assignment_quality_metrics['filtered_assignments'] += 1

                            self.log_debug(f"Link successful: total successful = {successful_links}")

                    track_update_time = time.time() - track_update_start_time

                    # NEW DEBUG: Frame processing summary with timing
                    frame_total_time = time.time() - frame_start_time
                    frame_summary = {
                        'frame_pair': f"{i_frame} -> {i_frame + 1}",
                        'successful_links': frame_successful_links,
                        'rejected_links': frame_rejected_links,
                        'active_tracks': len(active_tracks),
                        'processing_time': f"{frame_total_time:.4f}s",
                        'cost_matrix_time': f"{cost_matrix_time:.4f}s",
                        'assignment_time': f"{assignment_time:.4f}s",
                        'track_update_time': f"{track_update_time:.4f}s",
                        'unlinked_current': current_frame['num'] - len(used_current),
                        'unlinked_next': next_frame['num'] - len(used_next)
                    }
                    self.log_parameters(frame_summary, f"frame {i_frame} summary")

                    # Store frame metrics
                    frame_metric = {
                        'frame_pair': f"{i_frame}->{i_frame+1}",
                        'processing_time': frame_total_time,
                        'assignments_made': frame_successful_links,
                        'assignments_rejected': frame_rejected_links,
                        'cost_matrix_time': cost_matrix_time,
                        'assignment_time': assignment_time,
                        'track_update_time': track_update_time
                    }
                    frame_metrics.append(frame_metric)

                    self.performance_stats['frame_pairs_processed'] += 1
                    self.performance_stats['total_assignments_made'] += frame_successful_links
                    self.performance_stats['total_assignments_rejected'] += frame_rejected_links

                    # NEW DEBUG: Memory check during processing
                    if i_frame % 10 == 0:  # Check every 10 frames
                        try:
                            memory_info = process.memory_info()
                            self.log_debug(f"Memory at frame {i_frame}: {memory_info.rss / 1024 / 1024:.2f} MB")
                        except:
                            pass

            # NEW DEBUG: Overall assignment quality analysis
            self.log_info("=== ASSIGNMENT QUALITY ANALYSIS ===")
            self.log_parameters(assignment_quality_metrics, "assignment quality summary")

            if assignment_quality_metrics['total_possible_assignments'] > 0:
                efficiency_metrics = {
                    'assignment_efficiency': f"{100 * assignment_quality_metrics['filtered_assignments'] / assignment_quality_metrics['total_possible_assignments']:.3f}%",
                    'distance_rejection_rate': f"{100 * assignment_quality_metrics['distance_rejections'] / max(1, assignment_quality_metrics['cost_matrix_assignments']):.1f}%",
                    'cost_matrix_efficiency': f"{100 * assignment_quality_metrics['cost_matrix_assignments'] / assignment_quality_metrics['total_possible_assignments']:.3f}%"
                }
                self.log_parameters(efficiency_metrics, "efficiency metrics")

            self.log_info("LINKING COMPLETE")
            linking_summary = {
                'total_successful_links': successful_links,
                'total_rejected_links': rejected_links,
                'total_tracks_created': len(active_tracks),
                'assignment_success_rate': f"{100 * successful_links / max(1, successful_links + rejected_links):.1f}%"
            }
            self.log_parameters(linking_summary, "final linking summary")

            # NEW DEBUG: Track length distribution analysis
            if active_tracks:
                track_lengths = [len(track_info['frames']) for track_info in active_tracks.values()]
                track_length_stats = {
                    'min_track_length': min(track_lengths),
                    'max_track_length': max(track_lengths),
                    'mean_track_length': np.mean(track_lengths),
                    'median_track_length': np.median(track_lengths),
                    'tracks_length_1': sum(1 for length in track_lengths if length == 1),
                    'tracks_length_2': sum(1 for length in track_lengths if length == 2),
                    'tracks_length_3_plus': sum(1 for length in track_lengths if length >= 3)
                }
                self.log_parameters(track_length_stats, "track length distribution")

            # Show sample tracks
            for track_id, track_info in list(active_tracks.items())[:5]:  # Show first 5
                track_details = {
                    'track_id': track_id,
                    'length': len(track_info['frames']),
                    'frame_range': f"{track_info['frames'][0]}-{track_info['frames'][-1]}",
                    'start_coords': track_info['coordinates'][0][:2],
                    'end_coords': track_info['coordinates'][-1][:2]
                }
                self.log_parameters(track_details, f"track {track_id} details")

            # NEW DEBUG: Final performance summary
            self.performance_stats['total_linking_time'] = time.time() - time.time()  # This would need the start time
            self.log_info("=== PERFORMANCE SUMMARY ===")
            self.log_parameters(self.performance_stats, "final performance statistics")

            # Convert to matrix format
            self.log_info("CONVERTING TO MATRIX FORMAT")
            conversion_start_time = time.time()
            with self.time_operation("Track matrix conversion"):
                tracks_feat_indx_link = self._convert_continuous_tracks_to_matrix(active_tracks, num_frames, 'feat_indx')
                tracks_coord_amp_link = self._convert_continuous_tracks_to_matrix(active_tracks, num_frames, 'coord_amp')
                nn_dist_linked_feat = np.ones((len(active_tracks), num_frames))
            conversion_time = time.time() - conversion_start_time
            self.performance_stats['conversion_time'] = conversion_time

            conversion_results = {
                'feat_indx_shape': tracks_feat_indx_link.shape,
                'coord_amp_shape': tracks_coord_amp_link.shape,
                'nn_dist_shape': nn_dist_linked_feat.shape,
                'conversion_time': f"{conversion_time:.4f}s"
            }
            self.log_parameters(conversion_results, "matrix conversion results")

            # NEW DEBUG: Final data integrity checks
            integrity_results = self._validate_output_integrity(
                tracks_feat_indx_link, tracks_coord_amp_link, nn_dist_linked_feat, active_tracks, num_frames
            )
            self.log_parameters(integrity_results, "output integrity validation")

            final_results = {
                'tracks_feat_indx_link': tracks_feat_indx_link.shape,
                'tracks_coord_amp_link': tracks_coord_amp_link.shape,
                'kalman_info_frames': len(kalman_info),
                'nn_dist_linked_feat': nn_dist_linked_feat.shape,
                'linking_costs_entries': len(linking_costs),
                'err_flag': err_flag
            }
            self.log_parameters(final_results, "final linking results")

            # NEW DEBUG: Final memory usage
            try:
                memory_info = process.memory_info()
                self.log_info(f"Final memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
            except:
                pass

            self.log_info("=== FEATURE LINKING COMPLETE ===")

            return (tracks_feat_indx_link, tracks_coord_amp_link, kalman_info,
                   nn_dist_linked_feat, linking_costs, err_flag, None)

        except Exception as e:
            self.log_error(f"ERROR in feature linking: {str(e)}")
            self.log_error(f"Error type: {type(e).__name__}")
            self.log_error(f"Error location: {traceback.format_exc()}")
            self.logger.exception("Full traceback:")

            # NEW DEBUG: Enhanced error context
            error_context = {
                'function': 'link_features_kalman_sparse',
                'num_frames': len(movie_info) if movie_info else 'unknown',
                'prob_dim': prob_dim,
                'cost_function_available': callable(cost_function),
                'params_type': type(cost_parameters).__name__
            }
            self.log_parameters(error_context, "error context")

            return [], [], [], [], [], 1, None

    def _validate_inputs(self, movie_info, cost_function, cost_parameters, prob_dim):
        """NEW DEBUG: Comprehensive input validation"""
        self.log_debug("Starting comprehensive input validation")

        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }

        # Validate movie_info
        if not movie_info:
            validation_result['errors'].append("movie_info is empty or None")
            validation_result['is_valid'] = False
        elif not isinstance(movie_info, list):
            validation_result['errors'].append(f"movie_info must be a list, got {type(movie_info)}")
            validation_result['is_valid'] = False
        else:
            for i, frame_info in enumerate(movie_info):
                if not isinstance(frame_info, dict):
                    validation_result['errors'].append(f"Frame {i} is not a dictionary")
                    validation_result['is_valid'] = False
                elif 'num' not in frame_info:
                    validation_result['warnings'].append(f"Frame {i} missing 'num' field")
                elif frame_info['num'] < 0:
                    validation_result['errors'].append(f"Frame {i} has negative detection count")
                    validation_result['is_valid'] = False

        # Validate prob_dim
        if not isinstance(prob_dim, int) or prob_dim < 2 or prob_dim > 3:
            validation_result['errors'].append(f"prob_dim must be 2 or 3, got {prob_dim}")
            validation_result['is_valid'] = False

        # Validate cost_parameters
        if cost_parameters is None:
            validation_result['warnings'].append("cost_parameters is None")
        elif not isinstance(cost_parameters, (dict, object)):
            validation_result['errors'].append(f"cost_parameters must be dict or object, got {type(cost_parameters)}")
            validation_result['is_valid'] = False

        return validation_result

    def _analyze_movie_info(self, movie_info, prob_dim):
        """NEW DEBUG: Comprehensive movie_info analysis"""
        self.log_debug("Starting comprehensive movie_info analysis")

        analysis = {
            'total_frames': len(movie_info),
            'total_detections': 0,
            'frames_with_detections': 0,
            'empty_frames': 0,
            'coordinate_fields_consistent': True,
            'amplitude_fields_consistent': True,
            'data_integrity_issues': []
        }

        coordinate_fields = ['x_coord', 'y_coord', 'z_coord', 'all_coord', 'xCoord', 'yCoord']
        amplitude_fields = ['amp', 'amplitude', 'intensity']

        for i, frame_info in enumerate(movie_info):
            num_detections = frame_info.get('num', 0)
            analysis['total_detections'] += num_detections

            if num_detections > 0:
                analysis['frames_with_detections'] += 1

                # Check coordinate fields
                coord_field_found = False
                for field in coordinate_fields:
                    if field in frame_info:
                        coord_field_found = True
                        coord_data = frame_info[field]
                        if hasattr(coord_data, '__len__') and len(coord_data) != num_detections:
                            analysis['data_integrity_issues'].append(
                                f"Frame {i}: {field} length ({len(coord_data)}) != num ({num_detections})"
                            )

                if not coord_field_found:
                    analysis['coordinate_fields_consistent'] = False
                    analysis['data_integrity_issues'].append(f"Frame {i}: No coordinate fields found")

                # Check amplitude fields
                amp_field_found = False
                for field in amplitude_fields:
                    if field in frame_info:
                        amp_field_found = True
                        break

                if not amp_field_found:
                    analysis['amplitude_fields_consistent'] = False
            else:
                analysis['empty_frames'] += 1

        if analysis['total_detections'] > 0:
            analysis['avg_detections_per_frame'] = analysis['total_detections'] / len(movie_info)
            analysis['frame_occupancy_rate'] = analysis['frames_with_detections'] / len(movie_info)

        return analysis

    def _analyze_coordinate_distribution(self, current_coords, next_coords, frame_index):
        """NEW DEBUG: Analyze coordinate distribution and spatial patterns"""
        self.log_debug(f"Analyzing coordinate distribution for frame {frame_index}")

        analysis = {}

        # Current frame analysis
        if len(current_coords) > 0:
            current_mean = np.mean(current_coords, axis=0)
            current_std = np.std(current_coords, axis=0)
            current_extent = np.max(current_coords, axis=0) - np.min(current_coords, axis=0)

            analysis.update({
                'current_frame_center': current_mean.tolist(),
                'current_frame_std': current_std.tolist(),
                'current_frame_extent': current_extent.tolist(),
                'current_frame_density': len(current_coords) / (np.prod(current_extent) if np.all(current_extent > 0) else 1)
            })

        # Next frame analysis
        if len(next_coords) > 0:
            next_mean = np.mean(next_coords, axis=0)
            next_std = np.std(next_coords, axis=0)
            next_extent = np.max(next_coords, axis=0) - np.min(next_coords, axis=0)

            analysis.update({
                'next_frame_center': next_mean.tolist(),
                'next_frame_std': next_std.tolist(),
                'next_frame_extent': next_extent.tolist(),
                'next_frame_density': len(next_coords) / (np.prod(next_extent) if np.all(next_extent > 0) else 1)
            })

        # Inter-frame analysis
        if len(current_coords) > 0 and len(next_coords) > 0:
            center_displacement = np.linalg.norm(next_mean - current_mean)

            # Calculate minimum distances between all pairs
            from scipy.spatial.distance import cdist
            all_distances = cdist(current_coords, next_coords)
            min_distances = np.min(all_distances, axis=1)

            analysis.update({
                'center_displacement': center_displacement,
                'min_distance_stats': {
                    'mean': np.mean(min_distances),
                    'std': np.std(min_distances),
                    'min': np.min(min_distances),
                    'max': np.max(min_distances)
                }
            })

        return analysis

    def _analyze_cost_matrix(self, cost_matrix, frame_index):
        """NEW DEBUG: Analyze cost matrix properties and quality"""
        self.log_debug(f"Analyzing cost matrix for frame {frame_index}")

        analysis = {
            'shape': cost_matrix.shape,
            'total_elements': cost_matrix.size
        }

        # Finite value analysis
        finite_mask = np.isfinite(cost_matrix)
        infinite_mask = np.isinf(cost_matrix)
        nan_mask = np.isnan(cost_matrix)

        analysis.update({
            'finite_elements': np.sum(finite_mask),
            'infinite_elements': np.sum(infinite_mask),
            'nan_elements': np.sum(nan_mask),
            'finite_percentage': 100 * np.sum(finite_mask) / cost_matrix.size
        })

        # Statistics of finite values
        if np.any(finite_mask):
            finite_values = cost_matrix[finite_mask]
            analysis.update({
                'finite_value_stats': {
                    'min': np.min(finite_values),
                    'max': np.max(finite_values),
                    'mean': np.mean(finite_values),
                    'std': np.std(finite_values),
                    'median': np.median(finite_values)
                }
            })

            # Cost distribution analysis
            if len(finite_values) > 10:
                percentiles = np.percentile(finite_values, [10, 25, 50, 75, 90])
                analysis['finite_value_percentiles'] = {
                    '10th': percentiles[0],
                    '25th': percentiles[1],
                    '50th': percentiles[2],
                    '75th': percentiles[3],
                    '90th': percentiles[4]
                }

        # Sparsity analysis
        if np.any(finite_mask):
            finite_values = cost_matrix[finite_mask]
            very_high_cost_threshold = np.mean(finite_values) + 3 * np.std(finite_values)
            prohibitive_costs = np.sum(finite_values > very_high_cost_threshold)

            analysis.update({
                'sparsity_analysis': {
                    'prohibitive_cost_threshold': very_high_cost_threshold,
                    'prohibitive_costs': prohibitive_costs,
                    'effective_sparsity': 100 * (cost_matrix.size - np.sum(finite_mask) - prohibitive_costs) / cost_matrix.size
                }
            })

        return analysis

    def _validate_output_integrity(self, feat_matrix, coord_matrix, nn_dist_matrix, active_tracks, num_frames):
        """NEW DEBUG: Validate output data integrity"""
        self.log_debug("Validating output data integrity")

        validation = {
            'feat_matrix_valid': True,
            'coord_matrix_valid': True,
            'nn_dist_matrix_valid': True,
            'matrix_consistency': True,
            'issues_found': []
        }

        # Validate feature matrix
        if feat_matrix.shape[1] != num_frames:
            validation['feat_matrix_valid'] = False
            validation['issues_found'].append(f"Feature matrix has {feat_matrix.shape[1]} columns, expected {num_frames}")

        if feat_matrix.shape[0] != len(active_tracks):
            validation['feat_matrix_valid'] = False
            validation['issues_found'].append(f"Feature matrix has {feat_matrix.shape[0]} rows, expected {len(active_tracks)}")

        # Validate coordinate matrix
        expected_coord_cols = num_frames * 8
        if coord_matrix.shape[1] != expected_coord_cols:
            validation['coord_matrix_valid'] = False
            validation['issues_found'].append(f"Coord matrix has {coord_matrix.shape[1]} columns, expected {expected_coord_cols}")

        if coord_matrix.shape[0] != len(active_tracks):
            validation['coord_matrix_valid'] = False
            validation['issues_found'].append(f"Coord matrix has {coord_matrix.shape[0]} rows, expected {len(active_tracks)}")

        # Check for consistency between matrices
        if feat_matrix.shape[0] != coord_matrix.shape[0]:
            validation['matrix_consistency'] = False
            validation['issues_found'].append("Feature and coordinate matrices have different number of tracks")

        # Validate that non-zero feature indices correspond to finite coordinates
        consistency_errors = 0
        for track_idx in range(min(feat_matrix.shape[0], coord_matrix.shape[0])):
            for frame_idx in range(num_frames):
                feat_val = feat_matrix[track_idx, frame_idx]
                coord_start = frame_idx * 8
                coord_vals = coord_matrix[track_idx, coord_start:coord_start+4]

                if feat_val > 0 and np.all(np.isnan(coord_vals[:2])):  # Feature present but no coordinates
                    consistency_errors += 1
                elif feat_val == 0 and not np.all(np.isnan(coord_vals[:2])):  # No feature but coordinates present
                    consistency_errors += 1

        if consistency_errors > 0:
            validation['matrix_consistency'] = False
            validation['issues_found'].append(f"Found {consistency_errors} feature/coordinate inconsistencies")

        # Overall statistics
        validation.update({
            'total_tracks': len(active_tracks),
            'non_zero_features': np.count_nonzero(feat_matrix),
            'finite_coordinates': np.sum(np.isfinite(coord_matrix)),
            'feature_density': np.count_nonzero(feat_matrix) / feat_matrix.size if feat_matrix.size > 0 else 0,
            'coordinate_density': np.sum(np.isfinite(coord_matrix)) / coord_matrix.size if coord_matrix.size > 0 else 0
        })

        return validation

    def _calculate_cost_matrix_assignments(self, movie_info_pair, kalman_info, cost_function,
                                         cost_parameters, prob_dim, current_frame):
        """
        FIXED: Calculate cost matrix and get assignments that respect search radius
        Note: cost_parameters is now guaranteed to be a dictionary
        """
        log_function_call(self.logger, '_calculate_cost_matrix_assignments',
                         (movie_info_pair, kalman_info, cost_function, cost_parameters),
                         {'prob_dim': prob_dim, 'current_frame': current_frame})

        self.log_info(f"STARTING COST MATRIX CALCULATION for frame pair {current_frame} -> {current_frame + 1}")

        calculation_inputs = {
            'frame_pair': f"{current_frame} -> {current_frame + 1}",
            'prob_dim': prob_dim,
            'cost_parameters_type': type(cost_parameters),
            'cost_function': cost_function.__name__ if callable(cost_function) else str(cost_function)
        }
        self.log_parameters(calculation_inputs, "cost matrix calculation inputs")

        # Debug cost_parameters
        if isinstance(cost_parameters, dict):
            key_search_params = {
                'min_search_radius': cost_parameters.get('min_search_radius', 'missing'),
                'max_search_radius': cost_parameters.get('max_search_radius', 'missing'),
                'brown_std_mult': cost_parameters.get('brown_std_mult', 'missing'),
                'lin_std_mult': cost_parameters.get('lin_std_mult', 'missing')
            }
            self.log_parameters(key_search_params, "key search parameters")
        else:
            self.log_warning(f"cost_parameters is not a dictionary: {cost_parameters}")

        # Debug movie_info_pair
        if len(movie_info_pair) >= 2:
            frame1, frame2 = movie_info_pair
            frame_info = {
                'frame1_detections': frame1.get('num', 0),
                'frame2_detections': frame2.get('num', 0),
                'frame1_keys': list(frame1.keys()),
                'frame2_keys': list(frame2.keys())
            }
            self.log_parameters(frame_info, "frame pair analysis")

            # NEW DEBUG: Check for coordinate data availability
            coord_availability = {}
            for frame_name, frame_data in [('frame1', frame1), ('frame2', frame2)]:
                coord_fields = ['x_coord', 'y_coord', 'all_coord', 'xCoord', 'yCoord']
                available_coords = [field for field in coord_fields if field in frame_data]
                coord_availability[f'{frame_name}_coord_fields'] = available_coords

                if available_coords and frame_data.get('num', 0) > 0:
                    first_coord_field = available_coords[0]
                    coord_data = frame_data[first_coord_field]
                    if hasattr(coord_data, 'shape'):
                        coord_availability[f'{frame_name}_coord_shape'] = coord_data.shape
                    elif hasattr(coord_data, '__len__'):
                        coord_availability[f'{frame_name}_coord_length'] = len(coord_data)

            self.log_parameters(coord_availability, "coordinate data availability")
        else:
            self.log_warning(f"movie_info_pair has unexpected length: {len(movie_info_pair)}")

        # Debug kalman_info
        if isinstance(kalman_info, dict):
            kalman_summary = {
                'kalman_info_keys': list(kalman_info.keys()),
                'num_features': kalman_info.get('num_features', 'missing')
            }
            self.log_parameters(kalman_summary, "kalman info analysis")
        else:
            self.log_debug(f"kalman_info: {kalman_info}")

        try:
            # Use the actual cost function if available
            if cost_function and callable(cost_function):
                self.log_info(f"Using cost function: {cost_function.__name__}")
                self.log_debug(f"Passing cost_parameters as: {type(cost_parameters)}")

                # NEW DEBUG: Pre-cost function validation
                try:
                    cost_function_args = {
                        'movie_info_pair_length': len(movie_info_pair),
                        'kalman_info_type': type(kalman_info),
                        'cost_parameters_keys': list(cost_parameters.keys()) if isinstance(cost_parameters, dict) else 'not_dict',
                        'prob_dim': prob_dim,
                        'current_frame': current_frame
                    }
                    self.log_parameters(cost_function_args, "cost function arguments validation")
                except Exception as e:
                    self.log_warning(f"Error in pre-cost function validation: {e}")

                with PerformanceTimer(self.logger, "Cost function execution"):
                    cost_result = cost_function(
                        movie_info_pair,
                        kalman_info,
                        cost_parameters,  # Now a dictionary
                        np.array([]),  # nn_dist_features
                        prob_dim,
                        None,  # prev_cost
                        np.array([]),  # feat_lifetime
                        None,  # tracked_feature_indx
                        current_frame
                    )

                self.log_debug(f"Cost function returned: {type(cost_result)}")
                self.log_debug(f"Cost result length: {len(cost_result) if hasattr(cost_result, '__len__') else 'scalar'}")

                # NEW DEBUG: Detailed cost function result analysis
                if hasattr(cost_result, '__len__'):
                    cost_result_breakdown = {}
                    for i, item in enumerate(cost_result):
                        if hasattr(item, 'shape'):
                            cost_result_breakdown[f'item_{i}_shape'] = item.shape
                            cost_result_breakdown[f'item_{i}_type'] = type(item).__name__
                        else:
                            cost_result_breakdown[f'item_{i}'] = item
                    self.log_parameters(cost_result_breakdown, "cost function result breakdown")

                if len(cost_result) >= 4:
                    cost_matrix, _, _, nonlink_marker, err_flag = cost_result[:5]

                    cost_result_info = {
                        'cost_matrix_shape': cost_matrix.shape if hasattr(cost_matrix, 'shape') else type(cost_matrix),
                        'nonlink_marker': nonlink_marker,
                        'err_flag': err_flag
                    }
                    self.log_parameters(cost_result_info, "cost function results")

                    if err_flag != 0:
                        self.log_error(f"Cost function returned error flag: {err_flag}")
                        return None, (np.array([]), np.array([]))

                    if cost_matrix.size == 0:
                        self.log_error("Cost function returned empty matrix")
                        return None, (np.array([]), np.array([]))

                    self.log_info("Cost matrix successfully computed")

                    # Debug cost matrix statistics
                    if hasattr(cost_matrix, 'shape') and cost_matrix.size > 0:
                        log_array_info(self.logger, 'cost_matrix', cost_matrix, "computed cost matrix")

                        finite_costs = np.isfinite(cost_matrix)
                        if np.sum(finite_costs) > 0:
                            finite_values = cost_matrix[finite_costs]
                            cost_stats = {
                                'finite_values': f"{np.sum(finite_costs)}/{cost_matrix.size}",
                                'min_finite_cost': np.min(finite_values),
                                'max_finite_cost': np.max(finite_values),
                                'mean_finite_cost': np.mean(finite_values)
                            }
                            self.log_parameters(cost_stats, "cost matrix statistics")

                    # Apply assignment algorithm to cost matrix
                    with PerformanceTimer(self.logger, "Assignment problem solving"):
                        assignments = self._solve_assignment_problem(cost_matrix, nonlink_marker)

                    if assignments[0].size > 0:
                        self.log_info(f"Assignment successful: {len(assignments[0])} assignments")
                        sample_assignments = list(zip(assignments[0][:5], assignments[1][:5]))
                        self.log_debug(f"Assignment pairs (first 5): {sample_assignments}")

                        # NEW DEBUG: Assignment cost analysis
                        if len(assignments[0]) > 0:
                            assignment_costs = []
                            for r, c in zip(assignments[0], assignments[1]):
                                if (r < cost_matrix.shape[0] and c < cost_matrix.shape[1]):
                                    cost = cost_matrix[r, c]
                                    if np.isfinite(cost):
                                        assignment_costs.append(cost)

                            if assignment_costs:
                                assignment_cost_stats = {
                                    'min_assignment_cost': np.min(assignment_costs),
                                    'max_assignment_cost': np.max(assignment_costs),
                                    'mean_assignment_cost': np.mean(assignment_costs),
                                    'assignment_cost_std': np.std(assignment_costs)
                                }
                                self.log_parameters(assignment_cost_stats, "assignment cost analysis")
                    else:
                        self.log_warning("No valid assignments found")

                    return cost_matrix, assignments
                else:
                    self.log_error(f"Cost function returned unexpected result format (length {len(cost_result)})")
                    return None, (np.array([]), np.array([]))
            else:
                self.log_warning("No cost function available, using fallback distance method")
                # Fallback to distance-based method with explicit search radius constraint
                return self._fallback_distance_assignment(movie_info_pair, cost_parameters, prob_dim)

        except Exception as e:
            self.log_error(f"Error in cost matrix calculation: {str(e)}")
            self.log_error(f"Error type: {type(e).__name__}")
            self.log_error(f"Error traceback: {traceback.format_exc()}")
            self.logger.exception("Full traceback:")

            # NEW DEBUG: Enhanced error context for cost matrix calculation
            error_context = {
                'cost_function_callable': callable(cost_function),
                'cost_parameters_type': type(cost_parameters).__name__,
                'movie_info_pair_length': len(movie_info_pair) if movie_info_pair else 0,
                'prob_dim': prob_dim,
                'current_frame': current_frame
            }
            self.log_parameters(error_context, "cost matrix calculation error context")

            return None, (np.array([]), np.array([]))

    def _fallback_distance_assignment(self, movie_info_pair, cost_parameters, prob_dim):
        """
        Fallback method using distance with explicit search radius constraints
        Note: cost_parameters is guaranteed to be a dictionary
        """
        log_function_call(self.logger, '_fallback_distance_assignment',
                         (movie_info_pair, cost_parameters), {'prob_dim': prob_dim})

        self.log_info("STARTING FALLBACK DISTANCE ASSIGNMENT")

        fallback_inputs = {
            'cost_parameters_type': type(cost_parameters),
            'prob_dim': prob_dim
        }
        self.log_parameters(fallback_inputs, "fallback assignment inputs")

        try:
            current_frame, next_frame = movie_info_pair
            frame_detections = {
                'current_frame_detections': current_frame.get('num', 0),
                'next_frame_detections': next_frame.get('num', 0)
            }
            self.log_parameters(frame_detections, "frame detection counts")

            # Extract search radius parameters (now from dictionary)
            max_search_radius = cost_parameters.get('max_search_radius', 10.0)
            min_search_radius = cost_parameters.get('min_search_radius', 0.1)

            radius_params = {
                'max_search_radius': f"{max_search_radius} pixels",
                'min_search_radius': f"{min_search_radius} pixels"
            }
            self.log_parameters(radius_params, "search radius parameters")

            # Validate parameters
            if max_search_radius <= min_search_radius:
                self.log_warning("max_search_radius <= min_search_radius")

            self.log_info(f"FALLBACK: Using distance method with radius {min_search_radius:.1f}-{max_search_radius:.1f}")

            # Get coordinates
            self.log_debug("Extracting coordinates...")
            current_coords = self._get_frame_coordinates(current_frame, prob_dim)
            next_coords = self._get_frame_coordinates(next_frame, prob_dim)

            log_array_info(self.logger, 'current_coords', current_coords, "current frame")
            log_array_info(self.logger, 'next_coords', next_coords, "next frame")

            if len(current_coords) == 0 or len(next_coords) == 0:
                self.log_error("No coordinates available for assignment")
                return None, (np.array([]), np.array([]))

            # NEW DEBUG: Coordinate quality checks
            coord_quality = self._check_coordinate_quality(current_coords, next_coords)
            self.log_parameters(coord_quality, "coordinate quality analysis")

            # Calculate distance matrix
            with PerformanceTimer(self.logger, "Distance matrix calculation"):
                from scipy.spatial.distance import cdist
                dist_matrix = cdist(current_coords, next_coords)

            log_array_info(self.logger, 'dist_matrix', dist_matrix, "calculated distances")

            # NEW DEBUG: Distance matrix analysis
            distance_analysis = {
                'min_distance': np.min(dist_matrix),
                'max_distance': np.max(dist_matrix),
                'mean_distance': np.mean(dist_matrix),
                'median_distance': np.median(dist_matrix),
                'distance_std': np.std(dist_matrix)
            }
            self.log_parameters(distance_analysis, "distance matrix analysis")

            # FIXED: Apply search radius constraint
            self.log_info("Applying search radius constraints...")
            cost_matrix = dist_matrix.copy()

            # Count distances within radius before filtering
            within_radius = dist_matrix <= max_search_radius
            valid_links_count = np.sum(within_radius)

            radius_stats = {
                'distances_within_radius': f"{valid_links_count}/{dist_matrix.size}",
                'percentage_within_radius': f"{100*valid_links_count/dist_matrix.size:.1f}%"
            }
            self.log_parameters(radius_stats, "radius constraint analysis")

            cost_matrix[dist_matrix > max_search_radius] = 1e10  # Prohibitive cost

            # Count valid links
            valid_links = np.sum(dist_matrix <= max_search_radius)
            self.log_info(f"FALLBACK: {valid_links} potential links within search radius")

            if valid_links == 0:
                self.log_error("No valid links within search radius")
                return cost_matrix, (np.array([]), np.array([]))

            # NEW DEBUG: Pre-assignment cost matrix analysis
            cost_matrix_pre_assignment = {
                'prohibitive_costs': np.sum(cost_matrix >= 1e10),
                'valid_costs': np.sum(cost_matrix < 1e10),
                'cost_matrix_shape': cost_matrix.shape
            }
            self.log_parameters(cost_matrix_pre_assignment, "pre-assignment cost matrix")

            # Solve assignment with constraints
            with PerformanceTimer(self.logger, "Assignment problem solving"):
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assignment_info = {
                'initial_assignment_pairs': len(row_ind),
                'sample_pairs': list(zip(row_ind[:5], col_ind[:5]))
            }
            self.log_parameters(assignment_info, "initial assignment results")

            # Filter out assignments that exceed search radius
            self.log_info("Filtering assignments by search radius...")
            valid_assignments = []
            rejected_distance = 0
            rejected_cost = 0

            for r, c in zip(row_ind, col_ind):
                distance = dist_matrix[r, c]
                cost = cost_matrix[r, c]

                if distance <= max_search_radius and cost < 1e9:
                    valid_assignments.append((r, c))
                    self.log_debug(f"Assignment ({r}, {c}): distance {distance:.3f} <= {max_search_radius:.3f} - ACCEPTED")
                else:
                    if distance > max_search_radius:
                        rejected_distance += 1
                        self.log_debug(f"Assignment ({r}, {c}): distance {distance:.3f} > {max_search_radius:.3f} - REJECTED (distance)")
                    else:
                        rejected_cost += 1
                        self.log_debug(f"Assignment ({r}, {c}): cost {cost} too high - REJECTED (cost)")

            # NEW DEBUG: Assignment filtering summary
            filtering_summary = {
                'initial_assignments': len(row_ind),
                'valid_assignments': len(valid_assignments),
                'rejected_by_distance': rejected_distance,
                'rejected_by_cost': rejected_cost,
                'acceptance_rate': f"{100 * len(valid_assignments) / max(1, len(row_ind)):.1f}%"
            }
            self.log_parameters(filtering_summary, "assignment filtering summary")

            self.log_info(f"Final valid assignments: {len(valid_assignments)}")

            if valid_assignments:
                valid_rows, valid_cols = zip(*valid_assignments)
                final_assignments = (np.array(valid_rows), np.array(valid_cols))
                self.log_info(f"Returning {len(valid_rows)} valid assignments")

                # NEW DEBUG: Final assignment statistics
                if len(valid_assignments) > 0:
                    final_distances = [dist_matrix[r, c] for r, c in valid_assignments]
                    final_assignment_stats = {
                        'min_final_distance': np.min(final_distances),
                        'max_final_distance': np.max(final_distances),
                        'mean_final_distance': np.mean(final_distances),
                        'median_final_distance': np.median(final_distances)
                    }
                    self.log_parameters(final_assignment_stats, "final assignment distance statistics")

                return cost_matrix, final_assignments
            else:
                self.log_warning("No assignments passed distance filter")
                return cost_matrix, (np.array([]), np.array([]))

        except Exception as e:
            self.log_error(f"Error in fallback method: {str(e)}")
            self.log_error(f"Error type: {type(e).__name__}")
            self.log_error(f"Error traceback: {traceback.format_exc()}")
            self.logger.exception("Full traceback:")

            # NEW DEBUG: Enhanced error context for fallback method
            fallback_error_context = {
                'movie_info_pair_available': movie_info_pair is not None,
                'cost_parameters_available': cost_parameters is not None,
                'prob_dim': prob_dim
            }
            if movie_info_pair:
                fallback_error_context.update({
                    'current_frame_num': movie_info_pair[0].get('num', 'unknown'),
                    'next_frame_num': movie_info_pair[1].get('num', 'unknown')
                })
            self.log_parameters(fallback_error_context, "fallback method error context")

            return None, (np.array([]), np.array([]))

    def _check_coordinate_quality(self, current_coords, next_coords):
        """NEW DEBUG: Check coordinate data quality"""
        self.log_debug("Checking coordinate quality")

        quality = {
            'current_coords_valid': True,
            'next_coords_valid': True,
            'issues_found': []
        }

        # Check for NaN or infinite values
        if np.any(~np.isfinite(current_coords)):
            quality['current_coords_valid'] = False
            quality['issues_found'].append("Current coordinates contain NaN or infinite values")
            nan_count = np.sum(np.isnan(current_coords))
            inf_count = np.sum(np.isinf(current_coords))
            quality['current_nan_count'] = nan_count
            quality['current_inf_count'] = inf_count

        if np.any(~np.isfinite(next_coords)):
            quality['next_coords_valid'] = False
            quality['issues_found'].append("Next coordinates contain NaN or infinite values")
            nan_count = np.sum(np.isnan(next_coords))
            inf_count = np.sum(np.isinf(next_coords))
            quality['next_nan_count'] = nan_count
            quality['next_inf_count'] = inf_count

        # Check for reasonable coordinate ranges
        if len(current_coords) > 0:
            current_ranges = np.max(current_coords, axis=0) - np.min(current_coords, axis=0)
            quality['current_coord_ranges'] = current_ranges.tolist()
            if np.any(current_ranges > 10000):
                quality['issues_found'].append("Current coordinates have very large range (>10000 pixels)")

        if len(next_coords) > 0:
            next_ranges = np.max(next_coords, axis=0) - np.min(next_coords, axis=0)
            quality['next_coord_ranges'] = next_ranges.tolist()
            if np.any(next_ranges > 10000):
                quality['issues_found'].append("Next coordinates have very large range (>10000 pixels)")

        # Check for duplicate coordinates
        if len(current_coords) > 1:
            from scipy.spatial.distance import pdist
            current_distances = pdist(current_coords)
            duplicate_threshold = 1e-6
            duplicates = np.sum(current_distances < duplicate_threshold)
            if duplicates > 0:
                quality['current_duplicates'] = duplicates
                quality['issues_found'].append(f"Found {duplicates} duplicate current coordinates")

        if len(next_coords) > 1:
            next_distances = pdist(next_coords)
            duplicates = np.sum(next_distances < duplicate_threshold)
            if duplicates > 0:
                quality['next_duplicates'] = duplicates
                quality['issues_found'].append(f"Found {duplicates} duplicate next coordinates")

        return quality

    def _convert_continuous_tracks_to_matrix(self, active_tracks, num_frames, data_type):
        """Convert continuous tracks to matrix format"""
        log_function_call(self.logger, '_convert_continuous_tracks_to_matrix',
                         (active_tracks, num_frames, data_type))

        self.log_info("CONVERTING TRACKS TO MATRIX FORMAT")

        conversion_inputs = {
            'input_tracks': len(active_tracks),
            'num_frames': num_frames,
            'data_type': data_type
        }
        self.log_parameters(conversion_inputs, "track conversion inputs")

        if not active_tracks:
            self.log_warning("No active tracks to convert")
            if data_type == 'feat_indx':
                result = np.array([]).reshape(0, num_frames)
            else:
                result = np.array([]).reshape(0, num_frames * 8)
            self.log_info(f"Returning empty array: {result.shape}")
            return result

        num_tracks = len(active_tracks)
        self.log_info(f"Converting {num_tracks} tracks")

        # NEW DEBUG: Analyze track properties before conversion
        track_properties = self._analyze_track_properties(active_tracks, num_frames)
        self.log_parameters(track_properties, "track properties analysis")

        if data_type == 'feat_indx':
            self.log_debug("Creating feature index matrix...")
            feat_matrix = np.zeros((num_tracks, num_frames), dtype=int)

            conversion_stats = {
                'total_assignments': 0,
                'out_of_bounds_frames': 0,
                'invalid_features': 0
            }

            for i, (track_id, track_info) in enumerate(active_tracks.items()):
                track_frames = track_info['frames']
                track_features = track_info['features']
                self.log_debug(f"Track {track_id}: {len(track_frames)} frames, features: {track_features}")

                for frame_idx, feat_id in zip(track_frames, track_features):
                    if 0 <= frame_idx < num_frames:
                        if isinstance(feat_id, (int, np.integer)) and feat_id > 0:
                            feat_matrix[i, frame_idx] = feat_id
                            conversion_stats['total_assignments'] += 1
                        else:
                            conversion_stats['invalid_features'] += 1
                            self.log_warning(f"Invalid feature ID: {feat_id} for track {track_id}")
                    else:
                        conversion_stats['out_of_bounds_frames'] += 1
                        self.log_warning(f"Frame index {frame_idx} out of bounds [0, {num_frames})")

                    if i < 3:  # Debug first few tracks
                        self.log_debug(f"Frame {frame_idx}: feature {feat_id}")

            self.log_parameters(conversion_stats, "feature matrix conversion stats")
            self.log_info(f"Feature matrix created: {feat_matrix.shape}")
            self.log_info(f"Non-zero entries: {np.count_nonzero(feat_matrix)}")
            return feat_matrix

        else:  # coord_amp
            self.log_debug("Creating coordinate/amplitude matrix...")
            coord_matrix = np.full((num_tracks, num_frames * 8), np.nan)
            self.log_debug(f"Matrix shape: {coord_matrix.shape}")

            conversion_stats = {
                'total_coord_assignments': 0,
                'out_of_bounds_frames': 0,
                'invalid_coordinates': 0,
                'nan_coordinates': 0
            }

            for i, (track_id, track_info) in enumerate(active_tracks.items()):
                track_frames = track_info['frames']
                track_coords = track_info['coordinates']
                self.log_debug(f"Track {track_id}: {len(track_frames)} frames")

                for frame_idx, coords in zip(track_frames, track_coords):
                    if 0 <= frame_idx < num_frames:
                        coord_start = frame_idx * 8
                        coord_end = coord_start + 4

                        # Validate coordinates
                        if len(coords) >= 4:
                            coord_vals = coords[:4]
                            if all(isinstance(val, (int, float, np.number)) for val in coord_vals):
                                if all(np.isfinite(val) for val in coord_vals):
                                    coord_matrix[i, coord_start:coord_end] = coord_vals
                                    conversion_stats['total_coord_assignments'] += 1
                                else:
                                    conversion_stats['nan_coordinates'] += 1
                                    self.log_warning(f"Non-finite coordinates for track {track_id}: {coord_vals}")
                            else:
                                conversion_stats['invalid_coordinates'] += 1
                                self.log_warning(f"Invalid coordinate types for track {track_id}: {coords}")
                        else:
                            conversion_stats['invalid_coordinates'] += 1
                            self.log_warning(f"Insufficient coordinates for track {track_id}: {coords}")
                    else:
                        conversion_stats['out_of_bounds_frames'] += 1
                        self.log_warning(f"Frame index {frame_idx} out of bounds [0, {num_frames})")

                    if i < 3:  # Debug first few tracks
                        self.log_debug(f"Frame {frame_idx}: coords {coords[:4]} at indices {coord_start}:{coord_end}")

            self.log_parameters(conversion_stats, "coordinate matrix conversion stats")

            finite_entries = np.isfinite(coord_matrix)
            coord_stats = {
                'coordinate_matrix_shape': coord_matrix.shape,
                'finite_entries': f"{np.sum(finite_entries)}/{coord_matrix.size}",
                'finite_percentage': f"{100 * np.sum(finite_entries) / coord_matrix.size:.2f}%"
            }
            self.log_parameters(coord_stats, "coordinate matrix results")
            return coord_matrix

    def _analyze_track_properties(self, active_tracks, num_frames):
        """NEW DEBUG: Analyze properties of active tracks"""
        self.log_debug("Analyzing track properties")

        if not active_tracks:
            return {'no_tracks': True}

        track_lengths = []
        frame_spans = []
        coordinate_counts = []

        for track_id, track_info in active_tracks.items():
            track_length = len(track_info.get('frames', []))
            track_lengths.append(track_length)

            frames = track_info.get('frames', [])
            if frames:
                frame_span = max(frames) - min(frames) + 1
                frame_spans.append(frame_span)

            coords = track_info.get('coordinates', [])
            coordinate_counts.append(len(coords))

        properties = {
            'total_tracks': len(active_tracks),
            'track_length_stats': {
                'min': min(track_lengths) if track_lengths else 0,
                'max': max(track_lengths) if track_lengths else 0,
                'mean': np.mean(track_lengths) if track_lengths else 0,
                'median': np.median(track_lengths) if track_lengths else 0
            },
            'frame_span_stats': {
                'min': min(frame_spans) if frame_spans else 0,
                'max': max(frame_spans) if frame_spans else 0,
                'mean': np.mean(frame_spans) if frame_spans else 0
            },
            'tracks_by_length': {}
        }

        # Count tracks by length
        for length in set(track_lengths):
            properties['tracks_by_length'][f'length_{length}'] = track_lengths.count(length)

        # Check for data consistency
        inconsistencies = 0
        for track_id, track_info in active_tracks.items():
            frames = track_info.get('frames', [])
            features = track_info.get('features', [])
            coordinates = track_info.get('coordinates', [])

            if not (len(frames) == len(features) == len(coordinates)):
                inconsistencies += 1

        properties['data_inconsistencies'] = inconsistencies

        return properties

    def _solve_assignment_problem(self, cost_matrix, nonlink_marker):
        """
        Solve assignment problem from cost matrix
        """
        log_function_call(self.logger, '_solve_assignment_problem',
                         (cost_matrix, nonlink_marker))

        self.log_info("SOLVING ASSIGNMENT PROBLEM")

        assignment_inputs = {
            'cost_matrix_shape': cost_matrix.shape,
            'nonlink_marker': nonlink_marker
        }
        self.log_parameters(assignment_inputs, "assignment problem inputs")

        try:
            # Handle augmented cost matrix (includes birth/death costs)
            if cost_matrix.shape[0] == cost_matrix.shape[1]:
                # Square matrix - likely augmented with birth/death costs
                frame1_size = cost_matrix.shape[0] // 2  # Estimate original frame sizes
                frame2_size = cost_matrix.shape[1] // 2

                estimated_sizes = {
                    'estimated_frame1_size': frame1_size,
                    'estimated_frame2_size': frame2_size
                }
                self.log_parameters(estimated_sizes, "square matrix analysis")

                # Extract the actual linking costs (upper-left portion)
                if frame1_size > 0 and frame2_size > 0:
                    linking_costs = cost_matrix[:frame1_size, :frame2_size]
                    self.log_debug(f"Extracted linking costs: {linking_costs.shape}")
                else:
                    linking_costs = cost_matrix
                    self.log_debug("Using full matrix as linking costs")
            else:
                linking_costs = cost_matrix
                self.log_debug(f"Non-square matrix, using as-is: {linking_costs.shape}")

            # Replace nonlink markers with large values
            if nonlink_marker is not None:
                self.log_debug(f"Replacing nonlink markers ({nonlink_marker}) with large values...")
                linking_costs = linking_costs.copy()
                nonlink_count = np.sum(linking_costs == nonlink_marker)
                self.log_debug(f"Found {nonlink_count} nonlink markers")
                linking_costs[linking_costs == nonlink_marker] = 1e10

            # Debug cost matrix statistics
            log_array_info(self.logger, 'linking_costs', linking_costs, "final assignment matrix")

            finite_costs = np.isfinite(linking_costs)
            cost_stats = {
                'total_elements': linking_costs.size,
                'finite_elements': np.sum(finite_costs),
                'infinite_elements': np.sum(np.isinf(linking_costs)),
                'nan_elements': np.sum(np.isnan(linking_costs))
            }
            self.log_parameters(cost_stats, "cost matrix element analysis")

            if np.sum(finite_costs) > 0:
                finite_values = linking_costs[finite_costs]
                finite_stats = {
                    'min_finite_cost': np.min(finite_values),
                    'max_finite_cost': np.max(finite_values),
                    'mean_finite_cost': np.mean(finite_values),
                    'std_finite_cost': np.std(finite_values)  # NEW DEBUG: Added std
                }
                self.log_parameters(finite_stats, "finite cost statistics")

            # Only proceed if we have finite costs
            if not np.any(finite_costs):
                self.log_error("No finite costs in matrix, no assignments possible")
                return np.array([]), np.array([])

            # NEW DEBUG: Pre-assignment analysis
            pre_assignment_analysis = {
                'matrix_condition_number': np.linalg.cond(linking_costs) if linking_costs.shape[0] == linking_costs.shape[1] else 'N/A',
                'matrix_rank': np.linalg.matrix_rank(linking_costs),
                'matrix_determinant': np.linalg.det(linking_costs) if linking_costs.shape[0] == linking_costs.shape[1] else 'N/A'
            }
            self.log_parameters(pre_assignment_analysis, "matrix analysis before assignment")

            # Solve assignment
            assignment_solve_start = time.time()
            with PerformanceTimer(self.logger, "Linear sum assignment"):
                row_ind, col_ind = linear_sum_assignment(linking_costs)
            assignment_solve_time = time.time() - assignment_solve_start

            initial_assignment_info = {
                'initial_assignments': len(row_ind),
                'sample_pairs': list(zip(row_ind[:5], col_ind[:5])),
                'assignment_solve_time': f"{assignment_solve_time:.4f}s"
            }
            self.log_parameters(initial_assignment_info, "initial assignment results")

            # Filter out assignments with infinite or nonlink costs
            self.log_debug("Filtering assignments...")
            valid_assignments = []
            rejected_infinite = 0
            rejected_high_cost = 0

            for r, c in zip(row_ind, col_ind):
                if (r < linking_costs.shape[0] and c < linking_costs.shape[1]):
                    cost = linking_costs[r, c]
                    if np.isfinite(cost):
                        if cost < 1e9:  # Not a prohibited link
                            valid_assignments.append((r, c))
                            self.log_debug(f"Assignment ({r}, {c}): cost {cost:.3f} - ACCEPTED")
                        else:
                            rejected_high_cost += 1
                            self.log_debug(f"Assignment ({r}, {c}): cost {cost} - REJECTED (high cost)")
                    else:
                        rejected_infinite += 1
                        self.log_debug(f"Assignment ({r}, {c}): cost {cost} - REJECTED (infinite)")

            # NEW DEBUG: Assignment filtering statistics
            filtering_stats = {
                'initial_assignments': len(row_ind),
                'valid_assignments': len(valid_assignments),
                'rejected_infinite': rejected_infinite,
                'rejected_high_cost': rejected_high_cost,
                'filtering_efficiency': f"{100 * len(valid_assignments) / max(1, len(row_ind)):.1f}%"
            }
            self.log_parameters(filtering_stats, "assignment filtering statistics")

            self.log_info(f"Final valid assignments: {len(valid_assignments)}")

            if valid_assignments:
                valid_rows, valid_cols = zip(*valid_assignments)
                result = (np.array(valid_rows), np.array(valid_cols))

                # NEW DEBUG: Final assignment quality analysis
                final_costs = [linking_costs[r, c] for r, c in valid_assignments]
                if final_costs:
                    final_quality = {
                        'min_final_cost': np.min(final_costs),
                        'max_final_cost': np.max(final_costs),
                        'mean_final_cost': np.mean(final_costs),
                        'total_assignment_cost': np.sum(final_costs)
                    }
                    self.log_parameters(final_quality, "final assignment quality")

                self.log_info(f"Returning assignments: {len(valid_rows)} pairs")
                return result
            else:
                self.log_warning("No valid assignments after filtering")
                return np.array([]), np.array([])

        except Exception as e:
            self.log_error(f"Error solving assignment: {str(e)}")
            self.log_error(f"Error type: {type(e).__name__}")
            self.log_error(f"Error traceback: {traceback.format_exc()}")
            self.logger.exception("Full traceback:")

            # NEW DEBUG: Enhanced error context for assignment solving
            assignment_error_context = {
                'cost_matrix_shape': cost_matrix.shape,
                'cost_matrix_dtype': cost_matrix.dtype,
                'nonlink_marker': nonlink_marker,
                'finite_elements': np.sum(np.isfinite(cost_matrix))
            }
            self.log_parameters(assignment_error_context, "assignment solving error context")

            return np.array([]), np.array([])

    def _get_frame_coordinates(self, frame_info: Dict, prob_dim: int) -> np.ndarray:
        """Extract coordinates from frame info"""
        log_function_call(self.logger, '_get_frame_coordinates',
                         (frame_info, prob_dim))
        self.log_debug("EXTRACTING FRAME COORDINATES")
        coord_inputs = {
            'prob_dim': prob_dim,
            'num_features': frame_info.get('num', 0),
            'frame_keys': list(frame_info.keys())
        }
        self.log_parameters(coord_inputs, "coordinate extraction inputs")
        num_features = frame_info.get('num', 0)
        if num_features == 0:
            self.log_warning("No features in frame")
            result = np.array([]).reshape(0, prob_dim)
            self.log_debug(f"Returning empty array: {result.shape}")
            return result
        if 'all_coord' in frame_info and len(frame_info['all_coord']) > 0:
            self.log_debug("Using 'all_coord' field")
            coords = frame_info['all_coord'][:, ::2]  # Get x,y,z coordinates
            log_array_info(self.logger, 'all_coord', frame_info['all_coord'], "original all_coord")
            log_array_info(self.logger, 'extracted_coords', coords, "extracted coordinates")
            result = coords[:, :prob_dim]  # Ensure correct dimensionality
            self.log_debug(f"Final coords shape: {result.shape}")
            if len(result) > 0:
                self.log_debug(f"Sample coordinate: {result[0]}")
            return result
        else:
            # Fallback construction
            self.log_debug("Fallback to individual coordinate fields")
            x_coords = frame_info.get('x_coord', np.zeros((num_features, 2)))
            y_coords = frame_info.get('y_coord', np.zeros((num_features, 2)))
            coord_availability = {
                'x_coord_available': 'x_coord' in frame_info,
                'y_coord_available': 'y_coord' in frame_info
            }
            self.log_parameters(coord_availability, "coordinate field availability")
            if isinstance(x_coords, np.ndarray):
                log_array_info(self.logger, 'x_coords', x_coords, "x coordinates")
            if isinstance(y_coords, np.ndarray):
                log_array_info(self.logger, 'y_coords', y_coords, "y coordinates")
            # Handle indexing safely
            if x_coords.ndim > 1:
                x_vals = x_coords[:, 0]
                self.log_debug("Using x_coords[:, 0]")
            else:
                x_vals = x_coords
                self.log_debug("Using x_coords directly")
            if y_coords.ndim > 1:
                y_vals = y_coords[:, 0]
                self.log_debug("Using y_coords[:, 0]")
            else:
                y_vals = y_coords
                self.log_debug("Using y_coords directly")
            if prob_dim == 3:
                self.log_debug("3D coordinates requested")
                z_coords = frame_info.get('z_coord', np.zeros((num_features, 2)))
                if z_coords.ndim > 1:
                    z_vals = z_coords[:, 0]
                else:
                    z_vals = z_coords
                coords = np.column_stack([x_vals, y_vals, z_vals])
                self.log_debug(f"3D coords shape: {coords.shape}")
            else:
                self.log_debug("2D coordinates requested")
                coords = np.column_stack([x_vals, y_vals])
                self.log_debug(f"2D coords shape: {coords.shape}")
            self.log_info("Coordinates extracted using fallback method")
            log_array_info(self.logger, 'final_coords', coords, "fallback extracted coordinates")
            if len(coords) > 0:
                self.log_debug(f"Sample coordinate: {coords[0]}")
            return coords

def convert_tracks_to_array(track_data, data_type, num_frames=None):
    """Convert track data from lists to numpy arrays"""
    logger.info("CONVERTING TRACKS TO ARRAY")
    conversion_inputs = {
        'data_type': data_type,
        'num_frames': num_frames,
        'track_data_type': type(track_data),
        'track_data_length': len(track_data) if hasattr(track_data, 'len') else 'N/A'
    }
    for key, value in conversion_inputs.items():
        logger.debug(f"{key}: {value}")
    if not track_data or not isinstance(track_data, list):
        logger.warning("Invalid or empty track_data")
        if data_type == 'feat_indx':
            result = np.array([]).reshape(0, 0)
        elif data_type == 'coord_amp':
            result = np.array([]).reshape(0, 0)
        else:
            result = np.array([])
        logger.debug(f"Returning empty array: {result.shape if hasattr(result, 'shape') else type(result)}")
        return result
    if data_type == 'feat_indx':
        logger.debug("Converting feature indices...")
        # Convert feature indices to matrix
        if num_frames is None:
            max_length = max(len(track) for track in track_data) if track_data else 0
            logger.debug(f"Auto-detected max_length: {max_length}")
        else:
            max_length = num_frames
            logger.debug(f"Using provided num_frames: {max_length}")
        if max_length == 0:
            logger.warning("Max length is 0")
            return np.array([]).reshape(0, 0)
        feat_matrix = np.zeros((len(track_data), max_length), dtype=int)
        logger.debug(f"Created matrix: {feat_matrix.shape}")
        for i, track in enumerate(track_data):
            if len(track) > 0:
                fill_length = min(len(track), max_length)
                feat_matrix[i, :fill_length] = track[:fill_length]
                if i < 3:  # Debug first few tracks
                    logger.debug(f"Track {i}: {track[:fill_length]}")
        logger.info(f"Feature matrix: {feat_matrix.shape}")
        return feat_matrix
    elif data_type == 'coord_amp':
        logger.debug("Converting coordinates/amplitudes...")
        # Convert coordinates to matrix
        if not track_data:
            logger.warning("Empty track_data")
            return np.array([]).reshape(0, 0)
        with PerformanceTimer(logger, "Coordinate matrix conversion"):
            if num_frames is None:
                max_length = 0
                for track in track_data:
                    if isinstance(track, list):
                        max_length = max(max_length, len(track))
                logger.debug(f"Auto-detected max_length: {max_length}")
            else:
                max_length = num_frames
                logger.debug(f"Using provided num_frames: {max_length}")
            if max_length == 0:
                logger.warning("Max length is 0")
                return np.array([]).reshape(0, 0)
            coord_matrix = np.full((len(track_data), max_length * 8), np.nan)
            logger.debug(f"Created matrix: {coord_matrix.shape}")
            for i, track in enumerate(track_data):
                if isinstance(track, list):
                    for j, frame_data in enumerate(track):
                        if j >= max_length:
                            break
                        if isinstance(frame_data, (list, np.ndarray)) and len(frame_data) >= 4:
                            coord_start = j * 8
                            coord_matrix[i, coord_start:coord_start+4] = frame_data[:4]
                            if i < 3:  # Debug first few tracks
                                logger.debug(f"Track {i}, frame {j}: {frame_data[:4]} at {coord_start}:{coord_start+4}")
        finite_count = np.sum(np.isfinite(coord_matrix))
        logger.info(f"Coordinate matrix: {coord_matrix.shape}, finite values: {finite_count}")
        return coord_matrix
    elif data_type == 'nn_dist':
        logger.debug("Converting nearest neighbor distances...")
        # Convert nearest neighbor distances
        if not track_data:
            logger.warning("Empty track_data")
            return np.array([]).reshape(0, 0)
        if num_frames is None:
            max_length = max(len(track) if isinstance(track, list) else 1 for track in track_data)
            logger.debug(f"Auto-detected max_length: {max_length}")
        else:
            max_length = num_frames
            logger.debug(f"Using provided num_frames: {max_length}")
        nn_matrix = np.zeros((len(track_data), max_length))
        logger.debug(f"Created matrix: {nn_matrix.shape}")
        for i, track in enumerate(track_data):
            if isinstance(track, (list, np.ndarray)):
                track_array = np.array(track).flatten()
                fill_length = min(len(track_array), max_length)
                nn_matrix[i, :fill_length] = track_array[:fill_length]
                if i < 3:  # Debug first few tracks
                    logger.debug(f"Track {i}: {track_array[:fill_length]}")
            else:
                nn_matrix[i, 0] = float(track) if track is not None else 0.0
                if i < 3:  # Debug first few tracks
                    logger.debug(f"Track {i}: {float(track) if track is not None else 0.0}")
        logger.info(f"NN distance matrix: {nn_matrix.shape}")
        return nn_matrix
    logger.debug("Converting generic track data...")
    result = np.array(track_data)
    logger.debug(f"Generic conversion result: {result.shape if hasattr(result, 'shape') else type(result)}")
    return result

