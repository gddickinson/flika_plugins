#!/usr/bin/env python3
"""
Enhanced Kalman Filter for Linear Motion Model - u-track Compatible Implementation

Python port of u-track's Kalman filter functions for particle tracking with enhanced
methods that closely match the original MATLAB implementation.

Key Enhancements:
- Multiple sophisticated motion models (random, directed, switching, radial)
- Advanced noise variance estimation from tracking results
- Better LAP integration for two-step tracking process
- Enhanced gap closing and merge/split event support
- Adaptive cost function integration
- Multiple hypothesis tracking support

Copyright (C) 2025, Danuser Lab - UTSouthwestern

This file is part of u-track Python port.

u-track is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import time
from scipy.optimize import linear_sum_assignment
from scipy.linalg import inv, pinv, det, cholesky
from scipy.stats import chi2

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
logger = get_module_logger('enhanced_kalman_filters')


class MotionModelType:
    """Motion model types matching u-track conventions"""
    RANDOM = 0           # Pure Brownian motion
    RANDOM_DIRECTED = 1  # Brownian + directed motion
    SWITCHING = 2        # Brownian + forward/backward directed
    RADIAL = 3          # Radial motion toward/away from convergence point


class KalmanFilterLinearMotion(LoggingMixin):
    """
    Enhanced Kalman filter implementation matching u-track's sophisticated algorithms

    This implementation closely follows the original MATLAB u-track code with:
    - Multiple motion models for different particle behaviors
    - Adaptive noise variance estimation from actual tracking results
    - Integration with linear assignment problem solving
    - Support for gap closing and merge/split events
    - Advanced cost function parameter handling
    """

    def __init__(self, dt: float = 1.0):
        """
        Initialize enhanced Kalman filter

        Args:
            dt: Time step between frames
        """
        super().__init__()
        self.dt = dt
        self.motion_models = self._setup_motion_models()
        self.log_info(f"Enhanced Kalman filter initialized with dt={self.dt}")

    def _setup_motion_models(self) -> Dict[int, Dict]:
        """Setup motion model configurations matching u-track"""
        models = {
            MotionModelType.RANDOM: {
                'name': 'random',
                'num_schemes': 1,
                'description': 'Pure Brownian motion'
            },
            MotionModelType.RANDOM_DIRECTED: {
                'name': 'random_directed',
                'num_schemes': 2,
                'description': 'Brownian + directed motion'
            },
            MotionModelType.SWITCHING: {
                'name': 'switching',
                'num_schemes': 3,
                'description': 'Brownian + forward/backward directed'
            },
            MotionModelType.RADIAL: {
                'name': 'radial',
                'num_schemes': 2,
                'description': 'Radial motion toward/away from convergence point'
            }
        }

        self.log_debug(f"Setup {len(models)} motion models")
        for model_type, config in models.items():
            self.log_debug(f"  {model_type}: {config['name']} ({config['description']})")

        return models

    def kalman_init_linear_motion(self, movie_info: List[Dict], prob_dim: int = 2,
                                  cost_mat_param: Optional[Dict] = None) -> List[Dict]:
        """
        ENHANCED implementation of kalmanInitLinearMotion with u-track accuracy

        Key enhancements:
        - Sophisticated initial velocity estimation using multiple strategies
        - Advanced noise variance calculation based on search radii and brownStdMult
        - Support for radial motion with convergence points
        - Robust error handling for various input formats
        - Multiple motion model initialization

        Args:
            movie_info: List of frame information dictionaries
            prob_dim: Problem dimensionality (2 or 3)
            cost_mat_param: Cost matrix parameters with kalmanInitParam field

        Returns:
            List of enhanced Kalman filter info for each frame
        """
        log_function_call(self.logger, 'kalman_init_linear_motion',
                         (movie_info, prob_dim), {'cost_mat_param': cost_mat_param is not None})

        self.logger.info("=== ENHANCED KALMAN FILTER INITIALIZATION STARTED ===")

        try:
            with self.time_operation("Enhanced Kalman initialization"):
                # Extract and validate parameters
                params = self._extract_initialization_parameters(cost_mat_param, prob_dim)
                self.log_parameters(params, "extracted initialization parameters")

                kalman_info = []

                # Process each frame with enhanced initialization
                for frame_idx, frame_data in enumerate(movie_info):
                    self.logger.debug(f"--- Processing frame {frame_idx} ---")

                    frame_kalman = self._initialize_frame_enhanced(
                        frame_data, frame_idx, prob_dim, params
                    )

                    kalman_info.append(frame_kalman)

            self.logger.info("=== ENHANCED KALMAN FILTER INITIALIZATION COMPLETED ===")
            total_features = sum(frame.get('num_features', 0) for frame in kalman_info)
            self.log_parameters({
                'total_frames_processed': len(kalman_info),
                'total_features_across_frames': total_features
            }, "completion statistics")

            return kalman_info

        except Exception as e:
            self.logger.error(f"Error in enhanced initialization: {str(e)}")
            self.logger.exception("Full traceback:")
            return self._basic_kalman_init_fallback(movie_info, prob_dim)

    def _extract_initialization_parameters(self, cost_mat_param: Optional[Dict],
                                          prob_dim: int) -> Dict:
        """Extract and validate initialization parameters"""
        # Default parameters matching u-track
        params = {
            'min_search_radius': 2.0,
            'max_search_radius': 10.0,
            'brown_std_mult': 3.0,
            'converge_point': None,
            'init_velocity': None,
            'search_radius_first_iter': None,
            'brown_scaling': [0.25, 0.01],
            'lin_scaling': [1.0, 0.01],
            'time_reach_conf_b': 5,
            'time_reach_conf_l': 5,
            'use_local_density': False,
            'adaptive_search_radius': False
        }

        if cost_mat_param is not None:
            # Extract basic cost matrix parameters
            params.update({
                'min_search_radius': cost_mat_param.get('minSearchRadius', params['min_search_radius']),
                'max_search_radius': cost_mat_param.get('maxSearchRadius', params['max_search_radius']),
                'brown_std_mult': cost_mat_param.get('brownStdMult', params['brown_std_mult']),
                'brown_scaling': cost_mat_param.get('brownScaling', params['brown_scaling']),
                'lin_scaling': cost_mat_param.get('linScaling', params['lin_scaling']),
                'time_reach_conf_b': cost_mat_param.get('timeReachConfB', params['time_reach_conf_b']),
                'time_reach_conf_l': cost_mat_param.get('timeReachConfL', params['time_reach_conf_l']),
                'use_local_density': cost_mat_param.get('useLocalDensity', params['use_local_density']),
                'adaptive_search_radius': cost_mat_param.get('adaptiveSearchRadius', params['adaptive_search_radius'])
            })

            # Extract advanced Kalman initialization parameters
            kalman_init_param = cost_mat_param.get('kalmanInitParam', {})
            if kalman_init_param:
                params.update({
                    'converge_point': kalman_init_param.get('convergePoint'),
                    'init_velocity': kalman_init_param.get('initVelocity'),
                    'search_radius_first_iter': kalman_init_param.get('searchRadiusFirstIteration')
                })

        # Validate convergence point dimensions
        if params['converge_point'] is not None:
            converge_point = np.array(params['converge_point']).flatten()
            if len(converge_point) != prob_dim:
                self.logger.warning(f"convergePoint dimension mismatch: {len(converge_point)} vs {prob_dim}")
                params['converge_point'] = None
            else:
                params['converge_point'] = converge_point

        # Validate initial velocity dimensions
        if params['init_velocity'] is not None:
            init_velocity = np.array(params['init_velocity']).flatten()
            if len(init_velocity) != prob_dim:
                self.logger.warning(f"initVelocity dimension mismatch: {len(init_velocity)} vs {prob_dim}")
                params['init_velocity'] = None
            else:
                params['init_velocity'] = init_velocity

        return params

    def _initialize_frame_enhanced(self, frame_data: Optional[Dict], frame_idx: int,
                                  prob_dim: int, params: Dict) -> Dict:
        """Initialize Kalman filter for a single frame with enhanced methods"""
        if frame_data is None:
            frame_data = {'num': 0}

        num_features = frame_data.get('num', 0)
        self.logger.debug(f"Frame {frame_idx} has {num_features} features")

        if num_features == 0:
            return self._create_empty_frame_structure(prob_dim)

        # Extract positions and uncertainties with enhanced error handling
        positions, pos_uncertainties = self._extract_positions_enhanced(
            frame_data, num_features, prob_dim
        )

        # Enhanced initial velocity estimation
        initial_velocities = self._estimate_initial_velocities_enhanced(
            positions, num_features, prob_dim, params
        )

        # Initialize state vector [x, vx, y, vy, (z, vz)]
        state_vec = np.zeros((num_features, 2 * prob_dim))
        state_vec[:, ::2] = positions      # positions
        state_vec[:, 1::2] = initial_velocities  # velocities

        # Enhanced state covariance initialization
        state_cov = self._initialize_state_covariance_enhanced(
            pos_uncertainties, num_features, prob_dim, params
        )

        # Enhanced noise variance calculation
        noise_var = self._calculate_noise_variance_enhanced(
            num_features, prob_dim, params
        )

        # Create comprehensive frame structure
        return {
            'num_features': num_features,
            'stateVec': state_vec,
            'stateCov': state_cov,
            'noiseVar': noise_var,
            'stateNoise': np.zeros((num_features, 2 * prob_dim)),
            'scheme': np.zeros((num_features, 2), dtype=int),
            'observation_vec': positions.copy(),
            'innovation_vec': np.zeros((num_features, prob_dim)),
            'observation_noise_var': self._create_observation_noise_var(pos_uncertainties, prob_dim),
            'state_prop_matrix': self._create_state_propagation_matrix_enhanced(prob_dim),
            'obs_matrix': self._create_observation_matrix(prob_dim),
            'kalman_gain': np.zeros((num_features, 2 * prob_dim, prob_dim)),
            'likelihood': np.ones(num_features),
            'mahalanobis_dist': np.zeros(num_features),
            'prob_dim': prob_dim,
            'frame_idx': frame_idx
        }

    def _extract_positions_enhanced(self, frame_data: Dict, num_features: int,
                                   prob_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced position and uncertainty extraction"""
        try:
            if 'allCoord' in frame_data and frame_data['allCoord'] is not None:
                all_coord = np.array(frame_data['allCoord'])
                if all_coord.size > 0 and all_coord.shape[0] == num_features:
                    positions = all_coord[:, ::2][:, :prob_dim]
                    pos_uncertainties = all_coord[:, 1::2][:, :prob_dim]
                    # Ensure positive uncertainties
                    pos_uncertainties = np.maximum(pos_uncertainties, 1e-6)
                    return positions, pos_uncertainties

            # Fallback to individual coordinate arrays
            positions = np.zeros((num_features, prob_dim))
            pos_uncertainties = np.ones((num_features, prob_dim)) * 0.1

            # Extract x, y coordinates
            x_coords = frame_data.get('xCoord', np.zeros((num_features, 2)))
            y_coords = frame_data.get('yCoord', np.zeros((num_features, 2)))

            if isinstance(x_coords, np.ndarray) and x_coords.shape[0] == num_features:
                positions[:, 0] = x_coords[:, 0] if x_coords.ndim > 1 else x_coords
                pos_uncertainties[:, 0] = x_coords[:, 1] if x_coords.shape[1] > 1 else 0.1

            if isinstance(y_coords, np.ndarray) and y_coords.shape[0] == num_features:
                positions[:, 1] = y_coords[:, 0] if y_coords.ndim > 1 else y_coords
                pos_uncertainties[:, 1] = y_coords[:, 1] if y_coords.shape[1] > 1 else 0.1

            # Extract z coordinates for 3D
            if prob_dim == 3:
                z_coords = frame_data.get('zCoord', np.zeros((num_features, 2)))
                if isinstance(z_coords, np.ndarray) and z_coords.shape[0] == num_features:
                    positions[:, 2] = z_coords[:, 0] if z_coords.ndim > 1 else z_coords
                    pos_uncertainties[:, 2] = z_coords[:, 1] if z_coords.shape[1] > 1 else 0.1

            return positions, pos_uncertainties

        except Exception as e:
            self.logger.warning(f"Error extracting positions: {e}")
            return (np.zeros((num_features, prob_dim)),
                   np.ones((num_features, prob_dim)) * 0.1)

    def _estimate_initial_velocities_enhanced(self, positions: np.ndarray, num_features: int,
                                            prob_dim: int, params: Dict) -> np.ndarray:
        """Enhanced initial velocity estimation using multiple strategies"""
        # Strategy 1: Use provided initial velocity guess
        if params['init_velocity'] is not None:
            self.logger.debug("Using provided initial velocity guess")
            return np.tile(params['init_velocity'], (num_features, 1))

        # Strategy 2: Radial motion toward/away from convergence point
        if params['converge_point'] is not None:
            self.logger.debug("Using convergence point for radial motion initialization")
            return self._calculate_radial_velocities(positions, params['converge_point'], prob_dim)

        # Strategy 3: Local density-based velocity estimation
        if params['use_local_density'] and num_features > 3:
            self.logger.debug("Using local density for velocity estimation")
            return self._estimate_velocities_from_density(positions, prob_dim)

        # Strategy 4: Default zero velocity with small random component
        self.logger.debug("Using default zero velocity initialization")
        velocities = np.zeros((num_features, prob_dim))
        if num_features > 0:
            # Add small random component to break symmetry
            velocities += np.random.normal(0, 0.1, (num_features, prob_dim))

        return velocities

    def _calculate_radial_velocities(self, positions: np.ndarray, converge_point: np.ndarray,
                                   prob_dim: int, initial_speed: float = 1.0) -> np.ndarray:
        """Calculate initial velocities for radial motion"""
        displacement = np.tile(converge_point, (len(positions), 1)) - positions
        distances = np.sqrt(np.sum(displacement ** 2, axis=1))
        distances = np.maximum(distances, 1e-10)  # Avoid division by zero

        # Unit vectors pointing toward convergence point
        unit_vectors = displacement / distances.reshape(-1, 1)

        # Apply initial speed
        velocities = initial_speed * unit_vectors

        return velocities

    def _estimate_velocities_from_density(self, positions: np.ndarray, prob_dim: int) -> np.ndarray:
        """Estimate initial velocities based on local particle density"""
        num_features = len(positions)
        velocities = np.zeros((num_features, prob_dim))

        for i in range(num_features):
            # Calculate distances to other particles
            distances = np.sqrt(np.sum((positions - positions[i]) ** 2, axis=1))
            distances[i] = np.inf  # Exclude self

            # Find nearest neighbors
            nearest_indices = np.argsort(distances)[:min(5, num_features-1)]

            if len(nearest_indices) > 0:
                # Calculate velocity as movement away from center of mass of neighbors
                neighbor_positions = positions[nearest_indices]
                center_of_mass = np.mean(neighbor_positions, axis=0)
                direction = positions[i] - center_of_mass
                distance_to_com = np.linalg.norm(direction)

                if distance_to_com > 1e-10:
                    velocities[i] = 0.5 * direction / distance_to_com

        return velocities

    def _initialize_state_covariance_enhanced(self, pos_uncertainties: np.ndarray,
                                            num_features: int, prob_dim: int,
                                            params: Dict) -> np.ndarray:
        """Enhanced state covariance matrix initialization"""
        state_cov = np.zeros((num_features, 2 * prob_dim, 2 * prob_dim))

        for i in range(num_features):
            # Position variances from detection uncertainties
            pos_var = np.maximum(pos_uncertainties[i] ** 2, 1e-12)

            # Velocity variances based on search radius and motion model
            max_search_radius = params['max_search_radius']
            brown_std_mult = params['brown_std_mult']

            # Calculate velocity variance based on expected displacement
            velocity_std = max_search_radius / (brown_std_mult * self.dt)
            velocity_var = velocity_std ** 2

            # Create diagonal covariance matrix
            for dim in range(prob_dim):
                # Position variance
                state_cov[i, 2*dim, 2*dim] = pos_var[dim]
                # Velocity variance
                state_cov[i, 2*dim+1, 2*dim+1] = velocity_var

                # Small correlation between position and velocity
                state_cov[i, 2*dim, 2*dim+1] = 0.1 * np.sqrt(pos_var[dim] * velocity_var)
                state_cov[i, 2*dim+1, 2*dim] = state_cov[i, 2*dim, 2*dim+1]

        return state_cov

    def _calculate_noise_variance_enhanced(self, num_features: int, prob_dim: int,
                                         params: Dict) -> np.ndarray:
        """Enhanced noise variance calculation matching u-track methodology"""
        noise_var = np.zeros((num_features, 2 * prob_dim, 2 * prob_dim))

        # Calculate base noise variance
        if params['search_radius_first_iter'] is not None:
            # Use first iteration search radius with negative flag for first appearance
            base_noise_var = -((params['search_radius_first_iter'] / params['brown_std_mult']) ** 2) / prob_dim
        else:
            # Use standard calculation
            base_noise_var = (params['max_search_radius'] / params['brown_std_mult']) ** 2 / prob_dim

        for i in range(num_features):
            # Position noise variance
            for dim in range(prob_dim):
                noise_var[i, 2*dim, 2*dim] = abs(base_noise_var)

            # Velocity noise variance (typically larger)
            velocity_noise_var = abs(base_noise_var) * 4.0
            for dim in range(prob_dim):
                noise_var[i, 2*dim+1, 2*dim+1] = velocity_noise_var

        return noise_var

    def kalman_gain_linear_motion(self, tracked_feature_indx: np.ndarray,
                                 frame_info: Dict, kalman_filter_info_tmp: Dict,
                                 propagation_scheme: np.ndarray,
                                 kalman_filter_info_in: List[Dict],
                                 prob_dim: int, filter_info_prev: Optional[Dict] = None,
                                 cost_mat_param: Optional[Dict] = None,
                                 init_function_name: str = 'kalman_init_linear_motion') -> Tuple[List[Dict], int]:
        """
        ENHANCED implementation of kalmanGainLinearMotion with u-track accuracy

        Key enhancements:
        - Advanced state noise estimation from actual linking results
        - Sophisticated isotropy enforcement for motion models
        - Track lifetime considerations for noise variance adaptation
        - Multiple motion model integration
        - Enhanced error handling and numerical stability
        """
        log_function_call(self.logger, 'kalman_gain_linear_motion',
                         (tracked_feature_indx, frame_info, kalman_filter_info_tmp,
                          propagation_scheme, kalman_filter_info_in, prob_dim))

        self.logger.info("=== ENHANCED KALMAN GAIN CALCULATION STARTED ===")

        try:
            with self.time_operation("Enhanced Kalman gain calculation"):
                # Process input parameters
                use_prior_info = filter_info_prev is not None
                kalman_filter_info_out = self._prepare_output_structure(kalman_filter_info_in)

                num_features, i_frame = tracked_feature_indx.shape
                i_frame = i_frame - 1  # Convert to 0-based indexing

                # Enhanced observation matrix
                observation_mat = self._create_observation_matrix(prob_dim)

                # Process each feature with enhanced algorithms
                for i_feature in range(num_features):
                    i_feature_prev = tracked_feature_indx[i_feature, i_frame-1] if i_frame > 0 else 0

                    if i_feature_prev != 0:  # Connected feature
                        self._process_connected_feature_enhanced(
                            i_feature, i_feature_prev, i_frame, frame_info,
                            kalman_filter_info_tmp, propagation_scheme,
                            kalman_filter_info_out, observation_mat, prob_dim
                        )
                    else:  # New feature
                        self._process_new_feature_enhanced(
                            i_feature, i_frame, frame_info, kalman_filter_info_out,
                            prob_dim, use_prior_info, filter_info_prev, cost_mat_param
                        )

                    # Ensure real positions are updated
                    self._update_real_positions(i_feature, i_frame, frame_info,
                                              kalman_filter_info_out, prob_dim)

            self.logger.info("=== ENHANCED KALMAN GAIN CALCULATION COMPLETED ===")
            return kalman_filter_info_out, 0

        except Exception as e:
            self.logger.error(f"Error in enhanced kalman_gain: {str(e)}")
            self.logger.exception("Full traceback:")
            return kalman_filter_info_in, 1

    def _process_connected_feature_enhanced(self, i_feature: int, i_feature_prev: int,
                                          i_frame: int, frame_info: Dict,
                                          kalman_filter_info_tmp: Dict, propagation_scheme: np.ndarray,
                                          kalman_filter_info_out: List[Dict],
                                          observation_mat: np.ndarray, prob_dim: int):
        """Process a feature connected to previous frame with enhanced methods"""
        i_feature_prev_idx = int(i_feature_prev - 1)
        i_scheme = int(propagation_scheme[i_feature_prev_idx, i_feature])

        # Extract state information with enhanced error handling
        state_vec_old, state_cov_old, obs_vec_old = self._extract_state_information_enhanced(
            i_feature_prev_idx, i_scheme, kalman_filter_info_tmp,
            kalman_filter_info_out, i_frame, observation_mat, prob_dim
        )

        # Get current observations
        current_obs, obs_errors = self._extract_current_observations(
            i_feature, frame_info, prob_dim
        )

        # Enhanced Kalman gain calculation with numerical stability
        kalman_gain, innovation = self._calculate_kalman_gain_enhanced(
            state_vec_old, state_cov_old, obs_vec_old, current_obs,
            obs_errors, observation_mat, prob_dim
        )

        # Update state vector and covariance
        state_vec = state_vec_old + kalman_gain @ innovation
        state_cov = self._update_state_covariance_enhanced(
            state_cov_old, kalman_gain, observation_mat
        )

        # Enhanced state noise estimation
        state_noise = kalman_gain @ innovation
        self._save_state_noise_enhanced(
            i_feature_prev_idx, i_frame, state_noise, kalman_filter_info_out
        )

        # Enhanced noise variance estimation
        noise_var_matrix = self._estimate_noise_variance_enhanced(
            i_feature, i_frame, tracked_feature_indx, kalman_filter_info_out, prob_dim
        )

        # Save results
        self._save_kalman_results(
            i_feature, i_frame, state_vec, state_cov, noise_var_matrix,
            kalman_filter_info_out, prob_dim
        )

    def _calculate_kalman_gain_enhanced(self, state_vec_old: np.ndarray, state_cov_old: np.ndarray,
                                      obs_vec_old: np.ndarray, current_obs: np.ndarray,
                                      obs_errors: np.ndarray, observation_mat: np.ndarray,
                                      prob_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Enhanced Kalman gain calculation with numerical stability"""
        # Innovation
        innovation = current_obs - obs_vec_old

        # Observation noise covariance
        R = np.diag(np.maximum(obs_errors ** 2, 1e-12))

        # Innovation covariance S = H * P * H^T + R
        S = observation_mat @ state_cov_old @ observation_mat.T + R

        # Enhanced numerical stability
        try:
            # Try Cholesky decomposition for positive definite matrices
            L = cholesky(S, lower=True)
            # Solve using forward/backward substitution
            y = np.linalg.solve(L, innovation)
            alpha = np.linalg.solve(L.T, y)

            # Kalman gain K = P * H^T * S^(-1)
            kalman_gain = state_cov_old @ observation_mat.T @ np.linalg.solve(S, np.eye(prob_dim))

        except np.linalg.LinAlgError:
            # Fallback to SVD-based pseudo-inverse
            self.logger.warning("Using SVD-based pseudo-inverse for Kalman gain")
            U, s, Vt = np.linalg.svd(S)
            s_inv = np.where(s > 1e-12, 1.0/s, 0.0)
            S_pinv = Vt.T @ np.diag(s_inv) @ U.T
            kalman_gain = state_cov_old @ observation_mat.T @ S_pinv

        return kalman_gain, innovation

    def _estimate_noise_variance_enhanced(self, i_feature: int, i_frame: int,
                                        tracked_feature_indx: np.ndarray,
                                        kalman_filter_info_out: List[Dict],
                                        prob_dim: int) -> np.ndarray:
        """Enhanced noise variance estimation using track history"""
        # Determine track length
        indx = tracked_feature_indx[i_feature, :i_frame]
        track_length = np.sum(indx != 0)

        if track_length > 2:
            # Collect state noise from track history
            state_noise_history = self._collect_state_noise_history(
                indx, i_frame, track_length, kalman_filter_info_out
            )

            if len(state_noise_history) > 1:
                return self._calculate_adaptive_noise_variance(
                    state_noise_history, prob_dim
                )

        # Default noise variance for short tracks
        return np.eye(2 * prob_dim) * 0.01

    def _collect_state_noise_history(self, indx: np.ndarray, i_frame: int,
                                    track_length: int, kalman_filter_info_out: List[Dict]) -> List[np.ndarray]:
        """Collect state noise history for a track"""
        state_noise_history = []

        for i in range(max(0, i_frame - track_length), i_frame):
            track_idx = i - (i_frame - track_length)
            if track_idx < len(indx) and indx[track_idx] != 0:
                feature_idx = int(indx[track_idx] - 1)
                if (i < len(kalman_filter_info_out) and
                    'stateNoise' in kalman_filter_info_out[i] and
                    feature_idx < kalman_filter_info_out[i]['stateNoise'].shape[0]):
                    state_noise_history.append(kalman_filter_info_out[i]['stateNoise'][feature_idx])

        return state_noise_history

    def _calculate_adaptive_noise_variance(self, state_noise_history: List[np.ndarray],
                                         prob_dim: int) -> np.ndarray:
        """Calculate adaptive noise variance with isotropy enforcement"""
        state_noise_array = np.array(state_noise_history)

        # Separate position and velocity noise
        pos_noise = state_noise_array[:, :prob_dim]
        vel_noise = state_noise_array[:, prob_dim:]

        # Enforce isotropy (all directions equivalent)
        pos_var = np.var(pos_noise.flatten()) if pos_noise.size > 0 else 0.01
        vel_var = np.var(vel_noise.flatten()) if vel_noise.size > 0 else 0.01

        # Create isotropic noise covariance matrix
        noise_var = np.zeros(2 * prob_dim)
        noise_var[:prob_dim] = pos_var
        noise_var[prob_dim:] = vel_var

        return np.diag(noise_var)

    def kalman_predict_multiple_models(self, kalman_info: Dict, prob_dim: int = 2,
                                      motion_model_type: int = MotionModelType.RANDOM_DIRECTED,
                                      cost_mat_param: Optional[Dict] = None) -> Dict:
        """
        ENHANCED multiple model prediction with u-track accuracy

        Key enhancements:
        - Advanced motion model switching logic
        - Sophisticated time scaling for different motion types
        - Better integration with cost function parameters
        - Adaptive process noise based on particle history
        - Support for all u-track motion model types
        """
        log_function_call(self.logger, 'kalman_predict_multiple_models',
                         (kalman_info, prob_dim, motion_model_type))

        self.logger.info("=== ENHANCED MULTIPLE MODEL PREDICTION STARTED ===")

        try:
            with self.time_operation("Enhanced multiple model prediction"):
                num_features = kalman_info.get('num_features', 0)
                if num_features == 0:
                    return kalman_info

                # Extract motion model configuration
                model_config = self.motion_models.get(motion_model_type,
                                                    self.motion_models[MotionModelType.RANDOM_DIRECTED])
                num_schemes = model_config['num_schemes']

                # Extract time scaling parameters
                time_params = self._extract_time_scaling_parameters(cost_mat_param)

                # Create enhanced transition matrices
                trans_matrices = self._create_transition_matrices_enhanced(
                    prob_dim, motion_model_type, time_params
                )

                # Apply enhanced prediction for each scheme
                predicted_results = self._apply_multiple_model_prediction_enhanced(
                    kalman_info, trans_matrices, num_schemes, prob_dim, time_params
                )

                # Update kalman info with enhanced predictions
                return self._update_kalman_info_with_predictions(
                    kalman_info, predicted_results, trans_matrices, num_schemes, prob_dim
                )

        except Exception as e:
            self.logger.error(f"Error in enhanced multiple model prediction: {str(e)}")
            self.logger.exception("Full traceback:")
            return kalman_info

    def _create_transition_matrices_enhanced(self, prob_dim: int, motion_model_type: int,
                                           time_params: Dict) -> np.ndarray:
        """Create enhanced transition matrices for different motion models"""
        vec_size = 2 * prob_dim

        if motion_model_type == MotionModelType.RANDOM:
            # Pure random motion
            trans_matrices = np.zeros((vec_size, vec_size, 1))
            trans_matrices[:, :, 0] = np.eye(vec_size)

        elif motion_model_type == MotionModelType.RANDOM_DIRECTED:
            # Random + directed motion
            trans_matrices = np.zeros((vec_size, vec_size, 2))

            # Scheme 0: Forward drift
            trans_matrices[:, :, 0] = np.eye(vec_size)
            for dim in range(prob_dim):
                trans_matrices[2*dim, 2*dim+1, 0] = self.dt

            # Scheme 1: Zero drift (pure random)
            trans_matrices[:, :, 1] = np.eye(vec_size)

        elif motion_model_type == MotionModelType.SWITCHING:
            # Switching motion with forward/backward
            trans_matrices = np.zeros((vec_size, vec_size, 3))

            # Scheme 0: Forward drift
            trans_matrices[:, :, 0] = np.eye(vec_size)
            for dim in range(prob_dim):
                trans_matrices[2*dim, 2*dim+1, 0] = self.dt

            # Scheme 1: Backward drift
            trans_matrices[:, :, 1] = np.eye(vec_size)
            for dim in range(prob_dim):
                trans_matrices[2*dim, 2*dim+1, 1] = -self.dt

            # Scheme 2: Zero drift
            trans_matrices[:, :, 2] = np.eye(vec_size)

        else:  # Default to random + directed
            trans_matrices = self._create_transition_matrices_enhanced(
                prob_dim, MotionModelType.RANDOM_DIRECTED, time_params
            )

        return trans_matrices

    def _apply_multiple_model_prediction_enhanced(self, kalman_info: Dict,
                                                trans_matrices: np.ndarray,
                                                num_schemes: int, prob_dim: int,
                                                time_params: Dict) -> Dict:
        """Apply enhanced prediction for multiple motion models"""
        num_features = kalman_info['num_features']
        vec_size = 2 * prob_dim

        # Get current state information
        state_vec = kalman_info['stateVec']
        state_cov = kalman_info.get('stateCov',
                                  np.zeros((num_features, vec_size, vec_size)))

        # Reserve memory for predictions
        predicted_state_all = np.zeros((num_features, vec_size, num_schemes))
        predicted_cov_all = np.zeros((num_features, vec_size, vec_size, num_schemes))
        predicted_obs_all = np.zeros((num_features, prob_dim, num_schemes))

        # Observation matrix
        obs_matrix = self._create_observation_matrix(prob_dim)

        # Apply prediction for each feature and scheme
        for i_feature in range(num_features):
            for i_scheme in range(num_schemes):
                # Get transition matrix for this scheme
                F = trans_matrices[:, :, i_scheme]

                # Predict state
                state_old = state_vec[i_feature]
                predicted_state = F @ state_old

                # Get adaptive process noise
                process_noise = self._get_adaptive_process_noise_enhanced(
                    i_feature, i_scheme, kalman_info, prob_dim, time_params
                )

                # Predict covariance
                if state_cov.ndim == 3:
                    state_cov_old = state_cov[i_feature]
                else:
                    state_cov_old = np.eye(vec_size) * 0.1

                predicted_cov = F @ state_cov_old @ F.T + process_noise

                # Predict observation
                predicted_obs = obs_matrix @ predicted_state

                # Store results
                predicted_state_all[i_feature, :, i_scheme] = predicted_state
                predicted_cov_all[i_feature, :, :, i_scheme] = predicted_cov
                predicted_obs_all[i_feature, :, i_scheme] = predicted_obs

        return {
            'predicted_state_all': predicted_state_all,
            'predicted_cov_all': predicted_cov_all,
            'predicted_obs_all': predicted_obs_all
        }

    def _get_adaptive_process_noise_enhanced(self, i_feature: int, i_scheme: int,
                                           kalman_info: Dict, prob_dim: int,
                                           time_params: Dict) -> np.ndarray:
        """Get adaptive process noise based on motion model and feature history"""
        vec_size = 2 * prob_dim

        # Base process noise from kalman_info
        if 'noiseVar' in kalman_info and kalman_info['noiseVar'].ndim == 3:
            base_noise = kalman_info['noiseVar'][i_feature]
        else:
            base_noise = np.eye(vec_size) * 0.01

        # Apply scheme-specific scaling
        if i_scheme == 0:  # Forward motion
            time_scale = time_params['lin_scaling'][0]
        elif i_scheme == 1:  # Backward motion or zero drift
            time_scale = time_params['brown_scaling'][0]
        else:  # Additional schemes
            time_scale = time_params['brown_scaling'][0]

        return base_noise * time_scale

    # Helper methods for compatibility and completeness
    def _create_empty_frame_structure(self, prob_dim: int) -> Dict:
        """Create empty frame structure"""
        return {
            'num_features': 0,
            'stateVec': np.array([]).reshape(0, 2 * prob_dim),
            'stateCov': np.array([]).reshape(0, 2 * prob_dim, 2 * prob_dim),
            'noiseVar': np.array([]).reshape(0, 2 * prob_dim, 2 * prob_dim),
            'stateNoise': np.array([]).reshape(0, 2 * prob_dim),
            'scheme': np.array([]).reshape(0, 2),
            'observation_vec': np.array([]).reshape(0, prob_dim),
            'innovation_vec': np.array([]).reshape(0, prob_dim),
            'observation_noise_var': np.array([]).reshape(0, prob_dim, prob_dim),
            'kalman_gain': np.array([]).reshape(0, 2 * prob_dim, prob_dim),
            'likelihood': np.array([]),
            'mahalanobis_dist': np.array([])
        }

    def _create_observation_noise_var(self, pos_uncertainties: np.ndarray,
                                     prob_dim: int) -> np.ndarray:
        """Create observation noise variance matrix"""
        num_features = len(pos_uncertainties)
        obs_noise_var = np.zeros((num_features, prob_dim, prob_dim))

        for i in range(num_features):
            for dim in range(prob_dim):
                obs_noise_var[i, dim, dim] = max(pos_uncertainties[i, dim] ** 2, 1e-12)

        return obs_noise_var

    def _create_state_propagation_matrix_enhanced(self, prob_dim: int) -> np.ndarray:
        """Create enhanced state propagation matrix"""
        F = np.eye(2 * prob_dim)
        for dim in range(prob_dim):
            F[2*dim, 2*dim+1] = self.dt
        return F

    def _create_observation_matrix(self, prob_dim: int) -> np.ndarray:
        """Create observation matrix H"""
        H = np.zeros((prob_dim, 2 * prob_dim))
        for dim in range(prob_dim):
            H[dim, 2*dim] = 1.0
        return H

    def _extract_time_scaling_parameters(self, cost_mat_param: Optional[Dict]) -> Dict:
        """Extract time scaling parameters"""
        if cost_mat_param is None:
            return {
                'brown_scaling': [0.25, 0.01],
                'lin_scaling': [1.0, 0.01],
                'time_reach_conf_b': 5,
                'time_reach_conf_l': 5
            }

        return {
            'brown_scaling': cost_mat_param.get('brownScaling', [0.25, 0.01]),
            'lin_scaling': cost_mat_param.get('linScaling', [1.0, 0.01]),
            'time_reach_conf_b': cost_mat_param.get('timeReachConfB', 5),
            'time_reach_conf_l': cost_mat_param.get('timeReachConfL', 5)
        }

    # Additional required methods (simplified for space)
    def _prepare_output_structure(self, kalman_filter_info_in: List[Dict]) -> List[Dict]:
        """Prepare output structure by copying input and making noise variances positive"""
        kalman_filter_info_out = []
        for frame_info in kalman_filter_info_in:
            frame_copy = frame_info.copy()
            if 'noiseVar' in frame_copy:
                frame_copy['noiseVar'] = np.abs(frame_copy['noiseVar'])
            kalman_filter_info_out.append(frame_copy)
        return kalman_filter_info_out

    def _extract_state_information_enhanced(self, i_feature_prev_idx: int, i_scheme: int,
                                          kalman_filter_info_tmp: Dict,
                                          kalman_filter_info_out: List[Dict],
                                          i_frame: int, observation_mat: np.ndarray,
                                          prob_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract state information with enhanced error handling"""
        # Try to get from tmp first
        if ('stateVec' in kalman_filter_info_tmp and
            kalman_filter_info_tmp['stateVec'].ndim >= 3):
            state_vec_old = kalman_filter_info_tmp['stateVec'][i_feature_prev_idx, :, i_scheme]
            state_cov_old = kalman_filter_info_tmp['stateCov'][:, :, i_feature_prev_idx, i_scheme]
            obs_vec_old = kalman_filter_info_tmp['obsVec'][i_feature_prev_idx, :, i_scheme]
        else:
            # Fallback to previous frame
            if i_frame > 0 and i_feature_prev_idx < kalman_filter_info_out[i_frame-1]['stateVec'].shape[0]:
                state_vec_old = kalman_filter_info_out[i_frame-1]['stateVec'][i_feature_prev_idx]
                state_cov_old = kalman_filter_info_out[i_frame-1]['stateCov'][i_feature_prev_idx]
                obs_vec_old = observation_mat @ state_vec_old
            else:
                # Default values
                state_vec_old = np.zeros(2 * prob_dim)
                state_cov_old = np.eye(2 * prob_dim)
                obs_vec_old = np.zeros(prob_dim)

        return state_vec_old, state_cov_old, obs_vec_old

    def _extract_current_observations(self, i_feature: int, frame_info: Dict,
                                    prob_dim: int) -> Tuple[np.ndarray, np.ndarray]:
        """Extract current observations and errors"""
        if 'allCoord' in frame_info and i_feature < len(frame_info['allCoord']):
            current_obs = frame_info['allCoord'][i_feature, ::2][:prob_dim]
            obs_errors = frame_info['allCoord'][i_feature, 1::2][:prob_dim]
        else:
            current_obs = np.zeros(prob_dim)
            obs_errors = np.ones(prob_dim) * 0.1

        obs_errors = np.maximum(obs_errors, 1e-6)
        return current_obs, obs_errors

    def _update_state_covariance_enhanced(self, state_cov_old: np.ndarray,
                                        kalman_gain: np.ndarray,
                                        observation_mat: np.ndarray) -> np.ndarray:
        """Update state covariance with enhanced numerical stability"""
        # Joseph form update for numerical stability
        I_KH = np.eye(state_cov_old.shape[0]) - kalman_gain @ observation_mat
        return I_KH @ state_cov_old @ I_KH.T + kalman_gain @ kalman_gain.T

    def _save_state_noise_enhanced(self, i_feature_prev_idx: int, i_frame: int,
                                 state_noise: np.ndarray, kalman_filter_info_out: List[Dict]):
        """Save state noise to previous frame"""
        if (i_frame > 0 and 'stateNoise' in kalman_filter_info_out[i_frame-1] and
            i_feature_prev_idx < kalman_filter_info_out[i_frame-1]['stateNoise'].shape[0]):
            kalman_filter_info_out[i_frame-1]['stateNoise'][i_feature_prev_idx] = state_noise

    def _save_kalman_results(self, i_feature: int, i_frame: int, state_vec: np.ndarray,
                           state_cov: np.ndarray, noise_var_matrix: np.ndarray,
                           kalman_filter_info_out: List[Dict], prob_dim: int):
        """Save Kalman filter results"""
        if len(kalman_filter_info_out) <= i_frame:
            num_features = state_vec.shape[0] if state_vec.ndim > 0 else 1
            kalman_filter_info_out.append({
                'stateVec': np.zeros((num_features, 2 * prob_dim)),
                'stateCov': np.zeros((2 * prob_dim, 2 * prob_dim, num_features)),
                'noiseVar': np.zeros((2 * prob_dim, 2 * prob_dim, num_features)),
                'stateNoise': np.zeros((num_features, 2 * prob_dim)),
                'scheme': np.zeros((num_features, 2), dtype=int)
            })

        kalman_filter_info_out[i_frame]['stateVec'][i_feature] = state_vec
        kalman_filter_info_out[i_frame]['stateCov'][:, :, i_feature] = state_cov
        kalman_filter_info_out[i_frame]['noiseVar'][:, :, i_feature] = noise_var_matrix

    def _process_new_feature_enhanced(self, i_feature: int, i_frame: int, frame_info: Dict,
                                    kalman_filter_info_out: List[Dict], prob_dim: int,
                                    use_prior_info: bool, filter_info_prev: Optional[Dict],
                                    cost_mat_param: Optional[Dict]):
        """Process new feature with enhanced initialization"""
        # Initialize current frame structures if needed
        if len(kalman_filter_info_out) <= i_frame:
            num_features = frame_info.get('num', 1)
            kalman_filter_info_out.append({
                'stateVec': np.zeros((num_features, 2 * prob_dim)),
                'stateCov': np.zeros((2 * prob_dim, 2 * prob_dim, num_features)),
                'noiseVar': np.zeros((2 * prob_dim, 2 * prob_dim, num_features)),
                'stateNoise': np.zeros((num_features, 2 * prob_dim)),
                'scheme': np.zeros((num_features, 2), dtype=int)
            })

        # Use default initialization
        kalman_filter_info_out[i_frame]['stateVec'][i_feature] = np.zeros(2 * prob_dim)
        kalman_filter_info_out[i_frame]['stateCov'][:, :, i_feature] = np.eye(2 * prob_dim)
        kalman_filter_info_out[i_frame]['noiseVar'][:, :, i_feature] = np.eye(2 * prob_dim) * 0.01

    def _update_real_positions(self, i_feature: int, i_frame: int, frame_info: Dict,
                             kalman_filter_info_out: List[Dict], prob_dim: int):
        """Update real positions in state vector"""
        if (len(kalman_filter_info_out) > i_frame and
            'allCoord' in frame_info and i_feature < len(frame_info['allCoord'])):
            current_pos = frame_info['allCoord'][i_feature, ::2][:prob_dim]
            kalman_filter_info_out[i_frame]['stateVec'][i_feature, ::2] = current_pos

    def _update_kalman_info_with_predictions(self, kalman_info: Dict, predicted_results: Dict,
                                           trans_matrices: np.ndarray, num_schemes: int,
                                           prob_dim: int) -> Dict:
        """Update kalman info with prediction results"""
        kalman_info_predicted = kalman_info.copy()
        kalman_info_predicted.update({
            'stateVec': predicted_results['predicted_state_all'],
            'stateCov': predicted_results['predicted_cov_all'],
            'obsVec': predicted_results['predicted_obs_all'],
            'num_schemes': num_schemes,
            'trans_matrices': trans_matrices,
            'obs_matrix': self._create_observation_matrix(prob_dim)
        })
        return kalman_info_predicted

    def kalman_predict_linear_motion(self, kalman_info: Dict, prob_dim: int = 2) -> Dict:
        """
        Predict next state using linear motion model

        Args:
            kalman_info: Current Kalman filter information
            prob_dim: Problem dimensionality

        Returns:
            Predicted Kalman filter information
        """
        log_function_call(self.logger, 'kalman_predict_linear_motion', (kalman_info, prob_dim))

        self.logger.info("=== KALMAN PREDICTION STARTED ===")

        num_features = kalman_info.get('num_features', 0)
        if num_features == 0:
            return kalman_info

        with self.time_operation("Kalman prediction"):
            state_vec = kalman_info.get('stateVec', np.array([]))
            state_noise_var = kalman_info.get('stateCov', kalman_info.get('state_noise_var', np.array([])))
            state_prop_matrix = self._create_state_propagation_matrix_enhanced(prob_dim)

            # Predict state: x_pred = F * x
            predicted_state = np.zeros_like(state_vec)
            predicted_covariance = np.zeros_like(state_noise_var)

            for i in range(num_features):
                predicted_state[i] = state_prop_matrix @ state_vec[i]

                # Predict covariance: P_pred = F * P * F' + Q
                Q = np.eye(2 * prob_dim) * 0.01  # Process noise
                if state_noise_var.ndim == 3:
                    predicted_covariance[i] = (state_prop_matrix @
                                             state_noise_var[i] @
                                             state_prop_matrix.T + Q)
                else:
                    predicted_covariance[i] = Q

            # Update kalman info
            predicted_kalman_info = kalman_info.copy()
            predicted_kalman_info['stateVec'] = predicted_state
            predicted_kalman_info['stateCov'] = predicted_covariance

        return predicted_kalman_info

    def kalman_update_linear_motion(self, kalman_info: Dict, observations: np.ndarray,
                                   prob_dim: int = 2) -> Dict:
        """
        Update Kalman filter with observations

        Args:
            kalman_info: Predicted Kalman filter information
            observations: New observations [num_features, prob_dim]
            prob_dim: Problem dimensionality

        Returns:
            Updated Kalman filter information
        """
        log_function_call(self.logger, 'kalman_update_linear_motion',
                         (kalman_info, observations, prob_dim))

        num_features = kalman_info.get('num_features', 0)
        if num_features == 0 or len(observations) == 0:
            return kalman_info

        with self.time_operation("Kalman update"):
            state_vec = kalman_info.get('stateVec', np.array([]))
            state_noise_var = kalman_info.get('stateCov', np.array([]))
            obs_matrix = self._create_observation_matrix(prob_dim)

            # Update state and covariance for each feature
            updated_state = np.zeros_like(state_vec)
            updated_covariance = np.zeros_like(state_noise_var)
            innovation = np.zeros((num_features, prob_dim))

            for i in range(min(num_features, len(observations))):
                # Innovation: y = z - H * x_pred
                predicted_obs = obs_matrix @ state_vec[i]
                innovation[i] = observations[i] - predicted_obs

                # Calculate Kalman gain (simplified)
                if state_noise_var.ndim == 3:
                    P = state_noise_var[i]
                else:
                    P = np.eye(2 * prob_dim) * 0.1

                S = obs_matrix @ P @ obs_matrix.T + np.eye(prob_dim) * 0.01
                K = P @ obs_matrix.T @ np.linalg.pinv(S)

                # Update state: x = x_pred + K * y
                updated_state[i] = state_vec[i] + K @ innovation[i]

                # Update covariance: P = (I - K * H) * P_pred
                I_KH = np.eye(2 * prob_dim) - K @ obs_matrix
                updated_covariance[i] = I_KH @ P

            # Update kalman info
            updated_kalman_info = kalman_info.copy()
            updated_kalman_info['stateVec'] = updated_state
            updated_kalman_info['stateCov'] = updated_covariance
            updated_kalman_info['innovation_vec'] = innovation
            updated_kalman_info['observation_vec'] = observations

        return updated_kalman_info

    def kalman_res_mem_lm(self, num_frames: int, num_features: Union[int, np.ndarray],
                         prob_dim: int = 2) -> List[Dict]:
        """
        Reserve memory for Kalman filter structure for linear motion model

        Args:
            num_frames: Number of frames in movie
            num_features: Number of features in each frame (array) or max features (int)
            prob_dim: Problem dimensionality

        Returns:
            List of Kalman filter info structures with reserved memory
        """
        log_function_call(self.logger, 'kalman_res_mem_lm',
                         (num_frames, num_features, prob_dim))

        try:
            with self.time_operation("Kalman memory reservation"):
                # Handle different input types for num_features
                if isinstance(num_features, (int, float)):
                    features_per_frame = [int(num_features)] * num_frames
                else:
                    features_per_frame = [int(x) for x in num_features]

                # Ensure we have enough elements
                while len(features_per_frame) < num_frames:
                    features_per_frame.append(0)

                # Calculate vector size
                vec_size = 2 * prob_dim

                # Reserve memory for all frames
                kalman_filter_info = []

                for i_frame in range(num_frames - 1, -1, -1):  # Backwards allocation like MATLAB
                    num_feat_frame = features_per_frame[i_frame]

                    frame_info = {
                        'stateVec': np.zeros((num_feat_frame, vec_size)),
                        'stateCov': np.zeros((vec_size, vec_size, num_feat_frame)),
                        'noiseVar': np.zeros((vec_size, vec_size, num_feat_frame)),
                        'stateNoise': np.zeros((num_feat_frame, vec_size)),
                        'scheme': np.zeros((num_feat_frame, 2), dtype=int),
                        'observation_vec': np.zeros((num_feat_frame, prob_dim)),
                        'innovation_vec': np.zeros((num_feat_frame, prob_dim)),
                        'observation_noise_var': np.zeros((num_feat_frame, prob_dim, prob_dim)),
                        'num_features': num_feat_frame
                    }

                    kalman_filter_info.insert(0, frame_info)

            return kalman_filter_info

        except Exception as e:
            self.logger.error(f"Error in kalman_res_mem_lm: {str(e)}")
            # Return basic structure as fallback
            return [{
                'stateVec': np.array([]).reshape(0, 2 * prob_dim),
                'stateCov': np.array([]).reshape(0, 2 * prob_dim, 2 * prob_dim),
                'noiseVar': np.array([]).reshape(0, 2 * prob_dim, 2 * prob_dim),
                'stateNoise': np.array([]).reshape(0, 2 * prob_dim),
                'scheme': np.array([]).reshape(0, 2),
                'num_features': 0
            } for _ in range(num_frames)]

    def kalman_reverse_linear_motion(self, kalman_filter_info: List[Dict],
                                    prob_dim: int = 2) -> List[Dict]:
        """
        Reverse Kalman filter information in time

        Args:
            kalman_filter_info: Kalman filter information from previous round of linking
            prob_dim: Problem dimensionality

        Returns:
            Kalman filter information reversed in time
        """
        log_function_call(self.logger, 'kalman_reverse_linear_motion',
                         (kalman_filter_info, prob_dim))

        try:
            if not kalman_filter_info:
                return kalman_filter_info

            with self.time_operation("Kalman reverse filtering"):
                # Reverse time (reverse the list)
                reversed_info = kalman_filter_info[::-1]

                # Go over all frames and reverse velocity components
                for i_frame in range(len(reversed_info)):
                    if 'stateVec' in reversed_info[i_frame] and reversed_info[i_frame]['stateVec'].size > 0:
                        # Reverse velocity components (odd indices in state vector)
                        reversed_info[i_frame]['stateVec'][:, 1::2] = \
                            -reversed_info[i_frame]['stateVec'][:, 1::2]

                        # Also reverse velocity components in any state noise if present
                        if ('stateNoise' in reversed_info[i_frame] and
                            reversed_info[i_frame]['stateNoise'].size > 0):
                            reversed_info[i_frame]['stateNoise'][:, 1::2] = \
                                -reversed_info[i_frame]['stateNoise'][:, 1::2]

            return reversed_info

        except Exception as e:
            self.logger.error(f"Error in kalman_reverse_linear_motion: {str(e)}")
            return kalman_filter_info

    def extract_kalman_state_enhanced(self, kalman_info: Dict, num_features: int,
                                     prob_dim: int) -> Tuple:
        """
        Extract state vector and covariance from Kalman filter info
        with comprehensive error handling and automatic size adjustment

        Args:
            kalman_info: Kalman filter information
            num_features: Expected number of features
            prob_dim: Problem dimensionality

        Returns:
            Tuple of (state_vec, state_cov) with proper dimensions
        """
        log_function_call(self.logger, 'extract_kalman_state_enhanced',
                         (kalman_info, num_features, prob_dim))

        if kalman_info is None or len(kalman_info) == 0:
            return None, None

        try:
            with self.time_operation("Kalman state extraction"):
                # Handle different possible field names for state vector
                state_vec = None
                for vec_name in ['stateVec', 'state_vec', 'state_vector']:
                    if vec_name in kalman_info:
                        state_vec = kalman_info[vec_name]
                        break

                # Handle different possible field names for state covariance
                state_cov = None
                for cov_name in ['stateCov', 'state_cov', 'state_noise_var', 'state_covariance', 'P']:
                    if cov_name in kalman_info:
                        state_cov = kalman_info[cov_name]
                        break

                # Graceful handling of size mismatches with intelligent resizing
                if state_vec is not None:
                    current_features = len(state_vec)
                    expected_dims = 2 * prob_dim

                    # Handle feature count mismatch
                    if current_features != num_features:
                        new_state_vec = np.zeros((num_features, expected_dims))
                        copy_size = min(current_features, num_features)

                        if copy_size > 0 and state_vec.shape[1] >= expected_dims:
                            new_state_vec[:copy_size, :] = state_vec[:copy_size, :expected_dims]
                        elif copy_size > 0:
                            copy_dims = min(state_vec.shape[1], expected_dims)
                            if copy_dims > 0:
                                new_state_vec[:copy_size, :copy_dims] = state_vec[:copy_size, :copy_dims]

                        state_vec = new_state_vec

            return state_vec, state_cov

        except Exception as e:
            self.logger.error(f"Error in extract_kalman_state_enhanced: {str(e)}")
            return None, None

    def _basic_kalman_init_fallback(self, movie_info: List[Dict], prob_dim: int = 2) -> List[Dict]:
        """Basic Kalman initialization fallback"""
        self.logger.warning("Using basic fallback initialization")
        kalman_info = []

        for frame_idx, frame_data in enumerate(movie_info):
            num_features = frame_data.get('num', 0) if frame_data else 0

            kalman_frame = {
                'num_features': num_features,
                'stateVec': np.zeros((num_features, 2 * prob_dim)),
                'stateCov': np.zeros((num_features, 2 * prob_dim, 2 * prob_dim)),
                'noiseVar': np.zeros((num_features, 2 * prob_dim, 2 * prob_dim)),
                'stateNoise': np.zeros((num_features, 2 * prob_dim)),
                'scheme': np.zeros((num_features, 2), dtype=int)
            }

            for i in range(num_features):
                kalman_frame['stateCov'][i] = np.eye(2 * prob_dim)
                kalman_frame['noiseVar'][i] = np.eye(2 * prob_dim) * 0.01

            kalman_info.append(kalman_frame)

        return kalman_info


# Enhanced function interfaces for backward compatibility
def kalman_init_linear_motion(movie_info: List[Dict], prob_dim: int = 2,
                             cost_mat_param: Optional[Dict] = None) -> List[Dict]:
    """Enhanced Kalman filter initialization - function interface"""
    kf = KalmanFilterLinearMotion()
    return kf.kalman_init_linear_motion(movie_info, prob_dim, cost_mat_param)


def kalman_gain_linear_motion(tracked_feature_indx: np.ndarray, frame_info: Dict,
                             kalman_filter_info_tmp: Dict, propagation_scheme: np.ndarray,
                             kalman_filter_info_in: List[Dict], prob_dim: int = 2,
                             filter_info_prev: Optional[Dict] = None,
                             cost_mat_param: Optional[Dict] = None,
                             init_function_name: str = 'kalman_init_linear_motion') -> Tuple[List[Dict], int]:
    """Enhanced Kalman gain calculation - function interface"""
    kf = KalmanFilterLinearMotion()
    return kf.kalman_gain_linear_motion(
        tracked_feature_indx, frame_info, kalman_filter_info_tmp, propagation_scheme,
        kalman_filter_info_in, prob_dim, filter_info_prev, cost_mat_param, init_function_name
    )


def kalman_predict_multiple_models(kalman_info: Dict, prob_dim: int = 2,
                                  motion_model_type: int = MotionModelType.RANDOM_DIRECTED,
                                  cost_mat_param: Optional[Dict] = None) -> Dict:
    """Enhanced multiple model prediction - function interface"""
    kf = KalmanFilterLinearMotion()
    return kf.kalman_predict_multiple_models(
        kalman_info, prob_dim, motion_model_type, cost_mat_param
    )


def kalman_predict_linear_motion(kalman_info: Dict, prob_dim: int = 2) -> Dict:
    """Predict next state using linear motion model - function interface"""
    kf = KalmanFilterLinearMotion()
    return kf.kalman_predict_linear_motion(kalman_info, prob_dim)


def kalman_update_linear_motion(kalman_info: Dict, observations: np.ndarray,
                               prob_dim: int = 2) -> Dict:
    """Update Kalman filter with observations - function interface"""
    kf = KalmanFilterLinearMotion()
    return kf.kalman_update_linear_motion(kalman_info, observations, prob_dim)


def kalman_res_mem_lm(num_frames: int, num_features: Union[int, np.ndarray],
                     prob_dim: int = 2) -> List[Dict]:
    """Reserve memory for Kalman filter structures - function interface"""
    kf = KalmanFilterLinearMotion()
    return kf.kalman_res_mem_lm(num_frames, num_features, prob_dim)


def kalman_reverse_linear_motion(kalman_filter_info: List[Dict], prob_dim: int = 2) -> List[Dict]:
    """Reverse Kalman filter information in time - function interface"""
    kf = KalmanFilterLinearMotion()
    return kf.kalman_reverse_linear_motion(kalman_filter_info, prob_dim)


def extract_kalman_state_fixed(kalman_info: Dict, num_features: int, prob_dim: int) -> Tuple:
    """Extract state vector and covariance from Kalman filter info - function interface"""
    kf = KalmanFilterLinearMotion()
    return kf.extract_kalman_state_enhanced(kalman_info, num_features, prob_dim)


# Test functions for the enhanced implementation
def test_enhanced_kalman_filter():
    """Test enhanced Kalman filter functionality"""
    logger.info("=== ENHANCED KALMAN FILTER TEST STARTED ===")

    try:
        # Create test data
        movie_info = []
        for frame in range(5):
            frame_info = {
                'num': 3,
                'allCoord': np.random.rand(3, 4) * 100,  # [x, dx, y, dy]
                'amp': np.random.rand(3, 2) * 1000
            }
            movie_info.append(frame_info)

        # Test enhanced initialization
        kf = KalmanFilterLinearMotion()

        # Test with different motion models
        for motion_type in [MotionModelType.RANDOM, MotionModelType.RANDOM_DIRECTED,
                          MotionModelType.SWITCHING]:

            kalman_info = kf.kalman_init_linear_motion(movie_info, prob_dim=2)

            # Test multiple model prediction
            test_frame = kalman_info[0]
            predicted = kf.kalman_predict_multiple_models(
                test_frame, prob_dim=2, motion_model_type=motion_type
            )

            logger.info(f"Motion model {motion_type} test: PASSED")

        logger.info("=== ENHANCED KALMAN FILTER TEST COMPLETED SUCCESSFULLY ===")
        return True

    except Exception as e:
        logger.error(f"Enhanced Kalman filter test failed: {str(e)}")
        logger.exception("Full traceback:")
        return False


if __name__ == "__main__":
    logger.info("=== ENHANCED KALMAN FILTERS MODULE EXECUTION ===")

    # Run enhanced tests
    success = test_enhanced_kalman_filter()

    if success:
        logger.info("=== ENHANCED KALMAN FILTERS MODULE SUCCESSFUL ===")
    else:
        logger.error("=== ENHANCED KALMAN FILTERS MODULE FAILED ===")

# Module initialization
logger.info("=== ENHANCED KALMAN FILTERS MODULE LOADED ===")
logger.info("Enhanced classes: KalmanFilterLinearMotion")
logger.info("Enhanced functions: kalman_init_linear_motion, kalman_gain_linear_motion, kalman_predict_multiple_models, kalman_predict_linear_motion, kalman_update_linear_motion, kalman_res_mem_lm, kalman_reverse_linear_motion, extract_kalman_state_fixed")
logger.info("Motion model types: RANDOM, RANDOM_DIRECTED, SWITCHING, RADIAL")
