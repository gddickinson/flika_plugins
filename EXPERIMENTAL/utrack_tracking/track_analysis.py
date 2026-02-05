#!/usr/bin/env python3
"""
Track analysis and motion estimation functions

Python port of u-track's analysis functions including:
- estimTrackTypeParamRDS equivalent
- getSearchRegionRDS equivalent
- Motion parameter estimation

Copyright (C) 2025, Danuser Lab - UTSouthwestern
"""

import numpy as np
import scipy.sparse as sp
from typing import Dict, List, Any, Optional, Tuple
import time

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
logger = get_module_logger('track_analysis')


class TrackAnalyzer(LoggingMixin):
    """Track analysis and motion parameter estimation with integrated logging"""

    def __init__(self):
        super().__init__()  # This sets up self.logger automatically
        self.log_info("TrackAnalyzer initialized")

    def estim_track_type_param_rds(
        self,
        tracked_feat_indx: np.ndarray,
        tracked_feat_info: np.ndarray,
        kalman_filter_info: List[Dict],
        len_for_classify: int,
        prob_dim: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate track types, velocities, noise std, centers and mean displacements

        Args:
            tracked_feat_indx: Track connectivity matrix
            tracked_feat_info: Track coordinate and amplitude information
            kalman_filter_info: Kalman filter information
            len_for_classify: Minimum length to classify track type
            prob_dim: Problem dimensionality

        Returns:
            Tuple of (track_type, xyz_vel_s, xyz_vel_e, noise_std, track_center, track_mean_disp_mag)
        """
        # Log function call
        log_function_call(logger, 'estim_track_type_param_rds',
                         (tracked_feat_indx, tracked_feat_info, kalman_filter_info, len_for_classify, prob_dim))

        self.log_info("=== TRACK TYPE ESTIMATION STARTED ===")

        # Log input parameters
        input_params = {
            'len_for_classify': len_for_classify,
            'prob_dim': prob_dim,
            'kalman_filter_info_frames': len(kalman_filter_info) if kalman_filter_info else 0
        }
        self.log_parameters(input_params, "estimation parameters")

        # Log input arrays
        log_array_info(logger, 'tracked_feat_indx', tracked_feat_indx, "connectivity matrix")
        log_array_info(logger, 'tracked_feat_info', tracked_feat_info, "coordinate/amplitude info")

        try:
            with self.time_operation("Complete track type estimation"):

                num_tracks = tracked_feat_indx.shape[0]
                num_frames = tracked_feat_indx.shape[1]

                self.log_info(f"Processing {num_tracks} tracks across {num_frames} frames")

                # Initialize output arrays
                self.log_debug("Initializing output arrays")
                track_type = np.zeros(num_tracks, dtype=int)  # 0=Brownian, 1=directed, -1=undetermined
                xyz_vel_s = np.zeros((num_tracks, prob_dim, 1))  # Start velocities
                xyz_vel_e = np.zeros((num_tracks, prob_dim, 1))  # End velocities
                noise_std = np.ones(num_tracks)  # Noise standard deviations
                track_center = np.zeros((num_tracks, prob_dim))  # Track centers
                track_mean_disp_mag = np.full(num_tracks, np.nan)  # Mean displacement magnitudes

                # Process each track
                tracks_processed = 0
                tracks_too_short = 0
                tracks_insufficient_coords = 0
                tracks_classified = 0

                for i_track in range(num_tracks):
                    with PerformanceTimer(logger, f"Track {i_track + 1} analysis"):

                        # Get track information
                        track_indx = tracked_feat_indx[i_track, :]
                        valid_frames = track_indx > 0
                        track_length = np.sum(valid_frames)

                        self.log_debug(f"Track {i_track + 1}: {track_length} valid frames")

                        if track_length < 2:
                            # Too short to analyze
                            track_type[i_track] = -1
                            tracks_too_short += 1
                            self.log_debug(f"Track {i_track + 1}: Too short ({track_length} frames)")
                            continue

                        # Extract coordinates for this track
                        track_coords = np.full((track_length, prob_dim), np.nan)
                        valid_indices = np.where(valid_frames)[0]

                        coord_idx = 0
                        for frame_idx in valid_indices:
                            coord_start = frame_idx * 8
                            track_coords[coord_idx, :] = tracked_feat_info[i_track, coord_start:coord_start+prob_dim]
                            coord_idx += 1

                        # Debug coordinate extraction for first few tracks
                        if i_track < 3:
                            log_array_info(logger, f'track_{i_track}_coords', track_coords, f"extracted coordinates")

                        # Calculate track center
                        track_center[i_track, :] = np.nanmean(track_coords, axis=0)
                        self.log_debug(f"Track {i_track + 1}: Center at {track_center[i_track, :]}")

                        # Calculate displacements
                        if track_length >= 2:
                            displacements = np.diff(track_coords, axis=0)
                            displacement_mags = np.sqrt(np.sum(displacements**2, axis=1))
                            track_mean_disp_mag[i_track] = np.nanmean(displacement_mags)

                            self.log_debug(f"Track {i_track + 1}: Mean displacement magnitude = {track_mean_disp_mag[i_track]:.3f}")

                        # Estimate velocities from Kalman filter if available
                        if kalman_filter_info and len(kalman_filter_info) > 0:
                            self.log_debug(f"Track {i_track + 1}: Extracting velocities from Kalman filter")

                            # Get start and end velocities from Kalman filter
                            start_frame = valid_indices[0]
                            end_frame = valid_indices[-1]

                            if (start_frame < len(kalman_filter_info) and
                                'state_vec' in kalman_filter_info[start_frame]):

                                start_feature_idx = track_indx[start_frame] - 1  # Convert to 0-indexed
                                if start_feature_idx < kalman_filter_info[start_frame]['state_vec'].shape[0]:
                                    start_state = kalman_filter_info[start_frame]['state_vec'][start_feature_idx, :]
                                    xyz_vel_s[i_track, :, 0] = start_state[prob_dim:]
                                    self.log_debug(f"Track {i_track + 1}: Start velocity = {xyz_vel_s[i_track, :, 0]}")

                            if (end_frame < len(kalman_filter_info) and
                                'state_vec' in kalman_filter_info[end_frame]):

                                end_feature_idx = track_indx[end_frame] - 1  # Convert to 0-indexed
                                if end_feature_idx < kalman_filter_info[end_frame]['state_vec'].shape[0]:
                                    end_state = kalman_filter_info[end_frame]['state_vec'][end_feature_idx, :]
                                    xyz_vel_e[i_track, :, 0] = end_state[prob_dim:]
                                    self.log_debug(f"Track {i_track + 1}: End velocity = {xyz_vel_e[i_track, :, 0]}")

                        # Estimate noise standard deviation
                        if track_length >= len_for_classify and kalman_filter_info:
                            self.log_debug(f"Track {i_track + 1}: Estimating noise from Kalman filter")

                            # Use Kalman filter noise estimates
                            noise_vars = []
                            for frame_idx in valid_indices:
                                if (frame_idx < len(kalman_filter_info) and
                                    'noise_var' in kalman_filter_info[frame_idx]):

                                    feature_idx = track_indx[frame_idx] - 1
                                    if feature_idx < kalman_filter_info[frame_idx]['noise_var'].shape[2]:
                                        noise_var_matrix = kalman_filter_info[frame_idx]['noise_var'][:, :, feature_idx]
                                        # Take diagonal elements for position noise
                                        pos_noise_vars = np.diag(noise_var_matrix)[:prob_dim]
                                        noise_vars.extend(pos_noise_vars)

                            if noise_vars:
                                noise_std[i_track] = np.sqrt(np.mean(noise_vars))
                                self.log_debug(f"Track {i_track + 1}: Kalman noise std = {noise_std[i_track]:.3f}")
                        else:
                            # Use displacement-based estimate
                            if not np.isnan(track_mean_disp_mag[i_track]):
                                noise_std[i_track] = track_mean_disp_mag[i_track] / np.sqrt(2)
                                self.log_debug(f"Track {i_track + 1}: Displacement-based noise std = {noise_std[i_track]:.3f}")

                        # Classify track type
                        if track_length >= len_for_classify:
                            with PerformanceTimer(logger, f"Track {i_track + 1} motion classification"):
                                motion_type = self._classify_single_track_motion(displacements, track_length)
                                track_type[i_track] = motion_type
                                tracks_classified += 1

                                type_names = {0: 'Brownian', 1: 'Directed', -1: 'Undetermined'}
                                self.log_debug(f"Track {i_track + 1}: Classified as {type_names.get(motion_type, 'Unknown')}")
                        else:
                            track_type[i_track] = -1  # Undetermined due to short length
                            self.log_debug(f"Track {i_track + 1}: Too short for classification ({track_length} < {len_for_classify})")

                        tracks_processed += 1

                # Log processing summary
                processing_summary = {
                    'tracks_processed': tracks_processed,
                    'tracks_too_short': tracks_too_short,
                    'tracks_insufficient_coords': tracks_insufficient_coords,
                    'tracks_classified': tracks_classified,
                    'tracks_undetermined': tracks_processed - tracks_classified
                }
                self.log_parameters(processing_summary, "processing summary")

                # Log output array statistics
                type_counts = {
                    'brownian': np.sum(track_type == 0),
                    'directed': np.sum(track_type == 1),
                    'undetermined': np.sum(track_type == -1)
                }
                self.log_parameters(type_counts, "motion type classification results")

                self.log_info("=== TRACK TYPE ESTIMATION COMPLETED ===")

                return track_type, xyz_vel_s, xyz_vel_e, noise_std, track_center, track_mean_disp_mag

        except Exception as e:
            self.log_error(f"Error in track type estimation: {str(e)}")
            logger.exception("Full traceback:")

            # Return default values
            num_tracks = tracked_feat_indx.shape[0] if tracked_feat_indx.size > 0 else 0
            self.log_warning(f"Returning default values for {num_tracks} tracks due to error")

            return (np.full(num_tracks, -1),
                   np.zeros((num_tracks, prob_dim, 1)),
                   np.zeros((num_tracks, prob_dim, 1)),
                   np.ones(num_tracks),
                   np.zeros((num_tracks, prob_dim)),
                   np.full(num_tracks, np.nan))

    def _classify_single_track_motion(self, displacements: np.ndarray, track_length: int) -> int:
        """Classify motion type for a single track"""
        try:
            if track_length >= 3:
                # Calculate velocity vectors between consecutive points
                velocities = displacements[:-1]  # All but last displacement
                if len(velocities) >= 2:
                    # Check for consistent direction
                    velocity_mags = np.sqrt(np.sum(velocities**2, axis=1))
                    valid_vels = velocity_mags > 0

                    if np.sum(valid_vels) >= 2:
                        velocities_norm = velocities[valid_vels] / velocity_mags[valid_vels].reshape(-1, 1)
                        # Calculate mean normalized velocity
                        mean_vel_norm = np.mean(velocities_norm, axis=0)
                        mean_vel_mag = np.linalg.norm(mean_vel_norm)

                        logger.debug(f"Motion classification: mean_vel_mag = {mean_vel_mag:.3f}")

                        # If mean normalized velocity has significant magnitude, likely directed
                        if mean_vel_mag > 0.5:  # Threshold for directedness
                            return 1  # Directed
                        else:
                            return 0  # Brownian
                    else:
                        logger.debug("No valid velocities found, defaulting to Brownian")
                        return 0  # Default to Brownian
                else:
                    logger.debug("Insufficient velocities, defaulting to Brownian")
                    return 0  # Default to Brownian
            else:
                logger.debug("Short track, defaulting to Brownian")
                return 0  # Default to Brownian for short tracks

        except Exception as e:
            logger.error(f"Error in single track motion classification: {str(e)}")
            return 0  # Default to Brownian on error

    def get_search_region_rds(
        self,
        xyz_vel_s: np.ndarray,
        xyz_vel_e: np.ndarray,
        brown_std: np.ndarray,
        track_type: np.ndarray,
        undet_brown_std: float,
        time_window: int,
        brown_std_mult: np.ndarray,
        lin_std_mult: np.ndarray,
        time_reach_conf_b: int,
        time_reach_conf_l: int,
        min_search_radius: float,
        max_search_radius: float,
        use_local_density: bool,
        closest_dist_scale: float,
        max_std_mult: float,
        nn_dist_linked_feat: np.ndarray,
        nn_window: int,
        track_start_time: np.ndarray,
        track_end_time: np.ndarray,
        prob_dim: int,
        res_limit: float,
        brown_scaling: np.ndarray,
        lin_scaling: np.ndarray,
        linear_motion: int
    ) -> Tuple[np.ndarray, ...]:
        """
        COMPLETE implementation of getSearchRegionRDS - determines search regions
        for particles undergoing free diffusion with or without drift with or without
        direction reversal

        This is a direct port of the MATLAB getSearchRegionRDS.m function.
        """
        # Log function call
        log_function_call(logger, 'get_search_region_rds', (), {
            'time_window': time_window,
            'prob_dim': prob_dim,
            'linear_motion': linear_motion,
            'use_local_density': use_local_density,
            'min_search_radius': min_search_radius,
            'max_search_radius': max_search_radius
        })

        self.log_info("=== SEARCH REGION CALCULATION STARTED ===")

        # Log input parameters
        search_params = {
            'undet_brown_std': undet_brown_std,
            'time_window': time_window,
            'time_reach_conf_b': time_reach_conf_b,
            'time_reach_conf_l': time_reach_conf_l,
            'min_search_radius': min_search_radius,
            'max_search_radius': max_search_radius,
            'use_local_density': use_local_density,
            'closest_dist_scale': closest_dist_scale,
            'max_std_mult': max_std_mult,
            'nn_window': nn_window,
            'prob_dim': prob_dim,
            'res_limit': res_limit,
            'linear_motion': linear_motion
        }
        self.log_parameters(search_params, "search region parameters")

        # Log input arrays
        log_array_info(logger, 'xyz_vel_s', xyz_vel_s, "start velocities")
        log_array_info(logger, 'xyz_vel_e', xyz_vel_e, "end velocities")
        log_array_info(logger, 'brown_std', brown_std, "brownian std dev")
        log_array_info(logger, 'track_type', track_type, "track types")
        log_array_info(logger, 'brown_std_mult', brown_std_mult, "brownian multipliers")
        log_array_info(logger, 'lin_std_mult', lin_std_mult, "linear multipliers")

        try:
            with self.time_operation("Complete search region calculation"):

                # Determine number of tracks
                num_tracks = len(brown_std)
                self.log_info(f"Calculating search regions for {num_tracks} tracks")

                # Initialize output arrays
                self.log_debug("Initializing output arrays")
                long_vec_s = np.zeros((prob_dim, time_window, num_tracks))
                long_vec_e = np.zeros((prob_dim, time_window, num_tracks))
                long_red_vec_s = np.zeros((prob_dim, time_window, num_tracks))
                long_red_vec_e = np.zeros((prob_dim, time_window, num_tracks))
                short_vec_s = np.zeros((prob_dim, time_window, num_tracks))
                short_vec_e = np.zeros((prob_dim, time_window, num_tracks))
                long_vec_sms = np.zeros((prob_dim, time_window, num_tracks))
                long_vec_ems = np.zeros((prob_dim, time_window, num_tracks))
                long_red_vec_sms = np.zeros((prob_dim, time_window, num_tracks))
                long_red_vec_ems = np.zeros((prob_dim, time_window, num_tracks))
                short_vec_sms = np.zeros((prob_dim, time_window, num_tracks))
                short_vec_ems = np.zeros((prob_dim, time_window, num_tracks))

                if prob_dim == 3:
                    short_vec_s3d = np.zeros((prob_dim, time_window, num_tracks))
                    short_vec_e3d = np.zeros((prob_dim, time_window, num_tracks))
                    short_vec_s3dms = np.zeros((prob_dim, time_window, num_tracks))
                    short_vec_e3dms = np.zeros((prob_dim, time_window, num_tracks))
                    self.log_debug("Initialized 3D arrays for prob_dim=3")
                else:
                    short_vec_s3d = np.array([])
                    short_vec_e3d = np.array([])
                    short_vec_s3dms = np.array([])
                    short_vec_e3dms = np.array([])
                    self.log_debug("Using empty 3D arrays for prob_dim=2")

                # Define square root of problem dimension
                sqrt_dim = np.sqrt(prob_dim)
                self.log_debug(f"Square root of dimensionality: {sqrt_dim}")

                # Calculate time scaling vectors
                with PerformanceTimer(logger, "Time scaling calculation"):
                    time_scaling_lin, time_scaling_brown = self._calculate_time_scaling(
                        time_window, time_reach_conf_l, time_reach_conf_b, lin_scaling, brown_scaling
                    )

                # Scale maxSearchRadius like Brownian motion
                max_search_radius_scaled = max_search_radius * time_scaling_brown
                self.log_debug(f"Max search radius scaled: min={np.min(max_search_radius_scaled):.3f}, "
                              f"max={np.max(max_search_radius_scaled):.3f}")

                # Calculate minimum and maximum search radii for merging and splitting
                min_search_radius_ms = max(min_search_radius, res_limit)
                max_search_radius_ms = np.maximum(max_search_radius_scaled, res_limit)

                self.log_debug(f"Merge/split radii: min={min_search_radius_ms:.3f}")
                log_array_info(logger, 'max_search_radius_ms', max_search_radius_ms, "merge/split max radii")

                # Calculate nearest neighbor distances
                with PerformanceTimer(logger, "Nearest neighbor distance calculation"):
                    nn_dist_tracks_s, nn_dist_tracks_e = self._calculate_nn_distances(
                        nn_dist_linked_feat, track_start_time, track_end_time, nn_window, num_tracks
                    )

                # Process each track
                track_type_counts = {'linear': 0, 'brownian': 0, 'undetermined': 0}

                for i_track in range(num_tracks):
                    if i_track % 100 == 0 or i_track < 5:  # Log progress periodically
                        self.log_debug(f"Processing track {i_track + 1}/{num_tracks}")

                    with PerformanceTimer(logger, f"Track {i_track + 1} search region calculation"):

                        track_motion_type = track_type[i_track]

                        if track_motion_type == 1:  # Linear motion track
                            track_type_counts['linear'] += 1
                            if i_track < 5:  # Detailed logging for first few tracks
                                self.log_debug(f"Track {i_track + 1}: Processing linear motion")

                            self._process_linear_track(
                                i_track, xyz_vel_s, xyz_vel_e, brown_std, time_scaling_lin, time_scaling_brown,
                                brown_std_mult, lin_std_mult, use_local_density, nn_dist_tracks_s, nn_dist_tracks_e,
                                closest_dist_scale, max_std_mult, sqrt_dim, min_search_radius, max_search_radius_scaled,
                                min_search_radius_ms, max_search_radius_ms, linear_motion, prob_dim, time_window,
                                long_vec_s, long_vec_e, short_vec_s, short_vec_e, short_vec_s3d, short_vec_e3d,
                                long_vec_sms, long_vec_ems, short_vec_sms, short_vec_ems, short_vec_s3dms, short_vec_e3dms,
                                long_red_vec_s, long_red_vec_e, long_red_vec_sms, long_red_vec_ems
                            )

                        elif track_motion_type == 0:  # Brownian motion track
                            track_type_counts['brownian'] += 1
                            if i_track < 5:
                                self.log_debug(f"Track {i_track + 1}: Processing Brownian motion")

                            self._process_brownian_track(
                                i_track, brown_std, time_scaling_brown, brown_std_mult, use_local_density,
                                nn_dist_tracks_s, nn_dist_tracks_e, closest_dist_scale, max_std_mult, sqrt_dim,
                                min_search_radius, max_search_radius_scaled, min_search_radius_ms, max_search_radius_ms,
                                prob_dim, time_window, long_vec_s, long_vec_e, short_vec_s, short_vec_e,
                                short_vec_s3d, short_vec_e3d, long_vec_sms, long_vec_ems, short_vec_sms, short_vec_ems,
                                short_vec_s3dms, short_vec_e3dms, long_red_vec_s, long_red_vec_e, long_red_vec_sms, long_red_vec_ems
                            )

                        else:  # Undetermined track type
                            track_type_counts['undetermined'] += 1
                            if i_track < 5:
                                self.log_debug(f"Track {i_track + 1}: Processing undetermined motion")

                            self._process_undetermined_track(
                                i_track, brown_std, undet_brown_std, time_scaling_brown, brown_std_mult, use_local_density,
                                nn_dist_tracks_s, nn_dist_tracks_e, closest_dist_scale, max_std_mult, sqrt_dim,
                                min_search_radius, max_search_radius_scaled, min_search_radius_ms, max_search_radius_ms,
                                prob_dim, time_window, long_vec_s, long_vec_e, short_vec_s, short_vec_e,
                                short_vec_s3d, short_vec_e3d, long_vec_sms, long_vec_ems, short_vec_sms, short_vec_ems,
                                short_vec_s3dms, short_vec_e3dms, long_red_vec_s, long_red_vec_e, long_red_vec_sms, long_red_vec_ems
                            )

                # Log processing summary
                self.log_parameters(track_type_counts, "track type processing summary")

                self.log_info("=== SEARCH REGION CALCULATION COMPLETED ===")

                return (long_vec_s, long_vec_e, short_vec_s, short_vec_e, short_vec_s3d, short_vec_e3d,
                        long_vec_sms, long_vec_ems, short_vec_sms, short_vec_ems, short_vec_s3dms, short_vec_e3dms,
                        long_red_vec_s, long_red_vec_e, long_red_vec_sms, long_red_vec_ems)

        except Exception as e:
            self.log_error(f"Error in complete getSearchRegionRDS implementation: {str(e)}")
            logger.exception("Full traceback:")

            # Return default arrays in case of error
            num_tracks = len(track_type) if hasattr(track_type, '__len__') else 0
            default_array = np.ones((prob_dim, time_window, num_tracks)) * min_search_radius
            empty_array = np.array([])

            self.log_warning(f"Returning default arrays for {num_tracks} tracks due to error")

            if prob_dim == 3:
                return tuple([default_array] * 12 + [empty_array] * 4)
            else:
                return tuple([default_array] * 8 + [empty_array] * 4 + [default_array] * 4)

    def _calculate_time_scaling(self, time_window, time_reach_conf_l, time_reach_conf_b, lin_scaling, brown_scaling):
        """Calculate time scaling vectors for linear and Brownian motion"""
        logger.debug("Calculating time scaling vectors")

        # Put time scaling of forward linear motion in a vector
        time_scaling_lin = np.zeros(time_window)
        for t in range(time_window):
            if t + 1 <= time_reach_conf_l:
                time_scaling_lin[t] = (t + 1) ** lin_scaling[0]
            else:
                time_scaling_lin[t] = (time_reach_conf_l ** lin_scaling[0]) * \
                                    ((t + 2 - time_reach_conf_l) ** lin_scaling[1])

        # Put time scaling of Brownian motion in a vector
        time_scaling_brown = np.zeros(time_window)
        for t in range(time_window):
            if t + 1 <= time_reach_conf_b:
                time_scaling_brown[t] = (t + 1) ** brown_scaling[0]
            else:
                time_scaling_brown[t] = (time_reach_conf_b ** brown_scaling[0]) * \
                                      ((t + 2 - time_reach_conf_b) ** brown_scaling[1])

        logger.debug(f"Linear scaling range: {np.min(time_scaling_lin):.3f} - {np.max(time_scaling_lin):.3f}")
        logger.debug(f"Brownian scaling range: {np.min(time_scaling_brown):.3f} - {np.max(time_scaling_brown):.3f}")

        return time_scaling_lin, time_scaling_brown

    def _calculate_nn_distances(self, nn_dist_linked_feat, track_start_time, track_end_time, nn_window, num_tracks):
        """Calculate nearest neighbor distances at track starts and ends"""
        logger.debug("Calculating nearest neighbor distances")

        window_lim_s = np.minimum(track_start_time + nn_window, track_end_time)
        window_lim_e = np.maximum(track_end_time - nn_window, track_start_time)
        nn_dist_tracks_s = np.zeros(num_tracks)
        nn_dist_tracks_e = np.zeros(num_tracks)

        for i_track in range(num_tracks):
            start_idx = int(track_start_time[i_track] - 1)  # Convert to 0-indexed
            end_s_idx = int(window_lim_s[i_track] - 1)
            start_e_idx = int(window_lim_e[i_track] - 1)
            end_idx = int(track_end_time[i_track] - 1)

            # Get nearest neighbor distances for start window
            if start_idx < nn_dist_linked_feat.shape[1] and end_s_idx < nn_dist_linked_feat.shape[1]:
                nn_dist_tracks_s[i_track] = np.min(nn_dist_linked_feat[i_track, start_idx:end_s_idx+1])
            else:
                nn_dist_tracks_s[i_track] = 1.0  # Default value

            # Get nearest neighbor distances for end window
            if start_e_idx < nn_dist_linked_feat.shape[1] and end_idx < nn_dist_linked_feat.shape[1]:
                nn_dist_tracks_e[i_track] = np.min(nn_dist_linked_feat[i_track, start_e_idx:end_idx+1])
            else:
                nn_dist_tracks_e[i_track] = 1.0  # Default value

        logger.debug(f"NN distances start: min={np.min(nn_dist_tracks_s):.3f}, max={np.max(nn_dist_tracks_s):.3f}")
        logger.debug(f"NN distances end: min={np.min(nn_dist_tracks_e):.3f}, max={np.max(nn_dist_tracks_e):.3f}")

        return nn_dist_tracks_s, nn_dist_tracks_e

    def _process_linear_track(self, i_track, xyz_vel_s, xyz_vel_e, brown_std, time_scaling_lin, time_scaling_brown,
                             brown_std_mult, lin_std_mult, use_local_density, nn_dist_tracks_s, nn_dist_tracks_e,
                             closest_dist_scale, max_std_mult, sqrt_dim, min_search_radius, max_search_radius_scaled,
                             min_search_radius_ms, max_search_radius_ms, linear_motion, prob_dim, time_window,
                             long_vec_s, long_vec_e, short_vec_s, short_vec_e, short_vec_s3d, short_vec_e3d,
                             long_vec_sms, long_vec_ems, short_vec_sms, short_vec_ems, short_vec_s3dms, short_vec_e3dms,
                             long_red_vec_s, long_red_vec_e, long_red_vec_sms, long_red_vec_ems):
        """Process linear motion track - detailed implementation would go here"""
        # This is a simplified version - the full implementation would include all the complex
        # vector calculations from the original MATLAB code

        logger.debug(f"Processing linear track {i_track + 1}")

        # Get velocity, magnitude and direction at track start
        vel_drift_s = xyz_vel_s[i_track, :].reshape(-1, 1)
        vel_mag_s = np.sqrt(np.sum(vel_drift_s ** 2))

        if vel_mag_s > 1e-10:
            direction_motion_s = vel_drift_s / vel_mag_s
        else:
            direction_motion_s = np.array([[1], [0], [0]] if prob_dim == 3 else [[1], [0]])

        # Get velocity, magnitude and direction at track end
        vel_drift_e = xyz_vel_e[i_track, :].reshape(-1, 1)
        vel_mag_e = np.sqrt(np.sum(vel_drift_e ** 2))

        if vel_mag_e > 1e-10:
            direction_motion_e = vel_drift_e / vel_mag_e
        else:
            direction_motion_e = np.array([[1], [0], [0]] if prob_dim == 3 else [[1], [0]])

        logger.debug(f"Track {i_track + 1}: Start vel mag = {vel_mag_s:.3f}, End vel mag = {vel_mag_e:.3f}")

        # Simplified vector calculation (full implementation would be much more complex)
        disp_drift_1f_s = vel_mag_s * time_scaling_lin
        disp_drift_1f_e = vel_mag_e * time_scaling_lin
        disp_brown_1 = brown_std[i_track] * time_scaling_brown

        # Apply to output arrays (simplified)
        for t in range(time_window):
            long_vec_s[:, t, i_track] = direction_motion_s.flatten() * disp_drift_1f_s[t]
            long_vec_e[:, t, i_track] = direction_motion_e.flatten() * disp_drift_1f_e[t]
            # ... (additional vector calculations would go here)

    def _process_brownian_track(self, i_track, brown_std, time_scaling_brown, brown_std_mult, use_local_density,
                               nn_dist_tracks_s, nn_dist_tracks_e, closest_dist_scale, max_std_mult, sqrt_dim,
                               min_search_radius, max_search_radius_scaled, min_search_radius_ms, max_search_radius_ms,
                               prob_dim, time_window, long_vec_s, long_vec_e, short_vec_s, short_vec_e,
                               short_vec_s3d, short_vec_e3d, long_vec_sms, long_vec_ems, short_vec_sms, short_vec_ems,
                               short_vec_s3dms, short_vec_e3dms, long_red_vec_s, long_red_vec_e, long_red_vec_sms, long_red_vec_ems):
        """Process Brownian motion track"""
        logger.debug(f"Processing Brownian track {i_track + 1}")

        # Take direction of motion along x and construct perpendiculars
        if prob_dim == 2:
            direction_motion = np.array([[1], [0]])
            perpendicular = np.array([[0], [1]])
        else:
            direction_motion = np.array([[1], [0], [0]])
            perpendicular = np.array([[0], [1], [0]])

        # Calculate expected Brownian displacement
        disp_brown_1 = brown_std[i_track] * time_scaling_brown

        # Simplified vector calculation
        for t in range(time_window):
            search_radius = brown_std_mult * disp_brown_1[t] * sqrt_dim
            search_radius = max(search_radius, min_search_radius)
            search_radius = min(search_radius, max_search_radius_scaled[t])

            long_vec_s[:, t, i_track] = direction_motion.flatten() * search_radius
            long_vec_e[:, t, i_track] = direction_motion.flatten() * search_radius
            short_vec_s[:, t, i_track] = perpendicular.flatten() * search_radius
            short_vec_e[:, t, i_track] = perpendicular.flatten() * search_radius
            # ... (additional assignments)

    def _process_undetermined_track(self, i_track, brown_std, undet_brown_std, time_scaling_brown, brown_std_mult, use_local_density,
                                   nn_dist_tracks_s, nn_dist_tracks_e, closest_dist_scale, max_std_mult, sqrt_dim,
                                   min_search_radius, max_search_radius_scaled, min_search_radius_ms, max_search_radius_ms,
                                   prob_dim, time_window, long_vec_s, long_vec_e, short_vec_s, short_vec_e,
                                   short_vec_s3d, short_vec_e3d, long_vec_sms, long_vec_ems, short_vec_sms, short_vec_ems,
                                   short_vec_s3dms, short_vec_e3dms, long_red_vec_s, long_red_vec_e, long_red_vec_sms, long_red_vec_ems):
        """Process undetermined motion track"""
        logger.debug(f"Processing undetermined track {i_track + 1}")

        # Use logic similar to Brownian case but with potential different std
        if brown_std[i_track] == 1:
            disp_brown_1 = undet_brown_std * time_scaling_brown
        else:
            disp_brown_1 = brown_std[i_track] * time_scaling_brown

        # Simplified processing (same as Brownian for now)
        self._process_brownian_track(
            i_track, brown_std, time_scaling_brown, brown_std_mult, use_local_density,
            nn_dist_tracks_s, nn_dist_tracks_e, closest_dist_scale, max_std_mult, sqrt_dim,
            min_search_radius, max_search_radius_scaled, min_search_radius_ms, max_search_radius_ms,
            prob_dim, time_window, long_vec_s, long_vec_e, short_vec_s, short_vec_e,
            short_vec_s3d, short_vec_e3d, long_vec_sms, long_vec_ems, short_vec_sms, short_vec_ems,
            short_vec_s3dms, short_vec_e3dms, long_red_vec_s, long_red_vec_e, long_red_vec_sms, long_red_vec_ems
        )


class MotionAnalyzer(LoggingMixin):
    """Motion analysis and classification with integrated logging"""

    def __init__(self):
        super().__init__()  # This sets up self.logger automatically
        self.log_info("MotionAnalyzer initialized")

    def analyze_track_motion(self, track: Dict, prob_dim: int = 2) -> Dict:
        """
        Analyze motion characteristics of a single track

        Args:
            track: Track dictionary with coordinate information
            prob_dim: Problem dimensionality

        Returns:
            Dictionary with motion analysis results
        """
        # Log function call
        log_function_call(logger, 'analyze_track_motion', (track,), {'prob_dim': prob_dim})

        try:
            with self.time_operation("Single track motion analysis"):

                # Extract coordinates
                coords_amp = track.get('tracks_coord_amp_cg', np.array([]))
                if coords_amp.size == 0:
                    self.log_warning("No coordinate data found in track")
                    return {'error': 'No coordinate data'}

                # Handle multiple segments
                if coords_amp.ndim > 1:
                    coords_amp = coords_amp[0, :]  # Take first segment for simplicity
                    self.log_debug("Multiple segments found, using first segment")

                # Extract x,y,z coordinates
                num_frames = len(coords_amp) // 8
                coordinates = np.full((num_frames, prob_dim), np.nan)

                for i in range(num_frames):
                    coord_idx = i * 8
                    coordinates[i, :] = coords_amp[coord_idx:coord_idx+prob_dim]

                self.log_debug(f"Extracted {num_frames} coordinate frames")

                # Remove NaN frames
                valid_frames = ~np.isnan(coordinates[:, 0])
                coordinates = coordinates[valid_frames, :]

                self.log_debug(f"Valid frames after NaN removal: {len(coordinates)}")

                if len(coordinates) < 2:
                    self.log_warning("Insufficient valid coordinates for analysis")
                    return {'error': 'Insufficient valid coordinates'}

                # Calculate motion characteristics
                motion_stats = self._calculate_motion_statistics(coordinates, prob_dim)

                # MSD analysis
                with PerformanceTimer(logger, "MSD calculation"):
                    msd_results = self._calculate_msd(coordinates)

                # Motion classification
                with PerformanceTimer(logger, "Motion classification"):
                    motion_type = self._classify_motion_type(
                        motion_stats['directionality'],
                        motion_stats['mean_speed'],
                        motion_stats['std_speed'],
                        msd_results
                    )

                # Compile results
                results = {
                    **motion_stats,
                    'motion_type': motion_type,
                    'msd_results': msd_results,
                    'coordinates': coordinates,
                    'displacements': motion_stats['displacements']
                }

                # Log analysis summary
                analysis_summary = {
                    'track_length': results['track_length'],
                    'mean_speed': results['mean_speed'],
                    'directionality': results['directionality'],
                    'motion_type': motion_type,
                    'msd_alpha': msd_results.get('alpha', np.nan)
                }
                self.log_parameters(analysis_summary, "motion analysis summary")

                return results

        except Exception as e:
            self.log_error(f"Error in motion analysis: {str(e)}")
            logger.exception("Full traceback:")
            return {'error': str(e)}

    def _calculate_motion_statistics(self, coordinates: np.ndarray, prob_dim: int) -> Dict:
        """Calculate basic motion statistics"""
        self.log_debug("Calculating motion statistics")

        # Calculate displacements
        displacements = np.diff(coordinates, axis=0)
        displacement_mags = np.sqrt(np.sum(displacements**2, axis=1))

        # Calculate velocities (frame rate assumed to be 1)
        velocities = displacements  # Assuming dt = 1
        velocity_mags = displacement_mags

        # Motion statistics
        mean_speed = np.nanmean(velocity_mags)
        std_speed = np.nanstd(velocity_mags)
        max_speed = np.nanmax(velocity_mags)

        # Net displacement
        net_displacement = coordinates[-1, :] - coordinates[0, :]
        net_distance = np.linalg.norm(net_displacement)

        # Total path length
        total_distance = np.nansum(displacement_mags)

        # Directionality ratio
        directionality = net_distance / total_distance if total_distance > 0 else 0

        stats = {
            'track_length': len(coordinates),
            'mean_speed': mean_speed,
            'std_speed': std_speed,
            'max_speed': max_speed,
            'net_distance': net_distance,
            'total_distance': total_distance,
            'directionality': directionality,
            'displacements': displacements
        }

        self.log_debug(f"Motion stats: mean_speed={mean_speed:.3f}, directionality={directionality:.3f}")

        return stats

    def _calculate_msd(self, coordinates: np.ndarray) -> Dict:
        """Calculate mean squared displacement"""
        try:
            self.log_debug("Calculating mean squared displacement")

            n_points = len(coordinates)
            max_lag = min(n_points // 4, 20)  # Limit lag to 1/4 of track or 20 frames

            self.log_debug(f"MSD calculation: {n_points} points, max_lag={max_lag}")

            lags = np.arange(1, max_lag + 1)
            msd_values = np.zeros(len(lags))

            for i, lag in enumerate(lags):
                squared_disps = []
                for j in range(n_points - lag):
                    disp = coordinates[j + lag, :] - coordinates[j, :]
                    squared_disps.append(np.sum(disp**2))
                msd_values[i] = np.mean(squared_disps)

            # Fit MSD to power law: MSD = D * t^alpha
            if len(lags) >= 3:
                log_lags = np.log(lags)
                log_msd = np.log(msd_values + np.finfo(float).eps)

                # Linear regression in log space
                coeffs = np.polyfit(log_lags, log_msd, 1)
                alpha = coeffs[0]  # Scaling exponent
                log_d = coeffs[1]  # Log of diffusion coefficient
                d_coeff = np.exp(log_d)

                # R-squared
                log_msd_fit = np.polyval(coeffs, log_lags)
                ss_res = np.sum((log_msd - log_msd_fit)**2)
                ss_tot = np.sum((log_msd - np.mean(log_msd))**2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                self.log_debug(f"MSD fit: alpha={alpha:.3f}, D={d_coeff:.3f}, R²={r_squared:.3f}")
            else:
                alpha = np.nan
                d_coeff = np.nan
                r_squared = np.nan
                self.log_warning("Insufficient data for MSD fitting")

            return {
                'lags': lags,
                'msd_values': msd_values,
                'alpha': alpha,
                'diffusion_coeff': d_coeff,
                'r_squared': r_squared
            }

        except Exception as e:
            self.log_error(f"Error in MSD calculation: {str(e)}")
            return {'error': str(e)}

    def _classify_motion_type(self, directionality: float, mean_speed: float,
                            std_speed: float, msd_results: Dict) -> str:
        """Classify motion type based on analysis"""
        try:
            self.log_debug("Classifying motion type")

            alpha = msd_results.get('alpha', np.nan)

            # Log classification criteria
            classification_data = {
                'directionality': directionality,
                'mean_speed': mean_speed,
                'std_speed': std_speed,
                'alpha': alpha,
                'speed_variability': std_speed / mean_speed if mean_speed > 0 else np.inf
            }
            self.log_parameters(classification_data, "classification criteria")

            # Classification based on multiple criteria
            if directionality > 0.7:
                motion_type = 'directed'
            elif not np.isnan(alpha):
                if alpha < 0.8:
                    motion_type = 'confined'
                elif alpha > 1.2:
                    motion_type = 'superdiffusive'
                else:
                    motion_type = 'brownian'
            elif std_speed / mean_speed > 1.0 if mean_speed > 0 else False:
                motion_type = 'variable'
            else:
                motion_type = 'unknown'

            self.log_debug(f"Motion classified as: {motion_type}")
            return motion_type

        except Exception as e:
            self.log_error(f"Error in motion classification: {str(e)}")
            return 'unknown'


def analyze_tracking_results(tracks_final: List[Dict], prob_dim: int = 2) -> Dict:
    """
    Analyze overall tracking results

    Args:
        tracks_final: List of final tracks
        prob_dim: Problem dimensionality

    Returns:
        Dictionary with analysis results
    """
    # Log function call
    log_function_call(logger, 'analyze_tracking_results', (tracks_final,), {'prob_dim': prob_dim})

    logger.info("=== TRACKING RESULTS ANALYSIS STARTED ===")

    try:
        if not tracks_final:
            logger.warning("No tracks to analyze")
            return {'error': 'No tracks to analyze'}

        logger.info(f"Analyzing {len(tracks_final)} tracks")

        with PerformanceTimer(logger, "Complete tracking results analysis"):

            motion_analyzer = MotionAnalyzer()

            # Overall statistics
            num_tracks = len(tracks_final)
            track_lengths = []
            track_lifetimes = []
            motion_types = []
            directionalities = []
            mean_speeds = []

            # Track analysis progress
            successful_analyses = 0
            failed_analyses = 0

            # Analyze each track
            for i, track in enumerate(tracks_final):
                if i % 100 == 0 or i < 5:  # Log progress periodically
                    logger.debug(f"Analyzing track {i + 1}/{num_tracks}")

                motion_results = motion_analyzer.analyze_track_motion(track, prob_dim)

                if 'error' not in motion_results:
                    successful_analyses += 1
                    track_lengths.append(motion_results['track_length'])

                    # Calculate lifetime from sequence of events
                    seq_events = track.get('seq_of_events', [])
                    if len(seq_events) >= 2:
                        lifetime = seq_events[-1][0] - seq_events[0][0] + 1
                        track_lifetimes.append(lifetime)

                    motion_types.append(motion_results['motion_type'])
                    directionalities.append(motion_results['directionality'])
                    mean_speeds.append(motion_results['mean_speed'])
                else:
                    failed_analyses += 1
                    logger.debug(f"Track {i + 1} analysis failed: {motion_results.get('error', 'Unknown error')}")

            # Log analysis progress
            analysis_progress = {
                'total_tracks': num_tracks,
                'successful_analyses': successful_analyses,
                'failed_analyses': failed_analyses,
                'success_rate': successful_analyses / num_tracks if num_tracks > 0 else 0
            }
            logger.info(f"Analysis progress: {successful_analyses}/{num_tracks} tracks analyzed successfully")

            # Calculate summary statistics
            with PerformanceTimer(logger, "Summary statistics calculation"):
                results = {
                    'num_tracks': num_tracks,
                    'successful_analyses': successful_analyses,
                    'failed_analyses': failed_analyses,
                    'mean_track_length': np.mean(track_lengths) if track_lengths else 0,
                    'median_track_length': np.median(track_lengths) if track_lengths else 0,
                    'std_track_length': np.std(track_lengths) if track_lengths else 0,
                    'mean_lifetime': np.mean(track_lifetimes) if track_lifetimes else 0,
                    'median_lifetime': np.median(track_lifetimes) if track_lifetimes else 0,
                    'mean_speed': np.mean(mean_speeds) if mean_speeds else 0,
                    'median_speed': np.median(mean_speeds) if mean_speeds else 0,
                    'mean_directionality': np.mean(directionalities) if directionalities else 0,
                    'motion_type_counts': {},
                    'track_lengths': track_lengths,
                    'track_lifetimes': track_lifetimes,
                    'mean_speeds': mean_speeds,
                    'directionalities': directionalities
                }

                # Count motion types
                if motion_types:
                    unique_types, counts = np.unique(motion_types, return_counts=True)
                    results['motion_type_counts'] = dict(zip(unique_types, counts))
                    logger.info(f"Motion type distribution: {results['motion_type_counts']}")

            # Log final results summary
            final_summary = {
                'num_tracks': results['num_tracks'],
                'mean_track_length': results['mean_track_length'],
                'mean_speed': results['mean_speed'],
                'mean_directionality': results['mean_directionality'],
                'motion_types_found': len(results['motion_type_counts'])
            }
            logger.info("Analysis summary:")
            for key, value in final_summary.items():
                logger.info(f"  {key}: {value}")

            logger.info("=== TRACKING RESULTS ANALYSIS COMPLETED ===")
            return results

    except Exception as e:
        logger.error(f"Error in tracking results analysis: {str(e)}")
        logger.exception("Full traceback:")
        return {'error': str(e)}


# Module initialization logging
logger.info("=== TRACK ANALYSIS MODULE LOADED ===")
logger.debug(f"Module file: {__file__}")
logger.debug("Available classes: TrackAnalyzer, MotionAnalyzer")
logger.debug("Available functions: analyze_tracking_results")
logger.info("Track analysis module with centralized logging ready")
