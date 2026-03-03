# pytrack/utrack_linking.py - FIXED VERSION
"""
U-Track Compatible Linking System
FIXES:
1. Better handling of 2-frame sequences
2. Improved track segment creation
3. More robust DataFrame conversion
4. Enhanced debugging and error handling
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings
from pathlib import Path
import sys
import os

# Add current directory to path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Simple fallback implementations for missing dependencies
class SimpleConfig:
    def __init__(self):
        self.tracking = None

def get_config():
    """Fallback config provider"""
    return SimpleConfig()

def get_logger(name):
    """Fallback logger"""
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)

def log_function_call(log_timing=False):
    """Fallback decorator"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator


try:
    from config import get_config, TrackingConfig
    from logging_setup import get_logger, log_function_call
except ImportError:
    # Use fallback implementations defined above
    pass

@dataclass
class MovieInfo:
    """Information for a single frame"""
    x_coord: np.ndarray      # Shape: (n_particles, 2) - [position, uncertainty]
    y_coord: np.ndarray      # Shape: (n_particles, 2) - [position, uncertainty]
    z_coord: Optional[np.ndarray] = None  # Shape: (n_particles, 2) for 3D
    amp: np.ndarray = None   # Shape: (n_particles, 2) - [amplitude, uncertainty]
    num: int = 0             # Number of particles in frame
    all_coord: np.ndarray = None  # Shape: (n_particles, 2*prob_dim + 2) - concatenated coords
    nn_dist: np.ndarray = None    # Nearest neighbor distances


@dataclass
class KalmanFilterInfo:
    """Kalman filter state information"""
    state_vec: np.ndarray       # Shape: (n_particles, 2*prob_dim) - [pos, vel]
    state_cov: np.ndarray       # Shape: (2*prob_dim, 2*prob_dim, n_particles)
    noise_var: np.ndarray       # Shape: (2*prob_dim, 2*prob_dim, n_particles)
    observation_mat: np.ndarray = None  # Observation matrix H
    track_ages: np.ndarray = None       # Shape: (n_particles,) - frames since track start


@dataclass
class TrackSegment:
    """Individual track segment after frame-to-frame linking"""
    tracks_feat_indx: np.ndarray    # Feature indices (1-based, 0=gap)
    tracks_coord_amp: np.ndarray    # Coordinates and amplitudes
    n_segments: int = 1             # Number of segments
    n_frames: int = 0              # Number of frames
    abs_start_frame: int = 0       # Absolute start frame in the movie


@dataclass
class CompoundTrack:
    """Compound track after gap closing and merge/split detection"""
    tracks_feat_indx_cg: np.ndarray   # Feature connectivity matrix
    tracks_coord_amp_cg: np.ndarray   # Position and amplitude matrix
    seq_of_events: np.ndarray         # Sequence of events matrix

    def __post_init__(self):
        """Validate track structure"""
        if self.tracks_feat_indx_cg.size > 0:
            self.n_frames = len(self.tracks_feat_indx_cg)
        else:
            self.n_frames = 0


@dataclass
class CostMatrixParams:
    """Parameters for cost matrix calculation"""
    # Motion model parameters
    linear_motion: int = 0              # 0=Brownian only, 1=allow directed motion
    min_search_radius: float = 2.0      # Minimum search radius (pixels)
    max_search_radius: float = 10.0     # Maximum search radius (pixels)

    # Brownian motion parameters
    brown_std_mult: float = 3.0         # Search radius multiplier
    brown_scaling: List[float] = None   # [short_time_scaling, long_time_scaling]
    time_reach_conf_b: int = 3          # Time to reach confident Brownian estimate

    # Linear motion parameters
    lin_std_mult: float = 1.0           # Linear motion search multiplier
    lin_scaling: List[float] = None     # [short_time_scaling, long_time_scaling]
    time_reach_conf_l: int = 3          # Time to reach confident linear estimate
    max_angle_vv: float = 30.0          # Max angle between velocities (degrees)

    # Classification and density parameters
    len_for_classify: int = 5           # Minimum length for motion classification
    use_local_density: int = 1          # Use local density for search radius
    nn_window: int = 5                  # Window for nearest neighbor calculation

    # Optional parameters
    gap_penalty: float = 1.5            # Penalty for longer gaps
    alternative_cost_factor: float = 1.05  # Factor for birth/death costs

    def __post_init__(self):
        """Set default values for list parameters"""
        if self.brown_scaling is None:
            self.brown_scaling = [0.5, 0.01]
        if self.lin_scaling is None:
            self.lin_scaling = [1.0, 0.01]


@dataclass
class GapCloseParams:
    """Parameters for gap closing"""
    time_window: int = 5        # Maximum frames to bridge
    merge_split: int = 1        # 0=off, 1=both, 2=merge only, 3=split only
    min_track_len: int = 2      # Minimum track length to keep
    diagnostics: int = 0        # Diagnostic output level


class UTrackLinker:
    """
    FIXED: U-Track compatible linking system implementing the two-step LAP algorithm

    Step 1: Frame-to-frame linking with Kalman filtering
    Step 2: Gap closing, merging and splitting detection
    """

    def __init__(self, config: Optional[TrackingConfig] = None):
        self.config = config or get_config().tracking
        self.logger = get_logger('utrack_linking')

        # Convert config to u-track format
        self.cost_params_link = CostMatrixParams(
            linear_motion=1 if hasattr(self.config, 'motion_model') and self.config.motion_model == 'linear' else 0,
            min_search_radius=2.0,
            max_search_radius=getattr(self.config, 'max_linking_distance', 10.0),
            brown_std_mult=3.0,
            use_local_density=1 if getattr(self.config, 'linking_distance_auto', False) else 0
        )

        self.cost_params_gap = CostMatrixParams(
            linear_motion=self.cost_params_link.linear_motion,
            min_search_radius=2.0,
            max_search_radius=getattr(self.config, 'max_linking_distance', 15.0),
            brown_std_mult=4.0,
            gap_penalty=1.5
        )

        self.gap_close_params = GapCloseParams(
            time_window=getattr(self.config, 'max_gap_frames', 5),
            merge_split=1 if getattr(self.config, 'enable_merging', False) or getattr(self.config, 'enable_splitting', False) else 0,
            min_track_len=getattr(self.config, 'min_track_length', 3)
        )

        self.logger.info("U-Track linker initialized")

    @log_function_call(log_timing=True)
    def track_particles(self, detections: List[pd.DataFrame]) -> pd.DataFrame:
        """
        FIXED: Main tracking function implementing the u-track two-step algorithm
        Now handles sparse detections better

        Args:
            detections: List of detection DataFrames, one per frame

        Returns:
            DataFrame with complete tracks
        """
        if len(detections) < 1:
            return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

        # Count total particles
        total_particles = sum(len(det) for det in detections)
        non_empty_frames = sum(1 for det in detections if len(det) > 0)

        self.logger.info(f"Processing {len(detections)} frames with {total_particles} total particles ({non_empty_frames} non-empty frames)")

        # FIXED: Better handling of sparse data
        if total_particles < 2:
            self.logger.warning("Very few particles detected, returning minimal tracks")
            return self._handle_minimal_detections(detections)

        # FIXED: Better handling of single frame case
        if len(detections) == 1:
            return self._handle_single_frame(detections)

        try:
            self.logger.debug(f"Processing {len(detections)} frames")
            for i, det in enumerate(detections):
                self.logger.debug(f"Frame {i}: {len(det)} particles")

            # Convert detections to u-track format
            movie_info = self._convert_detections_to_movie_info(detections)

            # Step 1: Frame-to-frame linking with Kalman filtering
            # FIXED: Only run if we have multiple frames with particles
            if non_empty_frames >= 2:
                track_segments, kalman_info = self._link_features_kalman_sparse(movie_info, prob_dim=2)
                self.logger.debug(f"Generated {len(track_segments)} track segments")
            else:
                # Fallback for very sparse data
                track_segments = self._create_minimal_segments(movie_info)
                kalman_info = []

            if not track_segments:
                self.logger.warning("No track segments generated")
                return self._handle_minimal_detections(detections)

            # Step 2: Gap closing and merge/split detection
            compound_tracks = self._close_gaps_kalman_sparse(track_segments, kalman_info, prob_dim=2)

            self.logger.debug(f"Generated {len(compound_tracks)} compound tracks")

            # Convert back to standard format
            self.debug_compound_tracks(compound_tracks)
            tracks_df = self._convert_compound_tracks_to_dataframe_fixed(compound_tracks, detections)

            self.logger.info(f"Generated {len(compound_tracks)} compound tracks with {len(tracks_df)} total detections")
            return tracks_df

        except Exception as e:
            self.logger.error(f"Tracking failed: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            # Fallback to minimal tracking
            return self._handle_minimal_detections(detections)

    def _handle_minimal_detections(self, detections: List[pd.DataFrame]) -> pd.DataFrame:
        """Handle cases with very sparse detections by creating individual tracks"""
        all_data = []
        track_id = 1

        for frame_idx, frame_detections in enumerate(detections):
            if len(frame_detections) > 0:
                for _, particle in frame_detections.iterrows():
                    data_row = {
                        'particle_id': track_id,
                        'track_id': track_id,
                        'frame': frame_idx,
                        'x': particle['x'],
                        'y': particle['y'],
                        'intensity': particle.get('intensity', 100.0)
                    }
                    all_data.append(data_row)
                    track_id += 1

        if all_data:
            return pd.DataFrame(all_data)
        else:
            return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

    def _create_minimal_segments(self, movie_info: List[MovieInfo]) -> List[TrackSegment]:
        """Create minimal track segments for sparse data"""
        segments = []

        for frame_idx, frame_info in enumerate(movie_info):
            if frame_info.num > 0:
                for particle_idx in range(frame_info.num):
                    # Create single-frame track segment
                    feat_indx = np.zeros(1)
                    feat_indx[0] = particle_idx + 1  # 1-based indexing

                    coord_amp = np.full(8, np.nan)
                    if particle_idx < len(frame_info.all_coord):
                        coord_amp[0:2] = frame_info.all_coord[particle_idx, 0:2]    # x, dx
                        coord_amp[2:4] = frame_info.all_coord[particle_idx, 2:4]    # y, dy
                        coord_amp[6:8] = frame_info.all_coord[particle_idx, 4:6]    # amp, damp

                    segment = TrackSegment(
                        tracks_feat_indx=feat_indx,
                        tracks_coord_amp=coord_amp,
                        n_segments=1,
                        n_frames=1,
                        abs_start_frame=frame_idx
                    )
                    segments.append(segment)

        return segments

    def _convert_detections_to_movie_info(self, detections: List[pd.DataFrame]) -> List[MovieInfo]:
        """FIXED: Convert detection DataFrames to u-track MovieInfo format"""
        movie_info = []

        for frame_idx, frame_detections in enumerate(detections):
            if len(frame_detections) == 0:
                # Empty frame
                frame_info = MovieInfo(
                    x_coord=np.array([]).reshape(0, 2),
                    y_coord=np.array([]).reshape(0, 2),
                    amp=np.array([]).reshape(0, 2),
                    num=0,
                    all_coord=np.array([]).reshape(0, 6),
                    nn_dist=np.array([])
                )
            else:
                # Extract coordinates with uncertainties
                # Use actual detection uncertainty if available
                n_det = len(frame_detections)
                default_unc = 0.1

                if 'uncertainty [nm]' in frame_detections.columns:
                    x_unc = frame_detections['uncertainty [nm]'].values
                    y_unc = x_unc.copy()
                elif 'sigma [nm]' in frame_detections.columns:
                    x_unc = frame_detections['sigma [nm]'].values
                    y_unc = x_unc.copy()
                elif 'uncertainty_x' in frame_detections.columns:
                    x_unc = frame_detections['uncertainty_x'].values
                    y_unc = frame_detections.get('uncertainty_y', frame_detections['uncertainty_x']).values
                else:
                    x_unc = np.full(n_det, default_unc)
                    y_unc = np.full(n_det, default_unc)

                # Ensure positive uncertainties
                x_unc = np.maximum(x_unc, 1e-6)
                y_unc = np.maximum(y_unc, 1e-6)

                x_coord = np.column_stack([
                    frame_detections['x'].values,
                    x_unc
                ])
                y_coord = np.column_stack([
                    frame_detections['y'].values,
                    y_unc
                ])

                # Handle intensity column
                if 'intensity' in frame_detections.columns:
                    intensities = frame_detections['intensity'].values
                else:
                    intensities = np.full(len(frame_detections), 100.0)  # Default intensity

                amp = np.column_stack([
                    intensities,
                    intensities * 0.1  # 10% uncertainty
                ])

                # FIXED: Concatenate coordinates in the correct u-track format
                # Format: [x, dx, y, dy, amp, damp] - 6 columns total
                all_coord = np.column_stack([x_coord, y_coord, amp])

                self.logger.debug(f"Frame {frame_idx}: all_coord shape = {all_coord.shape}")
                if len(frame_detections) > 0:
                    self.logger.debug(f"Frame {frame_idx}: first particle all_coord = {all_coord[0]}")

                # Calculate nearest neighbor distances
                if len(frame_detections) > 1:
                    positions = frame_detections[['x', 'y']].values
                    dist_mat = cdist(positions, positions)
                    np.fill_diagonal(dist_mat, np.inf)
                    nn_dist = np.min(dist_mat, axis=1)
                else:
                    nn_dist = np.array([1000.0])

                frame_info = MovieInfo(
                    x_coord=x_coord,
                    y_coord=y_coord,
                    amp=amp,
                    num=len(frame_detections),
                    all_coord=all_coord,
                    nn_dist=nn_dist
                )

            movie_info.append(frame_info)

        return movie_info

    def _link_features_kalman_sparse(self, movie_info: List[MovieInfo],
                                   prob_dim: int) -> Tuple[List[TrackSegment], List[KalmanFilterInfo]]:
        """
        FIXED: Step 1: Frame-to-frame linking using Kalman filtering
        """
        n_frames = len(movie_info)
        if n_frames < 2:
            return [], []

        # Initialize storage
        all_assignments = []
        kalman_info_all = []

        # Initialize Kalman filter for first frame
        kalman_info = self._kalman_init_linear_motion(movie_info[0], prob_dim)
        kalman_info_all.append(kalman_info)

        # Link each consecutive frame pair
        for i_frame in range(n_frames - 1):
            frame1 = movie_info[i_frame]
            frame2 = movie_info[i_frame + 1]

            self.logger.debug(f"Linking frame {i_frame} ({frame1.num} particles) to frame {i_frame+1} ({frame2.num} particles)")

            # Update Kalman filter
            kalman_info = self._kalman_gain_linear_motion(kalman_info_all[i_frame], frame2, prob_dim)
            kalman_info_all.append(kalman_info)

            # Create cost matrix
            cost_matrix, nonlink_marker = self._cost_mat_linking(frame1, frame2, kalman_info, prob_dim)

            # Solve LAP
            if cost_matrix.size > 0:
                assignments = self._solve_lap(cost_matrix, nonlink_marker)
                all_assignments.append(assignments)
                self.logger.debug(f"Frame {i_frame}->{i_frame+1}: {len(assignments)} assignments")
            else:
                all_assignments.append([])
                self.logger.debug(f"Frame {i_frame}->{i_frame+1}: No valid assignments")

        # Convert assignments to track segments
        track_segments = self._assignments_to_tracks_fixed(all_assignments, movie_info, prob_dim)

        return track_segments, kalman_info_all

    def _assignments_to_tracks_fixed(self, all_assignments: List[np.ndarray],
                                   movie_info: List[MovieInfo], prob_dim: int) -> List[TrackSegment]:
        """Build multi-frame track segments by chaining frame-to-frame LAP assignments.

        Instead of creating separate 2-frame segments per frame pair (the old bug),
        this traces connected particle chains across all frames to form proper
        multi-frame tracks, matching the original U-Track 2.5 behaviour.
        """
        n_frames = len(movie_info)

        if n_frames < 2 or not all_assignments:
            return []

        self.logger.debug(f"Chaining assignments across {n_frames} frames")

        # ------------------------------------------------------------------
        # 1. Build forward-link graph: links[(frame, particle)] = (frame+1, particle)
        # ------------------------------------------------------------------
        forward_links = {}   # (frame_idx, particle_1based) -> (frame_idx+1, particle_1based)
        linked_as_target = set()  # particles that are linked TO (not track starts)

        for frame_idx, assignments in enumerate(all_assignments):
            if len(assignments) == 0:
                # Fallback: spatial proximity matching
                spatial_links = self._create_spatial_links(
                    movie_info[frame_idx], movie_info[frame_idx + 1])
                for src, tgt in spatial_links.items():
                    forward_links[(frame_idx, src)] = (frame_idx + 1, tgt)
                    linked_as_target.add((frame_idx + 1, tgt))
                continue

            n_source = movie_info[frame_idx].num
            n_target = movie_info[frame_idx + 1].num

            particle_links = {}
            for source_particle in range(1, n_source + 1):
                if source_particle < len(assignments):
                    target_assignment = assignments[source_particle]
                    if 1 <= target_assignment <= n_target:
                        particle_links[source_particle] = target_assignment

            # Fallback to spatial if LAP found nothing
            if len(particle_links) == 0:
                particle_links = self._create_spatial_links(
                    movie_info[frame_idx], movie_info[frame_idx + 1])

            for src, tgt in particle_links.items():
                forward_links[(frame_idx, src)] = (frame_idx + 1, tgt)
                linked_as_target.add((frame_idx + 1, tgt))

        # ------------------------------------------------------------------
        # 2. Find chain starts: any (frame, particle) that is NOT a link target
        # ------------------------------------------------------------------
        all_particles = set()
        for frame_idx in range(n_frames):
            for p in range(1, movie_info[frame_idx].num + 1):
                all_particles.add((frame_idx, p))

        chain_starts = all_particles - linked_as_target

        # ------------------------------------------------------------------
        # 3. Trace each chain to build multi-frame track segments
        # ------------------------------------------------------------------
        completed_tracks = []
        visited = set()

        for start in sorted(chain_starts):
            frames_list = []
            particles_list = []
            current = start

            while current is not None and current not in visited:
                visited.add(current)
                frames_list.append(current[0])
                particles_list.append(current[1])
                current = forward_links.get(current, None)

            if len(frames_list) > 0:
                track_data = {'frames': frames_list, 'particles': particles_list}
                segment = self._build_track_segment_from_data(track_data, movie_info, prob_dim)
                if segment is not None:
                    completed_tracks.append(segment)

        self.logger.debug(f"Created {len(completed_tracks)} chained track segments")
        return completed_tracks

    def _create_spatial_links(self, frame1_info: MovieInfo, frame2_info: MovieInfo) -> Dict[int, int]:
        """Create links based on spatial proximity when LAP fails"""
        if frame1_info.num == 0 or frame2_info.num == 0:
            return {}

        # Get positions
        pos1 = frame1_info.all_coord[:, [0, 2]]  # x, y positions
        pos2 = frame2_info.all_coord[:, [0, 2]]

        # Calculate distances
        from scipy.spatial.distance import cdist
        distances = cdist(pos1, pos2)

        # Create links for close particles
        max_distance = 5.0  # pixels
        links = {}
        used_targets = set()

        # Sort by distance and assign greedily
        assignments = []
        for i in range(len(pos1)):
            for j in range(len(pos2)):
                if distances[i, j] <= max_distance:
                    assignments.append((distances[i, j], i + 1, j + 1))  # Convert to 1-indexed

        # Sort by distance (closest first)
        assignments.sort()

        # Assign greedily
        for dist, source_p, target_p in assignments:
            if source_p not in links and target_p not in used_targets:
                links[source_p] = target_p
                used_targets.add(target_p)
                self.logger.debug(f"Spatial link: {source_p} -> {target_p} (dist: {dist:.3f})")

        return links

    def _build_track_segment_from_data(self, track_data: Dict, movie_info: List[MovieInfo],
                                         prob_dim: int) -> Optional[TrackSegment]:
        """FIXED: Build a TrackSegment from track data with correct coordinate handling"""
        frames = track_data['frames']
        particles = track_data['particles']

        if len(frames) == 0:
            return None

        start_frame = min(frames)
        end_frame = max(frames)
        n_frames_total = end_frame - start_frame + 1

        # Initialize arrays
        feat_indx = np.zeros(n_frames_total)
        coord_amp = np.full(n_frames_total * 8, np.nan)

        self.logger.debug(f"Building track segment: frames {start_frame}-{end_frame}, {len(frames)} data points")

        # Fill in track data
        for i, (frame, particle) in enumerate(zip(frames, particles)):
            relative_frame = frame - start_frame
            feat_indx[relative_frame] = particle

            # Get coordinates and amplitude from the original movie_info
            if frame < len(movie_info) and particle <= movie_info[frame].num:
                frame_info = movie_info[frame]
                particle_idx = particle - 1  # Convert to 0-indexed

                base_idx = relative_frame * 8

                # FIXED: Proper coordinate extraction
                if particle_idx < len(frame_info.all_coord):
                    # all_coord format: [x, dx, y, dy, amp, damp] (6 columns)
                    original_coords = frame_info.all_coord[particle_idx]

                    self.logger.debug(f"Frame {frame}, particle {particle}: original_coords = {original_coords}")

                    if len(original_coords) >= 6:
                        # Extract coordinates properly
                        coord_amp[base_idx + 0] = original_coords[0]     # x position
                        coord_amp[base_idx + 1] = original_coords[2]     # y position
                        coord_amp[base_idx + 2] = 0.0                    # z position (2D data)
                        coord_amp[base_idx + 3] = original_coords[4]     # amplitude
                        coord_amp[base_idx + 4] = original_coords[1]     # dx uncertainty
                        coord_amp[base_idx + 5] = original_coords[3]     # dy uncertainty
                        coord_amp[base_idx + 6] = 0.1                    # dz uncertainty
                        coord_amp[base_idx + 7] = original_coords[5]     # damp uncertainty

                        self.logger.debug(f"Stored coordinates: x={coord_amp[base_idx + 0]}, y={coord_amp[base_idx + 1]}, amp={coord_amp[base_idx + 3]}")
                    else:
                        self.logger.warning(f"Insufficient coordinate data for frame {frame}, particle {particle}")

        return TrackSegment(
            tracks_feat_indx=feat_indx,
            tracks_coord_amp=coord_amp,
            n_segments=1,
            n_frames=n_frames_total,
            abs_start_frame=start_frame
        )

    def _close_gaps_kalman_sparse(self, track_segments: List[TrackSegment],
                                kalman_info: List[KalmanFilterInfo],
                                prob_dim: int) -> List[CompoundTrack]:
        """Step 2: Gap closing, merge, and split detection via a second LAP.

        Implements the full U-Track 2.5 Step 2 algorithm:
        - Gap closing: connect segment end → segment start across gaps
        - Merge: segment end merges into the middle of another segment
        - Split: segment start splits from the middle of another segment
        The cost matrix is augmented with adaptive birth/death alternatives.
        """
        # Filter short tracks
        min_track_len = max(1, self.gap_close_params.min_track_len - 1)
        long_segments = []

        for segment in track_segments:
            valid_count = np.sum(segment.tracks_feat_indx > 0)
            if valid_count >= min_track_len:
                long_segments.append(segment)

        self.logger.debug(f"Kept {len(long_segments)} segments after length filtering "
                          f"(min_track_len={min_track_len})")

        if not long_segments:
            return []

        time_window = self.gap_close_params.time_window
        gap_penalty = self.cost_params_gap.gap_penalty
        max_gap_radius = self.cost_params_gap.max_search_radius

        # ------------------------------------------------------------------
        # 1. Extract segment end / start / interior information
        # ------------------------------------------------------------------
        n_seg = len(long_segments)
        seg_end_frame = np.zeros(n_seg, dtype=int)
        seg_end_pos = np.zeros((n_seg, prob_dim))
        seg_start_frame = np.zeros(n_seg, dtype=int)
        seg_start_pos = np.zeros((n_seg, prob_dim))

        # For merge/split: per-segment dict of {abs_frame: position}
        seg_interior = [{}] * n_seg  # abs_frame -> (x, y) for interior points

        for idx, seg in enumerate(long_segments):
            valid_frames = np.where(seg.tracks_feat_indx > 0)[0]
            if len(valid_frames) == 0:
                continue

            abs_offset = seg.abs_start_frame

            # Start info (absolute frame)
            sf = valid_frames[0]
            seg_start_frame[idx] = abs_offset + sf
            base = sf * 8
            seg_start_pos[idx, 0] = seg.tracks_coord_amp[base]
            seg_start_pos[idx, 1] = seg.tracks_coord_amp[base + 1]

            # End info (absolute frame)
            ef = valid_frames[-1]
            seg_end_frame[idx] = abs_offset + ef
            base = ef * 8
            seg_end_pos[idx, 0] = seg.tracks_coord_amp[base]
            seg_end_pos[idx, 1] = seg.tracks_coord_amp[base + 1]

            # Interior points (all valid frames except first and last)
            interior = {}
            for vf in valid_frames:
                abs_f = abs_offset + vf
                b = vf * 8
                interior[abs_f] = (seg.tracks_coord_amp[b], seg.tracks_coord_amp[b + 1])
            seg_interior[idx] = interior

        # ------------------------------------------------------------------
        # 2. Determine merge/split mode
        # ------------------------------------------------------------------
        ms_mode = self.gap_close_params.merge_split
        do_merge = ms_mode in (1, 2)
        do_split = ms_mode in (1, 3)

        # ------------------------------------------------------------------
        # 3. Build merge candidates: seg_end -> interior of another segment
        #    A merge means segment i ends, and its last position is close to
        #    some interior frame of segment j (the particle merged into j).
        # ------------------------------------------------------------------
        merge_candidates = []  # (end_seg_i, target_seg_j, target_abs_frame, cost)
        if do_merge:
            for i in range(n_seg):
                for j in range(n_seg):
                    if i == j:
                        continue
                    end_f = seg_end_frame[i]
                    # The merge must happen at or just after the end frame
                    for abs_f, pos_j in seg_interior[j].items():
                        dt = abs_f - end_f
                        if dt < 0 or dt > 1:
                            continue  # merge must be at frame end or end+1
                        d2 = (seg_end_pos[i, 0] - pos_j[0]) ** 2 + (seg_end_pos[i, 1] - pos_j[1]) ** 2
                        if np.sqrt(d2) > max_gap_radius:
                            continue
                        merge_candidates.append((i, j, abs_f, d2))

        n_merge = len(merge_candidates)

        # ------------------------------------------------------------------
        # 4. Build split candidates: interior of a segment -> seg_start
        #    A split means segment j starts, and its first position is close
        #    to some interior frame of segment i (the particle split from i).
        # ------------------------------------------------------------------
        split_candidates = []  # (source_seg_i, start_seg_j, source_abs_frame, cost)
        if do_split:
            for j in range(n_seg):
                for i in range(n_seg):
                    if i == j:
                        continue
                    start_f = seg_start_frame[j]
                    for abs_f, pos_i in seg_interior[i].items():
                        dt = start_f - abs_f
                        if dt < 0 or dt > 1:
                            continue
                        d2 = (pos_i[0] - seg_start_pos[j, 0]) ** 2 + (pos_i[1] - seg_start_pos[j, 1]) ** 2
                        if np.sqrt(d2) > max_gap_radius:
                            continue
                        split_candidates.append((i, j, abs_f, d2))

        n_split = len(split_candidates)

        self.logger.debug(f"Gap close candidates: {n_seg} segments, "
                          f"{n_merge} merge candidates, {n_split} split candidates")

        # ------------------------------------------------------------------
        # 5. Build gap-closing cost sub-matrix (n_seg x n_seg)
        # ------------------------------------------------------------------
        gc_cost = np.full((n_seg, n_seg), np.inf)

        for i in range(n_seg):
            for j in range(n_seg):
                if i == j:
                    continue
                dt = seg_start_frame[j] - seg_end_frame[i]
                if dt < 1 or dt > time_window:
                    continue
                d2 = np.sum((seg_end_pos[i] - seg_start_pos[j]) ** 2)
                if np.sqrt(d2) > max_gap_radius * np.sqrt(dt):
                    continue
                gc_cost[i, j] = d2 * (gap_penalty ** (dt - 1))

        # ------------------------------------------------------------------
        # 6. Build augmented cost matrix
        #    Layout (U-Track 2.5 style with merge/split):
        #
        #    Rows:    [seg_ends (n_seg)] [split_sources (n_split)]
        #    Cols:    [seg_starts (n_seg)] [merge_targets (n_merge)]
        #    + birth/death alternative blocks
        #
        #    Total rows = n_seg + n_split + n_seg + n_merge  (with alternatives)
        #    Total cols = n_seg + n_merge + n_seg + n_split  (with alternatives)
        #
        #    Simplified: we use the standard augmented LAP format:
        #      n_rows_real = n_seg + n_split
        #      n_cols_real = n_seg + n_merge
        #      matrix_size = n_rows_real + n_cols_real
        # ------------------------------------------------------------------
        n_rows = n_seg + n_split  # end segments + split sources
        n_cols = n_seg + n_merge  # start segments + merge targets

        matrix_size = n_rows + n_cols
        gc_full = np.full((matrix_size, matrix_size), np.inf)

        # (a) Upper-left: linking costs
        # Gap closing: rows 0..n_seg-1 (ends) × cols 0..n_seg-1 (starts)
        gc_full[:n_seg, :n_seg] = gc_cost

        # Merge costs: rows 0..n_seg-1 (ends) × cols n_seg..n_seg+n_merge-1
        for m_idx, (end_i, _, _, cost) in enumerate(merge_candidates):
            gc_full[end_i, n_seg + m_idx] = cost

        # Split costs: rows n_seg..n_seg+n_split-1 × cols 0..n_seg-1 (starts)
        for s_idx, (_, start_j, _, cost) in enumerate(split_candidates):
            gc_full[n_seg + s_idx, start_j] = cost

        # Collect all valid costs for adaptive thresholds
        all_valid = gc_full[:n_rows, :n_cols]
        valid_costs = all_valid[np.isfinite(all_valid)]

        if len(valid_costs) > 0:
            alt_cost_factor = self.cost_params_gap.alternative_cost_factor
            alt_cost = np.percentile(valid_costs, 90) * alt_cost_factor
            alt_cost = max(alt_cost, self.cost_params_gap.min_search_radius ** 2)
        else:
            alt_cost = max_gap_radius ** 2

        # (b) Upper-right: death alternative (n_rows × n_rows diagonal)
        for r in range(n_rows):
            gc_full[r, n_cols + r] = alt_cost

        # (c) Lower-left: birth alternative (n_cols × n_cols diagonal)
        for c in range(n_cols):
            gc_full[n_rows + c, c] = alt_cost

        # (d) Lower-right: dummy costs
        dummy_cost = np.min(valid_costs) if len(valid_costs) > 0 else alt_cost
        gc_full[n_rows:, n_cols:] = dummy_cost

        # ------------------------------------------------------------------
        # 7. Solve the gap-closing LAP
        # ------------------------------------------------------------------
        gc_assignments = self._solve_lap(gc_full, nonlink_marker=-1)

        # Parse assignments
        merge_map = {}       # end_seg_idx -> start_seg_idx (gap closing)
        merge_events = []    # (end_seg, target_seg, abs_frame)
        split_events = []    # (source_seg, start_seg, abs_frame)

        for row_1idx in range(1, n_rows + 1):
            if row_1idx >= len(gc_assignments):
                continue
            col_1idx = gc_assignments[row_1idx]
            if col_1idx < 1:
                continue

            row = row_1idx - 1  # 0-indexed
            col = col_1idx - 1  # 0-indexed

            if col >= n_cols:
                continue  # assigned to death alternative

            if row < n_seg and col < n_seg:
                # Gap closing: segment row's end → segment col's start
                if row != col:
                    merge_map[row] = col
            elif row < n_seg and col >= n_seg:
                # Merge: segment row's end merges into another segment
                m_idx = col - n_seg
                if m_idx < n_merge:
                    end_i, target_j, abs_f, _ = merge_candidates[m_idx]
                    merge_events.append((end_i, target_j, abs_f))
            elif row >= n_seg and col < n_seg:
                # Split: a segment splits to become segment col
                s_idx = row - n_seg
                if s_idx < n_split:
                    source_i, start_j, abs_f, _ = split_candidates[s_idx]
                    split_events.append((source_i, start_j, abs_f))

        n_gc = len(merge_map)
        self.logger.debug(f"Gap closing: {n_gc} closures, "
                          f"{len(merge_events)} merges, {len(split_events)} splits")

        # ------------------------------------------------------------------
        # 8. Chain merged segments into compound tracks
        # ------------------------------------------------------------------
        merged_into = set(merge_map.values())
        visited_seg = set()
        compound_tracks = []

        for seed in range(n_seg):
            if seed in visited_seg or seed in merged_into:
                continue

            chain = [seed]
            visited_seg.add(seed)
            current = seed
            while current in merge_map:
                nxt = merge_map[current]
                if nxt in visited_seg:
                    break
                chain.append(nxt)
                visited_seg.add(nxt)
                current = nxt

            ct = self._merge_segments_to_compound_track(chain, long_segments)
            if ct is not None:
                # Record merge/split events in seq_of_events
                ct = self._add_merge_split_events(
                    ct, chain, long_segments, merge_events, split_events)
                compound_tracks.append(ct)

        # Emit remaining unvisited segments
        for idx in range(n_seg):
            if idx not in visited_seg:
                ct = self._merge_segments_to_compound_track([idx], long_segments)
                if ct is not None:
                    compound_tracks.append(ct)

        self.logger.debug(f"Gap closing produced {len(compound_tracks)} compound tracks "
                          f"from {n_seg} segments ({n_gc} gap closures)")
        return compound_tracks

    def _merge_segments_to_compound_track(self, seg_indices: List[int],
                                          segments: List[TrackSegment]) -> Optional[CompoundTrack]:
        """Merge a chain of gap-closed segments into a single CompoundTrack."""
        if not seg_indices:
            return None

        # Collect valid segments with their absolute start frames
        seg_info = []  # (abs_start, segment)
        for idx in seg_indices:
            seg = segments[idx]
            if np.sum(seg.tracks_feat_indx > 0) == 0:
                continue
            seg_info.append((seg.abs_start_frame, seg))

        if not seg_info:
            return None

        # Compute global frame range
        global_start = min(s for s, _ in seg_info)
        global_end = max(s + seg.n_frames - 1 for s, seg in seg_info)
        total_frames = global_end - global_start + 1

        combined_feat = np.zeros(total_frames)
        combined_coord = np.full(total_frames * 8, np.nan)

        for abs_start, seg in seg_info:
            offset = abs_start - global_start
            for f in range(seg.n_frames):
                dest_f = offset + f
                if 0 <= dest_f < total_frames and seg.tracks_feat_indx[f] > 0:
                    combined_feat[dest_f] = seg.tracks_feat_indx[f]
                    src_base = f * 8
                    dst_base = dest_f * 8
                    combined_coord[dst_base:dst_base + 8] = seg.tracks_coord_amp[src_base:src_base + 8]

        valid = np.where(combined_feat > 0)[0]
        if len(valid) == 0:
            return None

        start_1idx = valid[0] + global_start + 1
        end_1idx = valid[-1] + global_start + 1

        seq_of_events = np.array([
            [start_1idx, 1, len(seg_info), np.nan],
            [end_1idx, 2, len(seg_info), np.nan]
        ])

        return CompoundTrack(
            tracks_feat_indx_cg=combined_feat,
            tracks_coord_amp_cg=combined_coord,
            seq_of_events=seq_of_events
        )

    def _add_merge_split_events(self, compound_track: CompoundTrack,
                                 chain: List[int],
                                 segments: List[TrackSegment],
                                 merge_events: List[Tuple],
                                 split_events: List[Tuple]) -> CompoundTrack:
        """Add merge/split event rows to a compound track's seq_of_events.

        U-Track seq_of_events format per row: [frame(1-idx), event_type, segment_idx, other_track_idx]
        Event types: 1=start, 2=end, 3=merge, 4=split
        """
        soe_rows = list(compound_track.seq_of_events)
        chain_set = set(chain)

        for end_seg, target_seg, abs_frame in merge_events:
            if end_seg in chain_set:
                # This track's segment merged into target_seg
                soe_rows.append([abs_frame + 1, 3, 1, target_seg + 1])

        for source_seg, start_seg, abs_frame in split_events:
            if start_seg in chain_set:
                # This track's segment split from source_seg
                soe_rows.append([abs_frame + 1, 4, 1, source_seg + 1])

        # Sort events by frame
        soe_rows.sort(key=lambda r: r[0])

        compound_track.seq_of_events = np.array(soe_rows)
        return compound_track

    def _convert_compound_tracks_to_dataframe_fixed(self, compound_tracks: List[CompoundTrack],
                                                  original_detections: List[pd.DataFrame]) -> pd.DataFrame:
        """Convert compound tracks back to standard DataFrame format."""
        if not compound_tracks:
            return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

        data = []

        for track_id, track in enumerate(compound_tracks):
            try:
                if not hasattr(track, 'tracks_feat_indx_cg') or len(track.tracks_feat_indx_cg) == 0:
                    continue

                n_frames = len(track.tracks_feat_indx_cg)
                coords_1d = track.tracks_coord_amp_cg

                if len(coords_1d) < n_frames * 8:
                    self.logger.debug(f"Track {track_id}: insufficient coordinate data")
                    continue

                # Determine the absolute start frame from seq_of_events
                start_frame_abs = 0
                if hasattr(track, 'seq_of_events') and track.seq_of_events is not None:
                    soe = track.seq_of_events
                    if len(soe) > 0:
                        start_frame_abs = int(soe[0, 0]) - 1  # Convert 1-indexed to 0-indexed

                for frame_idx in range(n_frames):
                    particle_id = track.tracks_feat_indx_cg[frame_idx]

                    if particle_id > 0:
                        base_idx = frame_idx * 8

                        if base_idx + 7 < len(coords_1d):
                            raw_coords = coords_1d[base_idx:base_idx + 8]

                            x_coord = raw_coords[0] if not np.isnan(raw_coords[0]) else 0.0
                            y_coord = raw_coords[1] if not np.isnan(raw_coords[1]) else 0.0
                            intensity = raw_coords[3] if not np.isnan(raw_coords[3]) else 100.0

                            data.append({
                                'particle_id': int(particle_id),
                                'track_id': track_id + 1,
                                'frame': start_frame_abs + frame_idx,
                                'x': x_coord,
                                'y': y_coord,
                                'intensity': intensity
                            })

            except Exception as e:
                self.logger.warning(f"Error processing track {track_id}: {e}")
                continue

        if data:
            result_df = pd.DataFrame(data)
            self.logger.info(f"Converted {len(compound_tracks)} compound tracks to {len(result_df)} track points")
            return result_df
        else:
            self.logger.warning("No valid track data could be extracted")
            return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])



    def debug_compound_tracks(self, compound_tracks: List[CompoundTrack]) -> None:
        """Debug function to inspect compound tracks structure"""
        self.logger.info(f"=== DEBUGGING {len(compound_tracks)} COMPOUND TRACKS ===")

        for i, track in enumerate(compound_tracks[:5]):  # Show first 5 tracks
            self.logger.info(f"Track {i}:")

            # Check tracks_feat_indx_cg
            if hasattr(track, 'tracks_feat_indx_cg'):
                feat_indx = track.tracks_feat_indx_cg
                self.logger.info(f"  feat_indx: shape={feat_indx.shape if hasattr(feat_indx, 'shape') else len(feat_indx)}, "
                               f"type={type(feat_indx)}")
                if len(feat_indx) > 0:
                    non_zero = np.sum(feat_indx > 0)
                    self.logger.info(f"  feat_indx: {non_zero}/{len(feat_indx)} non-zero values")
                    self.logger.info(f"  feat_indx sample: {feat_indx[:min(10, len(feat_indx))]}")

            # Check tracks_coord_amp_cg
            if hasattr(track, 'tracks_coord_amp_cg'):
                coord_amp = track.tracks_coord_amp_cg
                self.logger.info(f"  coord_amp: shape={coord_amp.shape if hasattr(coord_amp, 'shape') else len(coord_amp)}, "
                               f"type={type(coord_amp)}")
                if len(coord_amp) > 0:
                    non_nan = np.sum(~np.isnan(coord_amp))
                    self.logger.info(f"  coord_amp: {non_nan}/{len(coord_amp)} non-NaN values")
                    self.logger.info(f"  coord_amp sample: {coord_amp[:min(16, len(coord_amp))]}")

                    # Show structure interpretation
                    n_timepoints = len(coord_amp) // 8
                    self.logger.info(f"  Interpreted as {n_timepoints} time points")
                    if n_timepoints > 0:
                        self.logger.info(f"  First timepoint: x={coord_amp[0]:.3f}, y={coord_amp[1]:.3f}, "
                                       f"z={coord_amp[2]:.3f}, amp={coord_amp[3]:.3f}")

            # Check seq_of_events
            if hasattr(track, 'seq_of_events'):
                events = track.seq_of_events
                self.logger.info(f"  seq_of_events: shape={events.shape if hasattr(events, 'shape') else len(events)}")
                if hasattr(events, '__len__') and len(events) > 0:
                    self.logger.info(f"  seq_of_events: {events}")

            self.logger.info("")

    # Keep all other methods unchanged (Kalman filter, cost matrix, etc.)
    def _kalman_init_linear_motion(self, frame_info: MovieInfo, prob_dim: int) -> KalmanFilterInfo:
        """Initialize Kalman filter for linear motion model"""
        num_features = frame_info.num

        if num_features == 0:
            return KalmanFilterInfo(
                state_vec=np.array([]).reshape(0, 2*prob_dim),
                state_cov=np.zeros((2*prob_dim, 2*prob_dim, 0)),
                noise_var=np.zeros((2*prob_dim, 2*prob_dim, 0))
            )

        # Calculate initial noise variance
        max_search_radius = self.cost_params_link.max_search_radius
        brown_std_mult = self.cost_params_link.brown_std_mult
        noise_var_init = (max_search_radius / brown_std_mult) ** 2 / prob_dim

        # Extract positions
        positions = frame_info.all_coord[:, [0, 2]]  # x, y positions

        # Initialize state vector [position, velocity]
        state_vec = np.zeros((num_features, 2 * prob_dim))
        state_vec[:, :prob_dim] = positions  # Positions
        # Velocities initialized to zero

        # Initialize state covariance matrices
        state_cov = np.zeros((2 * prob_dim, 2 * prob_dim, num_features))
        noise_var = np.zeros((2 * prob_dim, 2 * prob_dim, num_features))

        for i in range(num_features):
            # Position uncertainties from actual detection data
            if i < len(frame_info.x_coord) and frame_info.x_coord.shape[1] > 1:
                pos_var = np.array([
                    max(frame_info.x_coord[i, 1] ** 2, 1e-6),
                    max(frame_info.y_coord[i, 1] ** 2, 1e-6)
                ])
            else:
                pos_var = np.array([0.01, 0.01])  # Default variance
            vel_var = np.full(prob_dim, 4.0)  # Large initial velocity uncertainty

            # Create diagonal covariance matrix
            diag_vals = np.concatenate([pos_var, vel_var])
            state_cov[:, :, i] = np.diag(diag_vals)

            # Noise covariance (process noise)
            noise_var[:, :, i] = np.diag(np.full(2 * prob_dim, noise_var_init))

        # Create observation matrix H = [I 0] to observe positions only
        observation_mat = np.zeros((prob_dim, 2 * prob_dim))
        observation_mat[:prob_dim, :prob_dim] = np.eye(prob_dim)

        return KalmanFilterInfo(
            state_vec=state_vec,
            state_cov=state_cov,
            noise_var=noise_var,
            observation_mat=observation_mat,
            track_ages=np.ones(num_features, dtype=int)  # All start at age 1
        )

    def _kalman_gain_linear_motion(self, kalman_info_prev: KalmanFilterInfo,
                                 frame_info: MovieInfo, prob_dim: int) -> KalmanFilterInfo:
        """Calculate Kalman gain and update state estimates"""
        num_features = frame_info.num

        if num_features == 0:
            # Return empty but properly shaped arrays
            return KalmanFilterInfo(
                state_vec=np.array([]).reshape(0, 2*prob_dim),
                state_cov=np.zeros((2*prob_dim, 2*prob_dim, 0)),
                noise_var=np.zeros((2*prob_dim, 2*prob_dim, 0)),
                observation_mat=np.zeros((prob_dim, 2*prob_dim))
            )

        # State transition matrix A = [I dt*I; 0 I] for constant velocity
        dt = 1.0  # Frame interval
        A = np.eye(2 * prob_dim)
        A[:prob_dim, prob_dim:] = dt * np.eye(prob_dim)

        # Observation matrix H = [I 0]
        H = np.zeros((prob_dim, 2 * prob_dim))
        H[:prob_dim, :prob_dim] = np.eye(prob_dim)

        # Predict and update for all features
        state_vec = np.zeros((num_features, 2 * prob_dim))
        state_cov = np.zeros((2 * prob_dim, 2 * prob_dim, num_features))
        noise_var = np.zeros((2 * prob_dim, 2 * prob_dim, num_features))

        # Handle case where previous frame had fewer features
        num_prev = kalman_info_prev.state_vec.shape[0] if kalman_info_prev.state_vec.size > 0 else 0

        for i in range(num_features):
            if i < num_prev and kalman_info_prev.state_vec.size > 0:
                # Predict from previous state
                try:
                    state_pred = A @ kalman_info_prev.state_vec[i]
                    cov_pred = A @ kalman_info_prev.state_cov[:, :, i] @ A.T + kalman_info_prev.noise_var[:, :, i]
                except (IndexError, ValueError):
                    # Fallback initialization
                    positions = frame_info.all_coord[i, [0, 2]] if i < len(frame_info.all_coord) else np.zeros(2)
                    state_pred = np.zeros(2 * prob_dim)
                    state_pred[:prob_dim] = positions
                    cov_pred = np.diag(np.concatenate([np.array([0.1, 0.1]), np.array([4.0, 4.0])]))
            else:
                # Initialize new feature
                positions = frame_info.all_coord[i, [0, 2]] if i < len(frame_info.all_coord) else np.zeros(2)
                state_pred = np.zeros(2 * prob_dim)
                state_pred[:prob_dim] = positions
                cov_pred = np.diag(np.concatenate([np.array([0.1, 0.1]), np.array([4.0, 4.0])]))

            # Update with observation
            z = frame_info.all_coord[i, [0, 2]] if i < len(frame_info.all_coord) else np.zeros(2)

            # Use actual detection uncertainty for observation noise R
            if i < len(frame_info.x_coord) and frame_info.x_coord.shape[1] > 1:
                r_x = max(frame_info.x_coord[i, 1] ** 2, 1e-6)
                r_y = max(frame_info.y_coord[i, 1] ** 2, 1e-6)
            else:
                r_x, r_y = 0.01, 0.01
            R = np.diag([r_x, r_y])

            # Innovation
            y = z - H @ state_pred
            S = H @ cov_pred @ H.T + R

            # Kalman gain
            try:
                K = cov_pred @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                try:
                    K = cov_pred @ H.T @ np.linalg.pinv(S)
                except:
                    K = np.zeros((2 * prob_dim, prob_dim))

            # Update state and covariance
            state_vec[i] = state_pred + K @ y
            state_cov[:, :, i] = (np.eye(2 * prob_dim) - K @ H) @ cov_pred

            # Process noise (adaptive)
            innovation_norm = np.linalg.norm(y)
            noise_scale = max(innovation_norm / np.sqrt(prob_dim), 0.1)
            noise_var[:, :, i] = np.diag(np.full(2 * prob_dim, noise_scale ** 2))

        # Track ages: increment for existing tracks, start at 1 for new ones
        track_ages = np.ones(num_features, dtype=int)
        prev_ages = kalman_info_prev.track_ages
        if prev_ages is not None:
            for i in range(min(num_prev, num_features)):
                track_ages[i] = prev_ages[i] + 1 if i < len(prev_ages) else 1

        return KalmanFilterInfo(
            state_vec=state_vec,
            state_cov=state_cov,
            noise_var=noise_var,
            observation_mat=H,
            track_ages=track_ages
        )

    def _cost_mat_linking(self, frame1_info: MovieInfo, frame2_info: MovieInfo,
                        kalman_info: KalmanFilterInfo, prob_dim: int) -> Tuple[np.ndarray, float]:
        """Create cost matrix for frame-to-frame linking"""
        n1 = frame1_info.num
        n2 = frame2_info.num

        # FIXED: Better handling of empty frames
        if n1 == 0 and n2 == 0:
            # Both frames empty - return minimal matrix
            return np.array([[0.0]]), -1
        elif n1 == 0:
            # Only first frame empty - create birth-only matrix
            cost_mat = np.full((n2, n2), 100.0)
            np.fill_diagonal(cost_mat, 50.0)
            return cost_mat, -1
        elif n2 == 0:
            # Only second frame empty - create death-only matrix
            cost_mat = np.full((n1, n1), 100.0)
            np.fill_diagonal(cost_mat, 50.0)
            return cost_mat, -1

        # Both frames have particles - proceed with normal linking
        try:
            # Get positions
            pos1 = frame1_info.all_coord[:, [0, 2]]  # x, y positions
            pos2 = frame2_info.all_coord[:, [0, 2]]

            # Use Mahalanobis distance when Kalman covariance is available
            use_mahalanobis = (
                kalman_info is not None
                and kalman_info.state_cov.ndim == 3
                and kalman_info.state_cov.shape[2] >= n1
                and kalman_info.observation_mat is not None
            )

            if use_mahalanobis:
                H = kalman_info.observation_mat  # (prob_dim, 2*prob_dim)
                base_cost_mat = np.full((n1, n2), np.inf)

                for i in range(n1):
                    # Innovation covariance: S = H @ P @ H^T + R
                    P_i = kalman_info.state_cov[:, :, i]
                    # Observation noise from detection uncertainties
                    if i < len(frame1_info.x_coord) and frame1_info.x_coord.shape[1] > 1:
                        r_x = max(frame1_info.x_coord[i, 1] ** 2, 1e-6)
                        r_y = max(frame1_info.y_coord[i, 1] ** 2, 1e-6)
                    else:
                        r_x, r_y = 0.01, 0.01
                    R_i = np.diag([r_x, r_y])

                    S_i = H @ P_i @ H.T + R_i  # (prob_dim, prob_dim)
                    try:
                        S_inv = np.linalg.inv(S_i)
                    except np.linalg.LinAlgError:
                        S_inv = np.linalg.pinv(S_i)

                    # Predicted position from Kalman state
                    pred_pos_i = kalman_info.state_vec[i, :prob_dim] if i < len(kalman_info.state_vec) else pos1[i]

                    for j in range(n2):
                        dx = pos2[j] - pred_pos_i
                        base_cost_mat[i, j] = dx @ S_inv @ dx  # Mahalanobis distance squared

                # Also compute Euclidean distances for search radius gating
                dist_mat = cdist(pos1, pos2, metric='euclidean')
            else:
                # Fallback to Euclidean (e.g. first frame, no Kalman info)
                dist_mat = cdist(pos1, pos2, metric='euclidean')
                base_cost_mat = dist_mat ** 2

            # Calculate adaptive search radii
            search_radii = self._calculate_search_radii(frame1_info, kalman_info, prob_dim)

            # Apply search radius constraints
            for i in range(n1):
                if i < len(search_radii):
                    mask = dist_mat[i, :] > search_radii[i]
                    base_cost_mat[i, mask] = np.inf

            # Intensity-based cost component (log-ratio penalty)
            use_intensity = getattr(self.config, 'use_intensity_costs', False)
            intensity_weight = getattr(self.config, 'intensity_weight', 0.1)
            if use_intensity and intensity_weight > 0:
                # Extract intensities from all_coord (column 4 = amplitude)
                amp1 = frame1_info.all_coord[:, 4] if frame1_info.all_coord.shape[1] > 4 else None
                amp2 = frame2_info.all_coord[:, 4] if frame2_info.all_coord.shape[1] > 4 else None
                if amp1 is not None and amp2 is not None:
                    for i in range(n1):
                        if amp1[i] <= 0:
                            continue
                        for j in range(n2):
                            if np.isfinite(base_cost_mat[i, j]) and amp2[j] > 0:
                                log_ratio = abs(np.log(amp2[j] / amp1[i]))
                                base_cost_mat[i, j] += intensity_weight * log_ratio

            # Velocity angle constraints for linear motion
            use_velocity = getattr(self.config, 'use_velocity_costs', False)
            velocity_weight = getattr(self.config, 'velocity_weight', 0.1)
            if use_velocity and velocity_weight > 0 and kalman_info is not None:
                if kalman_info.state_vec.shape[1] > prob_dim:
                    # Kalman state has velocity components
                    for i in range(min(n1, len(kalman_info.state_vec))):
                        vel_i = kalman_info.state_vec[i, prob_dim:2*prob_dim]
                        speed_i = np.linalg.norm(vel_i)
                        if speed_i < 1e-6:
                            continue  # no velocity info for stationary particles
                        pred_pos_i = kalman_info.state_vec[i, :prob_dim]
                        for j in range(n2):
                            if not np.isfinite(base_cost_mat[i, j]):
                                continue
                            displacement = pos2[j] - pred_pos_i
                            disp_norm = np.linalg.norm(displacement)
                            if disp_norm < 1e-6:
                                continue
                            cos_theta = np.dot(vel_i, displacement) / (speed_i * disp_norm)
                            cos_theta = np.clip(cos_theta, -1.0, 1.0)
                            angle_cost = (1.0 - cos_theta) * speed_i
                            base_cost_mat[i, j] += velocity_weight * angle_cost

            # Create augmented cost matrix for LAP
            matrix_size = n1 + n2
            cost_mat_full = np.full((matrix_size, matrix_size), np.inf)

            # Upper-left: linking costs
            cost_mat_full[:n1, :n2] = base_cost_mat

            # Adaptive birth/death costs (U-Track 2.5 style)
            # Use percentile of valid linking costs as the alternative cost
            valid_costs = base_cost_mat[np.isfinite(base_cost_mat)]
            if len(valid_costs) > 0:
                alt_cost_factor = self.cost_params_link.alternative_cost_factor
                percentile_cost = np.percentile(valid_costs, 90) * alt_cost_factor
                # Ensure a reasonable minimum so very sparse frames still work
                min_alt_cost = self.cost_params_link.min_search_radius ** 2
                alt_cost = max(percentile_cost, min_alt_cost)
            else:
                # No valid links at all — use search radius squared as fallback
                alt_cost = self.cost_params_link.max_search_radius ** 2

            death_costs = np.full(n1, alt_cost)
            birth_costs = np.full(n2, alt_cost)

            # Upper-right: death costs (diagonal)
            cost_mat_full[:n1, n2:n2 + n1] = np.diag(death_costs)

            # Lower-left: birth costs (diagonal)
            cost_mat_full[n1:n1 + n2, :n2] = np.diag(birth_costs)

            # Lower-right: dummy costs (must be <= min of row/col alternative cost
            # for LAP to produce valid results — use smallest valid link cost)
            if len(valid_costs) > 0:
                dummy_cost = np.min(valid_costs)
            else:
                dummy_cost = alt_cost
            cost_mat_full[n1:, n2:] = dummy_cost

            return cost_mat_full, -1

        except Exception as e:
            self.logger.debug(f"Cost matrix calculation failed: {e}")
            # Fallback to simple matrix
            matrix_size = max(n1, n2, 1)
            fallback_matrix = np.full((matrix_size, matrix_size), 1000.0)
            return fallback_matrix, -1

    def _calculate_search_radii(self, frame_info: MovieInfo,
                              kalman_info: KalmanFilterInfo, prob_dim: int) -> np.ndarray:
        """Calculate adaptive search radii for each particle.

        Uses Kalman filter uncertainty when available.  When
        ``use_local_density`` is enabled, scales each particle's radius
        down in dense regions so that the search area doesn't overlap
        too many neighbours (matching U-Track 2.5 behaviour).

        Applies confidence ramp-up (U-Track 2.5 timeReachConf): young
        tracks use wider search radii that narrow as the Kalman filter
        converges.
        """
        n_particles = frame_info.num

        if n_particles == 0:
            return np.array([])

        max_radius = self.cost_params_link.max_search_radius
        min_radius = self.cost_params_link.min_search_radius
        brown_std_mult = self.cost_params_link.brown_std_mult
        time_reach_conf = self.cost_params_link.time_reach_conf_b

        if kalman_info is not None and kalman_info.noise_var.shape[2] >= n_particles:
            # Use Kalman filter uncertainty
            kalman_radii = np.zeros(n_particles)
            for i in range(n_particles):
                pos_var = np.diag(kalman_info.noise_var[:prob_dim, :prob_dim, i])
                search_std = np.sqrt(np.mean(pos_var))
                kalman_radii[i] = brown_std_mult * search_std

            # Confidence ramp-up: interpolate between max_radius and Kalman radius
            # based on track age.  confidence = min(1, age / time_reach_conf)
            if kalman_info.track_ages is not None and time_reach_conf > 0:
                search_radii = np.zeros(n_particles)
                for i in range(n_particles):
                    age = kalman_info.track_ages[i] if i < len(kalman_info.track_ages) else 1
                    confidence = min(1.0, age / time_reach_conf)
                    # Interpolate: low confidence → max_radius, high confidence → Kalman radius
                    search_radii[i] = (1.0 - confidence) * max_radius + confidence * kalman_radii[i]
            else:
                search_radii = kalman_radii
        else:
            # Default uniform search radius
            search_radii = np.full(n_particles, max_radius)

        # Apply bounds
        search_radii = np.clip(search_radii, min_radius, max_radius)

        # Local density scaling (U-Track 2.5 style)
        # In dense regions, reduce search radius so it doesn't extend
        # beyond half the nearest-neighbour distance.
        if self.cost_params_link.use_local_density and n_particles > 1:
            nn_dist = frame_info.nn_dist
            if nn_dist is not None and len(nn_dist) == n_particles:
                # Clamp search radius to half the nearest-neighbour distance
                density_limit = nn_dist * 0.5
                density_limit = np.clip(density_limit, min_radius, max_radius)
                search_radii = np.minimum(search_radii, density_limit)

        return search_radii

    def _solve_lap(self, cost_matrix: np.ndarray, nonlink_marker: float = -1) -> np.ndarray:
        """Solve Linear Assignment Problem using scipy"""
        # Handle infinite costs
        cost_mat = cost_matrix.copy()
        inf_mask = np.isinf(cost_mat)
        if np.any(inf_mask):
            max_finite = np.max(cost_mat[~inf_mask]) if np.any(~inf_mask) else 1000.0
            cost_mat[inf_mask] = max_finite * 1000

        try:
            # Solve using scipy
            row_ind, col_ind = linear_sum_assignment(cost_mat)

            # Convert to 1-indexed assignment vector like MATLAB LAP
            n_rows = cost_matrix.shape[0]
            assignments = np.zeros(n_rows + 1, dtype=int)  # 1-indexed

            for row, col in zip(row_ind, col_ind):
                if not inf_mask[row, col]:  # Valid assignment
                    assignments[row + 1] = col + 1  # Convert to 1-indexed

            return assignments

        except Exception as e:
            warnings.warn(f"LAP solver failed: {e}")
            return np.zeros(cost_matrix.shape[0] + 1, dtype=int)

    def _handle_single_frame(self, detections: List[pd.DataFrame]) -> pd.DataFrame:
        """Handle case with only one frame"""
        if len(detections) == 0:
            return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

        frame_data = detections[0].copy()
        if len(frame_data) > 0:
            frame_data['track_id'] = range(1, len(frame_data) + 1)
            frame_data['particle_id'] = frame_data.index + 1

            # Ensure we have the required columns
            required_cols = ['particle_id', 'track_id', 'frame', 'x', 'y']
            for col in required_cols:
                if col not in frame_data.columns:
                    if col == 'frame':
                        frame_data[col] = 0
                    elif col in ['particle_id', 'track_id']:
                        frame_data[col] = range(1, len(frame_data) + 1)
                    else:
                        frame_data[col] = 0

            # Add intensity if missing
            if 'intensity' not in frame_data.columns:
                frame_data['intensity'] = 100.0

            final_cols = ['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity']
            return frame_data[final_cols]

        return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])


def create_utrack_linker(config: Optional[TrackingConfig] = None) -> UTrackLinker:
    """Factory function to create a u-track compatible linker"""
    return UTrackLinker(config)
