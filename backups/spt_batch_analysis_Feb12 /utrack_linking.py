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


@dataclass
class TrackSegment:
    """Individual track segment after frame-to-frame linking"""
    tracks_feat_indx: np.ndarray    # Feature indices (1-based, 0=gap)
    tracks_coord_amp: np.ndarray    # Coordinates and amplitudes
    n_segments: int = 1             # Number of segments
    n_frames: int = 0              # Number of frames


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
                        n_frames=1
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
                x_coord = np.column_stack([
                    frame_detections['x'].values,
                    np.full(len(frame_detections), 0.1)  # Default uncertainty
                ])
                y_coord = np.column_stack([
                    frame_detections['y'].values,
                    np.full(len(frame_detections), 0.1)
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
        """PRACTICAL FIX: Create tracks based on spatial proximity when LAP fails"""
        n_frames = len(movie_info)

        if n_frames < 2 or not all_assignments:
            return []

        self.logger.debug(f"Converting assignments to tracks for {n_frames} frames")

        # Process assignments for each frame pair
        completed_tracks = []

        for frame_idx, assignments in enumerate(all_assignments):
            if len(assignments) == 0:
                continue

            n_source = movie_info[frame_idx].num
            n_target = movie_info[frame_idx + 1].num

            self.logger.debug(f"Processing frame {frame_idx}->{frame_idx + 1}: {n_source}->{n_target} particles")

            # First try: Extract LAP links
            particle_links = {}
            for source_particle in range(1, n_source + 1):  # 1-indexed
                if source_particle < len(assignments):
                    target_assignment = assignments[source_particle]
                    if 1 <= target_assignment <= n_target:
                        particle_links[source_particle] = target_assignment
                        self.logger.debug(f"LAP link: source {source_particle} -> target {target_assignment}")

            self.logger.debug(f"Found {len(particle_links)} LAP particle links")

            # PRACTICAL FIX: If LAP found no links but particles are close, create spatial links
            if len(particle_links) == 0:
                self.logger.debug("No LAP links found, trying spatial proximity matching...")
                particle_links = self._create_spatial_links(movie_info[frame_idx], movie_info[frame_idx + 1])
                self.logger.debug(f"Created {len(particle_links)} spatial links")

            # Create tracks from links
            for source_p, target_p in particle_links.items():
                track_data = {
                    'frames': [frame_idx, frame_idx + 1],
                    'particles': [source_p, target_p]
                }
                segment = self._build_track_segment_from_data(track_data, movie_info, prob_dim)
                if segment is not None:
                    completed_tracks.append(segment)
                    self.logger.debug(f"Created 2-frame track: {source_p} -> {target_p}")

            # Create single-frame tracks for unlinked particles
            linked_sources = set(particle_links.keys())
            linked_targets = set(particle_links.values())

            # Unlinked source particles (deaths)
            for source_p in range(1, n_source + 1):
                if source_p not in linked_sources:
                    track_data = {'frames': [frame_idx], 'particles': [source_p]}
                    segment = self._build_track_segment_from_data(track_data, movie_info, prob_dim)
                    if segment is not None:
                        completed_tracks.append(segment)
                        self.logger.debug(f"Death track: source {source_p}")

            # Unlinked target particles (births)
            for target_p in range(1, n_target + 1):
                if target_p not in linked_targets:
                    track_data = {'frames': [frame_idx + 1], 'particles': [target_p]}
                    segment = self._build_track_segment_from_data(track_data, movie_info, prob_dim)
                    if segment is not None:
                        completed_tracks.append(segment)
                        self.logger.debug(f"Birth track: target {target_p}")

        self.logger.debug(f"Created {len(completed_tracks)} total track segments")
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
            n_frames=n_frames_total
        )

    # def _build_track_segment_fixed(self, track_data: Dict, movie_info: List[MovieInfo],
    #                              prob_dim: int) -> Optional[TrackSegment]:
    #     """FIXED: Build a TrackSegment from track data"""
    #     frames = track_data['frames']
    #     particles = track_data['particles']

    #     if len(frames) == 0:
    #         return None

    #     start_frame = min(frames)
    #     end_frame = max(frames)
    #     n_frames_total = end_frame - start_frame + 1

    #     # Initialize arrays
    #     feat_indx = np.zeros(n_frames_total)
    #     coord_amp = np.full(n_frames_total * 8, np.nan)

    #     # Fill in track data
    #     for i, (frame, particle) in enumerate(zip(frames, particles)):
    #         relative_frame = frame - start_frame
    #         feat_indx[relative_frame] = particle

    #         # Get coordinates and amplitude
    #         if frame < len(movie_info) and particle <= movie_info[frame].num:
    #             frame_info = movie_info[frame]
    #             particle_idx = particle - 1  # Convert to 0-indexed

    #             base_idx = relative_frame * 8

    #             # Fill coordinates (handle potential missing data)
    #             if particle_idx < len(frame_info.all_coord):
    #                 coord_amp[base_idx:base_idx+2] = frame_info.all_coord[particle_idx, 0:2]    # x, dx
    #                 coord_amp[base_idx+2:base_idx+4] = frame_info.all_coord[particle_idx, 2:4]  # y, dy
    #                 coord_amp[base_idx+6:base_idx+8] = frame_info.all_coord[particle_idx, 4:6]  # amp, damp

    #     return TrackSegment(
    #         tracks_feat_indx=feat_indx,
    #         tracks_coord_amp=coord_amp,
    #         n_segments=1,
    #         n_frames=n_frames_total
    #     )

    def _close_gaps_kalman_sparse(self, track_segments: List[TrackSegment],
                                kalman_info: List[KalmanFilterInfo],
                                prob_dim: int) -> List[CompoundTrack]:
        """
        FIXED: Step 2: Gap closing and merge/split detection
        """
        # Filter short tracks
        min_track_len = max(1, self.gap_close_params.min_track_len - 1)  # Be more lenient for 2-frame sequences
        long_segments = []

        for segment in track_segments:
            valid_count = np.sum(segment.tracks_feat_indx > 0)
            if valid_count >= min_track_len:
                long_segments.append(segment)
            else:
                self.logger.debug(f"Filtered out track with {valid_count} points (< {min_track_len})")

        self.logger.debug(f"Kept {len(long_segments)} segments after length filtering")

        if not long_segments:
            return []

        # For now, convert each segment to a compound track directly
        # In full implementation, this would involve gap closing LAP
        compound_tracks = []
        for segment_idx, segment in enumerate(long_segments):
            # Create sequence of events
            valid_frames = np.where(segment.tracks_feat_indx > 0)[0]
            if len(valid_frames) > 0:
                start_frame = valid_frames[0] + 1  # Convert to 1-indexed
                end_frame = valid_frames[-1] + 1

                seq_of_events = np.array([
                    [start_frame, 1, 1, np.nan],  # Start event
                    [end_frame, 2, 1, np.nan]     # End event
                ])

                compound_track = CompoundTrack(
                    tracks_feat_indx_cg=segment.tracks_feat_indx,
                    tracks_coord_amp_cg=segment.tracks_coord_amp,
                    seq_of_events=seq_of_events
                )
                compound_tracks.append(compound_track)
                self.logger.debug(f"Created compound track {segment_idx}: frames {start_frame}-{end_frame}")

        return compound_tracks

    # def _convert_compound_tracks_to_dataframe_fixed(self, compound_tracks: List[CompoundTrack],
    #                                               original_detections: List[pd.DataFrame]) -> pd.DataFrame:
    #     """CORRECTED: Convert compound tracks back to standard DataFrame format"""
    #     if not compound_tracks:
    #         return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

    #     data = []

    #     for track_id, track in enumerate(compound_tracks):
    #         try:
    #             # Get the length based on tracks_feat_indx_cg
    #             if hasattr(track, 'tracks_feat_indx_cg') and len(track.tracks_feat_indx_cg) > 0:
    #                 n_frames = len(track.tracks_feat_indx_cg)
    #             elif hasattr(track, 'n_frames') and track.n_frames > 0:
    #                 n_frames = track.n_frames
    #             else:
    #                 # Try to infer from coordinate data
    #                 coord_length = len(track.tracks_coord_amp_cg) if hasattr(track, 'tracks_coord_amp_cg') else 0
    #                 n_frames = coord_length // 8  # 8 elements per time point

    #             if n_frames <= 0:
    #                 self.logger.debug(f"Track {track_id}: no valid frames")
    #                 continue

    #             # CORRECTED: tracksCoordAmpCG is a 1D array with 8 elements per time point
    #             coords_1d = track.tracks_coord_amp_cg

    #             if len(coords_1d) < n_frames * 8:
    #                 self.logger.debug(f"Track {track_id}: insufficient coordinate data")
    #                 continue

    #             self.logger.debug(f"Track {track_id}: processing {n_frames} frames")

    #             for frame_idx in range(n_frames):
    #                 # Check if this frame has a valid particle
    #                 particle_id = track.tracks_feat_indx_cg[frame_idx] if frame_idx < len(track.tracks_feat_indx_cg) else 0

    #                 if particle_id > 0:  # Valid particle (1-based indexing, 0 = gap)
    #                     # Extract coordinates for this time point
    #                     base_idx = frame_idx * 8

    #                     # CORRECTED: Extract from 1D array
    #                     x_coord = coords_1d[base_idx] if not np.isnan(coords_1d[base_idx]) else 0.0
    #                     y_coord = coords_1d[base_idx + 1] if not np.isnan(coords_1d[base_idx + 1]) else 0.0
    #                     # Skip z_coord (index 2) since we're working in 2D
    #                     intensity = coords_1d[base_idx + 3] if not np.isnan(coords_1d[base_idx + 3]) else 100.0

    #                     data.append({
    #                         'particle_id': int(particle_id),
    #                         'track_id': track_id + 1,  # 1-indexed
    #                         'frame': frame_idx,
    #                         'x': x_coord,
    #                         'y': y_coord,
    #                         'intensity': intensity
    #                     })

    #         except Exception as e:
    #             self.logger.warning(f"Error processing track {track_id}: {e}")
    #             self.logger.debug(f"Track structure: feat_indx length={len(track.tracks_feat_indx_cg) if hasattr(track, 'tracks_feat_indx_cg') else 'N/A'}, "
    #                             f"coord_amp length={len(track.tracks_coord_amp_cg) if hasattr(track, 'tracks_coord_amp_cg') else 'N/A'}")
    #             continue

    #     if data:
    #         result_df = pd.DataFrame(data)
    #         self.logger.info(f"Successfully converted {len(compound_tracks)} compound tracks to {len(result_df)} track points")
    #         return result_df
    #     else:
    #         self.logger.warning("No valid track data could be extracted from compound tracks")
    #         return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

    def _convert_compound_tracks_to_dataframe_fixed(self, compound_tracks: List[CompoundTrack],
                                                  original_detections: List[pd.DataFrame]) -> pd.DataFrame:
        """DEBUG VERSION: Convert compound tracks with detailed logging"""
        if not compound_tracks:
            return pd.DataFrame(columns=['particle_id', 'track_id', 'frame', 'x', 'y', 'intensity'])

        data = []

        for track_id, track in enumerate(compound_tracks[:3]):  # Only process first 3 tracks for debugging
            try:
                self.logger.info(f"=== DEBUGGING TRACK {track_id} ===")

                # Get the length based on tracks_feat_indx_cg
                if hasattr(track, 'tracks_feat_indx_cg') and len(track.tracks_feat_indx_cg) > 0:
                    n_frames = len(track.tracks_feat_indx_cg)
                    self.logger.info(f"Track {track_id}: n_frames from feat_indx = {n_frames}")
                    self.logger.info(f"Track {track_id}: feat_indx = {track.tracks_feat_indx_cg}")
                else:
                    self.logger.warning(f"Track {track_id}: No valid feat_indx")
                    continue

                # DETAILED COORDINATE ARRAY INSPECTION
                coords_1d = track.tracks_coord_amp_cg
                self.logger.info(f"Track {track_id}: coords_1d length = {len(coords_1d)}")
                self.logger.info(f"Track {track_id}: coords_1d type = {type(coords_1d)}")

                # Show the entire coordinate array for first track
                if track_id == 0:
                    self.logger.info(f"Track {track_id}: FULL coords_1d = {coords_1d}")

                if len(coords_1d) < n_frames * 8:
                    self.logger.warning(f"Track {track_id}: insufficient coordinate data ({len(coords_1d)} < {n_frames * 8})")
                    continue

                # Process each frame with detailed logging
                for frame_idx in range(min(n_frames, 3)):  # Only first 3 frames for debugging
                    particle_id = track.tracks_feat_indx_cg[frame_idx] if frame_idx < len(track.tracks_feat_indx_cg) else 0

                    self.logger.info(f"  Frame {frame_idx}: particle_id = {particle_id}")

                    if particle_id > 0:  # Valid particle
                        base_idx = frame_idx * 8
                        self.logger.info(f"  Frame {frame_idx}: base_idx = {base_idx}")

                        # Extract and log each coordinate component
                        if base_idx + 7 < len(coords_1d):
                            raw_coords = coords_1d[base_idx:base_idx+8]
                            self.logger.info(f"  Frame {frame_idx}: raw 8 values = {raw_coords}")

                            x_coord = raw_coords[0] if not np.isnan(raw_coords[0]) else 0.0
                            y_coord = raw_coords[1] if not np.isnan(raw_coords[1]) else 0.0
                            z_coord = raw_coords[2] if not np.isnan(raw_coords[2]) else 0.0
                            intensity = raw_coords[3] if not np.isnan(raw_coords[3]) else 100.0
                            dx = raw_coords[4] if not np.isnan(raw_coords[4]) else 0.1
                            dy = raw_coords[5] if not np.isnan(raw_coords[5]) else 0.1
                            dz = raw_coords[6] if not np.isnan(raw_coords[6]) else 0.1
                            damp = raw_coords[7] if not np.isnan(raw_coords[7]) else 0.1

                            self.logger.info(f"  Frame {frame_idx}: extracted x={x_coord}, y={y_coord}, z={z_coord}, amp={intensity}")
                            self.logger.info(f"  Frame {frame_idx}: uncertainties dx={dx}, dy={dy}, dz={dz}, damp={damp}")

                            data.append({
                                'particle_id': int(particle_id),
                                'track_id': track_id + 1,
                                'frame': frame_idx,
                                'x': x_coord,
                                'y': y_coord,
                                'intensity': intensity
                            })
                        else:
                            self.logger.warning(f"  Frame {frame_idx}: base_idx {base_idx} would exceed array bounds")

            except Exception as e:
                self.logger.error(f"Error processing track {track_id}: {e}")
                import traceback
                self.logger.debug(f"Full traceback: {traceback.format_exc()}")
                continue

        if data:
            result_df = pd.DataFrame(data)
            self.logger.info(f"DEBUG: Generated DataFrame with {len(result_df)} rows")
            self.logger.info(f"DEBUG: DataFrame sample:\n{result_df.head()}")
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
            # Position uncertainties from detection
            pos_var = np.array([0.1, 0.1])  # Default uncertainty
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
            observation_mat=observation_mat
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
            R = np.diag([0.1, 0.1])  # Observation noise

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

        return KalmanFilterInfo(
            state_vec=state_vec,
            state_cov=state_cov,
            noise_var=noise_var,
            observation_mat=H
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

            # Calculate pairwise distances
            from scipy.spatial.distance import cdist
            dist_mat = cdist(pos1, pos2, metric='euclidean')

            # Base cost matrix (squared distances)
            base_cost_mat = dist_mat ** 2

            # Calculate adaptive search radii
            search_radii = self._calculate_search_radii(frame1_info, kalman_info, prob_dim)

            # Apply search radius constraints
            for i in range(n1):
                if i < len(search_radii):
                    mask = dist_mat[i, :] > search_radii[i]
                    base_cost_mat[i, mask] = np.inf

            # Create augmented cost matrix for LAP
            matrix_size = n1 + n2
            cost_mat_full = np.full((matrix_size, matrix_size), np.inf)

            # Upper-left: linking costs
            cost_mat_full[:n1, :n2] = base_cost_mat

            # Calculate birth and death costs
            death_costs = np.full(n1, 100.0)
            birth_costs = np.full(n2, 100.0)

            # Upper-right: death costs (diagonal)
            if n1 <= matrix_size - n2:
                cost_mat_full[:n1, n2:n2+n1] = np.diag(death_costs)

            # Lower-left: birth costs (diagonal)
            if n2 <= matrix_size - n1:
                cost_mat_full[n1:n1+n2, :n2] = np.diag(birth_costs)

            # Lower-right: dummy costs
            cost_mat_full[n1:, n2:] = 200.0

            return cost_mat_full, -1

        except Exception as e:
            self.logger.debug(f"Cost matrix calculation failed: {e}")
            # Fallback to simple matrix
            matrix_size = max(n1, n2, 1)
            fallback_matrix = np.full((matrix_size, matrix_size), 1000.0)
            return fallback_matrix, -1

    def _calculate_search_radii(self, frame_info: MovieInfo,
                              kalman_info: KalmanFilterInfo, prob_dim: int) -> np.ndarray:
        """Calculate adaptive search radii for each particle"""
        n_particles = frame_info.num

        if n_particles == 0:
            return np.array([])

        max_radius = self.cost_params_link.max_search_radius
        min_radius = self.cost_params_link.min_search_radius
        brown_std_mult = self.cost_params_link.brown_std_mult

        if kalman_info is not None and kalman_info.noise_var.shape[2] >= n_particles:
            # Use Kalman filter uncertainty
            search_radii = np.zeros(n_particles)
            for i in range(n_particles):
                pos_var = np.diag(kalman_info.noise_var[:prob_dim, :prob_dim, i])
                search_std = np.sqrt(np.mean(pos_var))
                search_radii[i] = brown_std_mult * search_std
        else:
            # Default uniform search radius
            search_radii = np.full(n_particles, max_radius)

        # Apply bounds
        search_radii = np.clip(search_radii, min_radius, max_radius)

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
