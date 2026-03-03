"""
Post-Processing Module
=====================

Implements post-processing methods:
- Drift correction (fiducial markers and cross-correlation)
- Merging of reappearing molecules
- Duplicate removal
- Filtering based on localization quality
- Local density filtering
- Z-stage offset correction
"""

import numpy as np
from scipy import ndimage, signal
from scipy.spatial import cKDTree
from sklearn.linear_model import RANSACRegressor


class DriftCorrector:
    """Correct for sample drift during acquisition.
    
    Supports two methods:
    1. Fiducial marker tracking
    2. Cross-correlation of reconstructed images
    
    Parameters
    ----------
    method : str
        'fiducial' or 'cross_correlation'
    smoothing : float
        Smoothing parameter for drift trajectory
    """
    
    def __init__(self, method='cross_correlation', smoothing=0.25):
        self.method = method
        self.smoothing = smoothing
        self.drift_x = None
        self.drift_y = None
        
    def compute_drift_fiducial(self, localizations, fiducial_region, frames):
        """Compute drift from fiducial markers.
        
        Parameters
        ----------
        localizations : dict
            Localization data with 'x', 'y', 'frame'
        fiducial_region : tuple
            (x_min, x_max, y_min, y_max) defining fiducial region
        frames : ndarray
            Frame numbers
            
        Returns
        -------
        drift_x, drift_y : ndarray
            Drift in x and y for each frame
        """
        x_min, x_max, y_min, y_max = fiducial_region
        
        # Extract fiducial localizations
        mask = ((localizations['x'] >= x_min) & (localizations['x'] <= x_max) &
                (localizations['y'] >= y_min) & (localizations['y'] <= y_max))
        
        fid_x = localizations['x'][mask]
        fid_y = localizations['y'][mask]
        fid_frames = localizations['frame'][mask]
        
        # Compute mean position per frame
        drift_x = np.zeros(len(frames))
        drift_y = np.zeros(len(frames))
        
        for i, frame in enumerate(frames):
            frame_mask = fid_frames == frame
            if np.sum(frame_mask) > 0:
                drift_x[i] = np.mean(fid_x[frame_mask])
                drift_y[i] = np.mean(fid_y[frame_mask])
        
        # Smooth drift trajectory using LOWESS-like smoothing
        drift_x = self._smooth_trajectory(drift_x)
        drift_y = self._smooth_trajectory(drift_y)
        
        # Convert to relative drift (subtract initial position)
        drift_x = drift_x - drift_x[0]
        drift_y = drift_y - drift_y[0]
        
        self.drift_x = drift_x
        self.drift_y = drift_y
        
        return drift_x, drift_y
        
    def compute_drift_xcorr(self, localizations, frames, pixel_size=10, 
                           image_size=None, segment_frames=500):
        """Compute drift using cross-correlation.
        
        Parameters
        ----------
        localizations : dict
            Localization data with 'x', 'y', 'frame'
        frames : ndarray
            Frame numbers
        pixel_size : float
            Pixel size for reconstruction (nm)
        image_size : tuple, optional
            (width, height) of reconstruction
        segment_frames : int
            Number of frames per segment for cross-correlation
            
        Returns
        -------
        drift_x, drift_y : ndarray
            Drift in x and y for each frame
        """
        # Determine image size
        if image_size is None:
            x_max = int(np.max(localizations['x']) / pixel_size) + 1
            y_max = int(np.max(localizations['y']) / pixel_size) + 1
            image_size = (x_max, y_max)
        
        # Split frames into segments
        n_segments = len(frames) // segment_frames + 1
        segment_drift_x = []
        segment_drift_y = []
        segment_centers = []
        
        # Reconstruct first segment as reference
        ref_frames = frames[:segment_frames]
        ref_img = self._reconstruct_image(
            localizations, ref_frames, pixel_size, image_size
        )
        
        # Process each segment
        for i in range(n_segments):
            start_frame = i * segment_frames
            end_frame = min((i + 1) * segment_frames, len(frames))
            
            if start_frame >= len(frames):
                break
                
            seg_frames = frames[start_frame:end_frame]
            seg_img = self._reconstruct_image(
                localizations, seg_frames, pixel_size, image_size
            )
            
            # Cross-correlate with reference
            xcorr = signal.correlate2d(ref_img, seg_img, mode='same')
            
            # Find peak
            peak_y, peak_x = np.unravel_index(np.argmax(xcorr), xcorr.shape)
            
            # Convert to drift (relative to center)
            center_y, center_x = np.array(xcorr.shape) // 2
            drift_x_pix = peak_x - center_x
            drift_y_pix = peak_y - center_y
            
            segment_drift_x.append(drift_x_pix * pixel_size)
            segment_drift_y.append(drift_y_pix * pixel_size)
            segment_centers.append((start_frame + end_frame) / 2)
        
        # Interpolate drift for all frames
        drift_x = np.interp(np.arange(len(frames)), segment_centers, segment_drift_x)
        drift_y = np.interp(np.arange(len(frames)), segment_centers, segment_drift_y)
        
        # Smooth
        drift_x = self._smooth_trajectory(drift_x)
        drift_y = self._smooth_trajectory(drift_y)
        
        self.drift_x = drift_x
        self.drift_y = drift_y
        
        return drift_x, drift_y
        
    def apply_drift_correction(self, localizations):
        """Apply computed drift correction to localizations.
        
        Parameters
        ----------
        localizations : dict
            Localization data with 'x', 'y', 'frame'
            
        Returns
        -------
        corrected : dict
            Drift-corrected localizations
        """
        if self.drift_x is None or self.drift_y is None:
            raise ValueError("Must compute drift first")
        
        corrected = localizations.copy()
        
        # Apply drift correction per frame
        for frame_idx in np.unique(localizations['frame']):
            mask = localizations['frame'] == frame_idx
            if frame_idx < len(self.drift_x):
                corrected['x'][mask] -= self.drift_x[int(frame_idx)]
                corrected['y'][mask] -= self.drift_y[int(frame_idx)]
        
        return corrected
        
    def _reconstruct_image(self, localizations, frames, pixel_size, image_size):
        """Reconstruct super-resolution image from localizations.

        image_size is (width, height) i.e. (n_cols, n_rows).
        """
        width, height = image_size
        img = np.zeros((height, width))

        for frame in frames:
            mask = localizations['frame'] == frame
            x_coords = (localizations['x'][mask] / pixel_size).astype(int)
            y_coords = (localizations['y'][mask] / pixel_size).astype(int)

            # Clip to image bounds
            valid = ((x_coords >= 0) & (x_coords < width) &
                    (y_coords >= 0) & (y_coords < height))

            x_coords = x_coords[valid]
            y_coords = y_coords[valid]

            # Accumulate
            for x, y in zip(x_coords, y_coords):
                img[y, x] += 1

        return img
        
    def _smooth_trajectory(self, trajectory):
        """Smooth drift trajectory using Gaussian filter."""
        window = int(len(trajectory) * self.smoothing)
        if window < 3:
            window = 3
        sigma = window / 6.0

        # Handle NaN values
        valid = ~np.isnan(trajectory)
        if not np.any(valid):
            return trajectory

        # Interpolate NaN values
        interp_traj = np.copy(trajectory)
        if not np.all(valid):
            indices = np.arange(len(trajectory))
            interp_traj = np.interp(indices, indices[valid], trajectory[valid])

        # Smooth
        smoothed = ndimage.gaussian_filter1d(interp_traj, sigma)

        return smoothed

    def apply_drift_correction_df(self, df, x_col='x', y_col='y', frame_col='frame'):
        """Apply drift correction to a pandas DataFrame.

        Convenience adapter that converts DataFrame ↔ dict format
        used by the existing compute/apply methods.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain x, y, frame columns.
        x_col, y_col, frame_col : str
            Column names.

        Returns
        -------
        corrected_df : DataFrame
            Drift-corrected copy of the input.
        drift_x, drift_y : ndarray
            Drift vectors per frame.
        """
        import pandas as pd

        localizations = {
            'x': df[x_col].values.copy().astype(float),
            'y': df[y_col].values.copy().astype(float),
            'frame': df[frame_col].values.copy().astype(float),
        }

        frames = np.arange(int(localizations['frame'].min()),
                           int(localizations['frame'].max()) + 1)

        # Compute drift if not already done
        if self.drift_x is None:
            if self.method == 'fiducial':
                raise ValueError("Fiducial method requires fiducial_region; "
                                 "call compute_drift_fiducial first.")
            self.compute_drift_xcorr(localizations, frames)

        corrected = self.apply_drift_correction(localizations)

        corrected_df = df.copy()
        corrected_df[x_col] = corrected['x']
        corrected_df[y_col] = corrected['y']

        return corrected_df, self.drift_x, self.drift_y


class MolecularMerger:
    """Merge molecules appearing in subsequent frames.
    
    Accounts for blinking - molecules can disappear for several frames.
    
    Parameters
    ----------
    max_distance : float
        Maximum distance (nm) to consider molecules the same
    max_frame_gap : int
        Maximum frame gap for reappearance
    """
    
    def __init__(self, max_distance=50, max_frame_gap=1):
        self.max_distance = max_distance
        self.max_frame_gap = max_frame_gap
        
    def merge(self, localizations):
        """Merge reappearing molecules.
        
        Parameters
        ----------
        localizations : dict
            Localization data
            
        Returns
        -------
        merged : dict
            Merged localizations
        """
        # Sort by frame
        sort_idx = np.argsort(localizations['frame'])
        
        x = localizations['x'][sort_idx]
        y = localizations['y'][sort_idx]
        frames = localizations['frame'][sort_idx]
        
        # Track which localizations have been merged
        merged_into = -np.ones(len(x), dtype=int)  # -1 means not merged
        
        # Process frame by frame
        unique_frames = np.unique(frames)
        
        for i, frame in enumerate(unique_frames[:-1]):
            # Get localizations in current frame
            current_mask = frames == frame
            current_indices = np.where(current_mask)[0]
            
            # Look ahead up to max_frame_gap
            for gap in range(1, self.max_frame_gap + 1):
                if i + gap >= len(unique_frames):
                    break
                    
                next_frame = unique_frames[i + gap]
                next_mask = frames == next_frame
                next_indices = np.where(next_mask)[0]
                
                # Build KD-tree for current frame
                current_positions = np.column_stack([x[current_indices], y[current_indices]])
                
                if len(current_positions) == 0:
                    continue
                    
                tree = cKDTree(current_positions)
                
                # Query for each next frame localization
                next_positions = np.column_stack([x[next_indices], y[next_indices]])
                
                for j, next_idx in enumerate(next_indices):
                    if merged_into[next_idx] >= 0:
                        # Already merged
                        continue
                        
                    # Find nearest neighbor in current frame
                    dist, idx = tree.query(next_positions[j])
                    
                    if dist <= self.max_distance:
                        # Merge: mark as merged into current localization
                        current_idx = current_indices[idx]
                        merged_into[next_idx] = current_idx
        
        # Create merged localizations
        # Keep only localizations that are either:
        # 1. Not merged (merged_into == -1)
        # 2. Are the target of a merge
        
        keep_mask = merged_into == -1
        
        merged = {}
        for key in localizations:
            if key in ['x', 'y', 'intensity', 'background', 'sigma_x', 'sigma_y', 'frame']:
                merged[key] = localizations[key][sort_idx][keep_mask]
        
        return merged


class LocalizationFilter:
    """Filter localizations based on quality criteria.
    
    Parameters
    ----------
    min_intensity : float, optional
        Minimum intensity
    max_intensity : float, optional
        Maximum intensity  
    max_uncertainty : float, optional
        Maximum localization uncertainty (nm)
    min_sigma : float, optional
        Minimum PSF sigma
    max_sigma : float, optional
        Maximum PSF sigma
    """
    
    def __init__(self, min_intensity=None, max_intensity=None,
                 max_uncertainty=None, min_sigma=None, max_sigma=None):
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.max_uncertainty = max_uncertainty
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
    def filter(self, localizations):
        """Apply filters to localizations.
        
        Parameters
        ----------
        localizations : dict
            Localization data
            
        Returns
        -------
        filtered : dict
            Filtered localizations
        """
        mask = np.ones(len(localizations['x']), dtype=bool)
        
        # Apply intensity filters
        if self.min_intensity is not None:
            mask &= localizations['intensity'] >= self.min_intensity
            
        if self.max_intensity is not None:
            mask &= localizations['intensity'] <= self.max_intensity
        
        # Apply uncertainty filter
        if self.max_uncertainty is not None and 'uncertainty' in localizations:
            mask &= localizations['uncertainty'] <= self.max_uncertainty
        
        # Apply sigma filters
        if self.min_sigma is not None:
            mask &= localizations['sigma_x'] >= self.min_sigma
            
        if self.max_sigma is not None:
            mask &= localizations['sigma_x'] <= self.max_sigma
        
        # Apply mask
        filtered = {}
        for key, value in localizations.items():
            if isinstance(value, np.ndarray):
                filtered[key] = value[mask]
            else:
                filtered[key] = value
                
        return filtered


class LocalDensityFilter:
    """Filter based on local density (removes isolated localizations).
    
    Parameters
    ----------
    radius : float
        Search radius (nm)
    min_neighbors : int
        Minimum number of neighbors required
    use_3d : bool
        Use 3D distance (requires 'z' coordinate)
    """
    
    def __init__(self, radius=50, min_neighbors=3, use_3d=False):
        self.radius = radius
        self.min_neighbors = min_neighbors
        self.use_3d = use_3d
        
    def filter(self, localizations):
        """Filter by local density.
        
        Parameters
        ----------
        localizations : dict
            Localization data
            
        Returns
        -------
        filtered : dict
            Filtered localizations
        """
        # Build position array
        if self.use_3d and 'z' in localizations:
            positions = np.column_stack([
                localizations['x'],
                localizations['y'],
                localizations['z']
            ])
        else:
            positions = np.column_stack([
                localizations['x'],
                localizations['y']
            ])
        
        # Build KD-tree
        tree = cKDTree(positions)
        
        # Query for neighbors
        neighbors = tree.query_ball_point(positions, self.radius)
        
        # Count neighbors (excluding self)
        neighbor_counts = np.array([len(n) - 1 for n in neighbors])
        
        # Filter
        mask = neighbor_counts >= self.min_neighbors
        
        filtered = {}
        for key, value in localizations.items():
            if isinstance(value, np.ndarray):
                filtered[key] = value[mask]
            else:
                filtered[key] = value
                
        return filtered


class DuplicateRemover:
    """Remove duplicate localizations from multi-emitter fitting.
    
    Parameters
    ----------
    max_distance : float
        Maximum distance (nm) to consider duplicates
    """
    
    def __init__(self, max_distance=20):
        self.max_distance = max_distance
        
    def remove_duplicates(self, localizations):
        """Remove duplicate localizations within same frame.
        
        Parameters
        ----------
        localizations : dict
            Localization data
            
        Returns
        -------
        unique : dict
            Localizations with duplicates removed
        """
        unique_mask = np.ones(len(localizations['x']), dtype=bool)
        
        # Process frame by frame
        for frame in np.unique(localizations['frame']):
            frame_mask = localizations['frame'] == frame
            frame_indices = np.where(frame_mask)[0]
            
            if len(frame_indices) <= 1:
                continue
            
            # Build KD-tree for frame
            positions = np.column_stack([
                localizations['x'][frame_indices],
                localizations['y'][frame_indices]
            ])
            
            tree = cKDTree(positions)
            
            # Find pairs within max_distance
            pairs = tree.query_pairs(self.max_distance)
            
            # For each pair, keep the brighter one
            for i, j in pairs:
                idx_i = frame_indices[i]
                idx_j = frame_indices[j]
                
                if localizations['intensity'][idx_i] < localizations['intensity'][idx_j]:
                    unique_mask[idx_i] = False
                else:
                    unique_mask[idx_j] = False
        
        # Apply mask
        unique = {}
        for key, value in localizations.items():
            if isinstance(value, np.ndarray):
                unique[key] = value[unique_mask]
            else:
                unique[key] = value
                
        return unique


class MSDAnalyzer:
    """Mean Squared Displacement analysis for single-particle trajectories.

    Computes per-track MSD curves, fits diffusion coefficients and
    anomalous exponents, and classifies motion type.

    Parameters
    ----------
    max_lag : int or None
        Maximum lag (in frames) for MSD computation.
        None or 0 = auto (track_length // 4).
    n_fit_points : int
        Number of initial MSD points used for linear D fit.
    anomalous_alpha_low : float
        Alpha below this is classified as confined/subdiffusive.
    anomalous_alpha_high : float
        Alpha above this is classified as directed/superdiffusive.
    """

    def __init__(self, max_lag=None, n_fit_points=4,
                 anomalous_alpha_low=0.7, anomalous_alpha_high=1.3):
        self.max_lag = max_lag
        self.n_fit_points = n_fit_points
        self.anomalous_alpha_low = anomalous_alpha_low
        self.anomalous_alpha_high = anomalous_alpha_high

    # ------------------------------------------------------------------
    # Core MSD computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_msd(track_xy, max_lag=None):
        """Compute time-averaged MSD for a single track.

        Parameters
        ----------
        track_xy : ndarray, shape (N, 2)
            XY positions sorted by time.
        max_lag : int or None
            Maximum lag in frames.  None = N // 4.

        Returns
        -------
        msd : ndarray, shape (max_lag,)
            MSD values for lags 1 .. max_lag.
        n_pairs : ndarray, shape (max_lag,)
            Number of displacement pairs averaged at each lag.
        """
        N = len(track_xy)
        if max_lag is None or max_lag <= 0:
            max_lag = max(1, N // 4)
        max_lag = min(max_lag, N - 1)

        msd = np.zeros(max_lag)
        n_pairs = np.zeros(max_lag, dtype=int)

        for tau in range(1, max_lag + 1):
            displacements = track_xy[tau:] - track_xy[:-tau]
            sq_disp = np.sum(displacements ** 2, axis=1)
            msd[tau - 1] = np.mean(sq_disp)
            n_pairs[tau - 1] = len(sq_disp)

        return msd, n_pairs

    def fit_diffusion_coefficient(self, msd_values, dt, n_fit_points=None):
        """Fit diffusion coefficient from MSD curve.

        Linear fit to first *n_fit_points*:
            MSD = 4 * D * t + 4 * sigma^2   (2-D)

        Parameters
        ----------
        msd_values : ndarray
            MSD values (lags 1 .. N).
        dt : float
            Frame interval in seconds.
        n_fit_points : int or None
            Points to fit.  None = self.n_fit_points.

        Returns
        -------
        D : float
            Diffusion coefficient (same spatial units^2 / s).
        sigma : float
            Localization error estimate (same spatial units).
        r_squared : float
            R^2 of the linear fit.
        """
        if n_fit_points is None:
            n_fit_points = self.n_fit_points
        n_fit_points = min(n_fit_points, len(msd_values))
        if n_fit_points < 2:
            return 0.0, 0.0, 0.0

        t = np.arange(1, n_fit_points + 1) * dt
        y = msd_values[:n_fit_points]

        # Linear least-squares: y = 4*D*t + offset
        A = np.column_stack([t, np.ones(n_fit_points)])
        result, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = result

        D = slope / 4.0
        sigma = np.sqrt(max(intercept / 4.0, 0.0))

        # R-squared
        ss_res = np.sum((y - (slope * t + intercept)) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

        return max(D, 0.0), sigma, r_squared

    def fit_anomalous_exponent(self, msd_values, dt):
        """Fit anomalous diffusion exponent via log-log regression.

        Model: MSD = 4 * D * t^alpha
        => log(MSD) = log(4D) + alpha * log(t)

        Parameters
        ----------
        msd_values : ndarray
            MSD values.
        dt : float
            Frame interval.

        Returns
        -------
        alpha : float
            Anomalous exponent (< 1 subdiffusive, 1 Brownian, > 1 superdiffusive).
        D_alpha : float
            Generalised diffusion coefficient.
        r_squared : float
            R^2 of the log-log fit.
        """
        n = len(msd_values)
        if n < 3:
            return 1.0, 0.0, 0.0

        # Use points where MSD > 0
        t = np.arange(1, n + 1) * dt
        valid = msd_values > 0
        if np.sum(valid) < 3:
            return 1.0, 0.0, 0.0

        log_t = np.log(t[valid])
        log_msd = np.log(msd_values[valid])

        # Linear fit in log-log space
        A = np.column_stack([log_t, np.ones(len(log_t))])
        result, _, _, _ = np.linalg.lstsq(A, log_msd, rcond=None)
        alpha, log_4D = result

        D_alpha = np.exp(log_4D) / 4.0

        # R-squared
        y_pred = alpha * log_t + log_4D
        ss_res = np.sum((log_msd - y_pred) ** 2)
        ss_tot = np.sum((log_msd - np.mean(log_msd)) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-30)

        return alpha, max(D_alpha, 0.0), r_squared

    def classify_motion_from_msd(self, alpha, D):
        """Classify motion type from anomalous exponent and D.

        Returns
        -------
        motion_type : str
            'confined', 'brownian', 'directed', or 'anomalous'
        """
        if alpha < self.anomalous_alpha_low:
            return 'confined'
        elif alpha > self.anomalous_alpha_high:
            return 'directed'
        else:
            return 'brownian'


class VelocityPersistenceAnalyzer:
    """Velocity autocorrelation and directional persistence metrics."""

    @staticmethod
    def compute_velocity_autocorrelation(track_xy, max_lag=10):
        """Normalised velocity autocorrelation C(tau).

        C(tau) = <v(t) . v(t+tau)> / <|v|^2>

        Parameters
        ----------
        track_xy : ndarray, shape (N, 2)
        max_lag : int

        Returns
        -------
        autocorr : ndarray, shape (max_lag,)
            C(tau) for tau = 1 .. max_lag.
        """
        velocities = np.diff(track_xy, axis=0)  # (N-1, 2)
        N = len(velocities)
        if N < 2:
            return np.zeros(max_lag)

        max_lag = min(max_lag, N - 1)
        mean_v2 = np.mean(np.sum(velocities ** 2, axis=1))
        if mean_v2 < 1e-30:
            return np.zeros(max_lag)

        autocorr = np.zeros(max_lag)
        for tau in range(1, max_lag + 1):
            dots = np.sum(velocities[:N - tau] * velocities[tau:], axis=1)
            autocorr[tau - 1] = np.mean(dots) / mean_v2

        return autocorr

    @staticmethod
    def compute_directional_persistence(track_xy):
        """End-to-end distance / total path length.

        Returns 0 for stationary, ~1 for directed motion.
        """
        if len(track_xy) < 2:
            return 0.0

        end_to_end = np.linalg.norm(track_xy[-1] - track_xy[0])
        step_lengths = np.linalg.norm(np.diff(track_xy, axis=0), axis=1)
        total_path = np.sum(step_lengths)

        if total_path < 1e-30:
            return 0.0
        return end_to_end / total_path

    @staticmethod
    def compute_confinement_ratio(track_xy, window=10):
        """Sliding-window confinement ratio.

        For each window of *window* frames, computes
        max_displacement / window_displacement.  Returns the mean.
        A value close to 1 indicates confinement; close to 0 indicates
        directed motion.
        """
        N = len(track_xy)
        if N < window + 1:
            return 0.0

        ratios = []
        for start in range(N - window):
            segment = track_xy[start:start + window + 1]
            max_disp = 0.0
            for i in range(len(segment)):
                for j in range(i + 1, len(segment)):
                    d = np.linalg.norm(segment[j] - segment[i])
                    if d > max_disp:
                        max_disp = d
            window_disp = np.linalg.norm(segment[-1] - segment[0])
            if max_disp > 1e-30:
                ratios.append(window_disp / max_disp)
            else:
                ratios.append(1.0)

        return np.mean(ratios) if ratios else 0.0


def z_stage_offset_correction(localizations, z_stage_positions, frame_to_zstage):
    """Correct Z positions for multi-Z-stage acquisition.
    
    Parameters
    ----------
    localizations : dict
        Localization data with 'z' positions
    z_stage_positions : dict
        Mapping of Z-stage index to absolute Z position
    frame_to_zstage : dict
        Mapping of frame number to Z-stage index
        
    Returns
    -------
    corrected : dict
        Z-corrected localizations
    """
    if 'z' not in localizations:
        return localizations
        
    corrected = localizations.copy()
    
    for frame in np.unique(localizations['frame']):
        if frame in frame_to_zstage:
            z_stage_idx = frame_to_zstage[frame]
            if z_stage_idx in z_stage_positions:
                z_offset = z_stage_positions[z_stage_idx]
                mask = localizations['frame'] == frame
                corrected['z'][mask] += z_offset
    
    return corrected
