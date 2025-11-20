#!/usr/bin/env python3
"""
Python equivalent of detectSubResFeatures2D_StandAlone.m

DETECTSUBRESFEATURES2D_STANDALONE detects subresolution features in a series of images

Khuloud Jaqaman, September 2007
Converted to Python 2025

Copyright (C) 2025, Danuser Lab - UTSouthwestern

This file is part of u-track Python port.

u-track is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

u-track is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with u-track.  If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import scipy.io
from skimage import io
from scipy import stats
import matplotlib.pyplot as plt

# Import helper modules
try:
    # Try relative imports first (when run as module)
    from .image_processing import (
        filter_gauss_2d,
        spatial_move_ave_bg,
        locmax2d
    )
    from .fitting import (
        gauss_fit_nd,
        robust_mean
    )
    from .detection import (
        detect_sub_res_features_2d_v2,
        centroid_sub_res_features_2d
    )
    from .utils import (
        progress_text,
        create_distance_matrix,
        normcdf,
        robust_nanmean,
        robust_nanstd,
        validate_movie_info,
        save_movie_info_matlab,
        MovieInfoFrame
    )
except ImportError:
    # Fallback to direct imports (when run as script)
    from image_processing import (
        filter_gauss_2d,
        spatial_move_ave_bg,
        locmax2d
    )
    from fitting import (
        gauss_fit_nd,
        robust_mean
    )
    from detection import (
        detect_sub_res_features_2d_v2,
        centroid_sub_res_features_2d
    )
    from utils import (
        progress_text,
        create_distance_matrix,
        normcdf,
        robust_nanmean,
        robust_nanstd,
        validate_movie_info,
        save_movie_info_matlab,
        MovieInfoFrame
    )


@dataclass
class CandidateFeature:
    """Structure representing a candidate feature"""
    status: int = 1
    IBkg: float = 0.0
    Lmax: List[float] = field(default_factory=list)  # [x, y] position
    amp: float = 0.0
    pValue: float = 0.0


@dataclass
class LocalMaximaFrame:
    """Structure for local maxima in a single frame"""
    cands: List[CandidateFeature] = field(default_factory=list)


@dataclass
class MovieInfo:
    """Structure containing detected feature information for one frame"""
    xCoord: np.ndarray = field(default_factory=lambda: np.array([]))
    yCoord: np.ndarray = field(default_factory=lambda: np.array([]))
    amp: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class Exceptions:
    """Structure containing exception information"""
    emptyFrames: List[int] = field(default_factory=list)
    framesFailedLocMax: List[int] = field(default_factory=list)
    framesFailedMMF: List[int] = field(default_factory=list)


@dataclass
class Background:
    """Structure containing background information"""
    meanRawLast5: float = 0.0
    stdRawLast5: float = 0.0
    meanIntegFLast1: float = 0.0
    stdIntegFLast1: float = 0.0
    meanIntegFFirst1: float = 0.0
    stdIntegFFirst1: float = 0.0

def setup_detection_logging(verbose=True):
    """Setup logging for detection pipeline."""
    import logging

    if verbose:
        # Set up logging to show important messages only
        logging.basicConfig(
            level=logging.INFO,
            format='%(levelname)s: %(message)s'
        )
    else:
        # Suppress most debug output
        logging.basicConfig(level=logging.WARNING)

    # Suppress matplotlib debug messages
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

def detect_sub_res_features_2d_standalone(
    movie_param: 'MovieParam',
    detection_param: 'DetectionParam',
    save_results: Optional['SaveResults'] = None,
    verbose: bool = True
) -> Tuple[List[MovieInfo], Exceptions, List[LocalMaximaFrame], Background, float]:
    """
    Detect subresolution features in a series of 2D images.

    Args:
        movie_param: Structure with movie parameters
        detection_param: Structure with detection parameters
        save_results: Structure with save parameters or None
        verbose: Whether to show progress

    Returns:
        Tuple containing:
        - movie_info: List of MovieInfo structures for each frame
        - exceptions: Exception information
        - local_maxima: List of local maxima for each frame
        - background: Background information
        - psf_sigma: Estimated PSF sigma
    """

    # Initialize output variables
    movie_info = []
    exceptions = Exceptions()
    local_maxima = []
    background = Background()
    psf_sigma = detection_param.psf_sigma

    # Input validation
    if movie_param is None or detection_param is None:
        print('--detectSubResFeatures2D_StandAlone: Incorrect number of input arguments!')
        return movie_info, exceptions, local_maxima, background, psf_sigma

    # Get movie parameters
    has_image_dir = (hasattr(movie_param, 'image_dir') and
                    movie_param.image_dir is not None and
                    movie_param.image_dir != "")
    if has_image_dir:
        image_dir = movie_param.image_dir
        filename_base = movie_param.filename_base
        digits_4_enum = movie_param.digits_4_enum
        channel = None
    else:
        channel = movie_param.channel
        image_dir = None
        filename_base = None
        digits_4_enum = None

    first_image_num = movie_param.first_image_num
    last_image_num = movie_param.last_image_num

    # Get initial guess of PSF sigma
    psf_sigma = detection_param.psf_sigma

    # Get position calculation method
    calc_method = getattr(detection_param, 'calc_method', 'g')

    # Get statistical test alpha values
    if not hasattr(detection_param, 'test_alpha') or detection_param.test_alpha is None:
        test_alpha = {
            'alphaR': 0.05,
            'alphaA': 0.05,
            'alphaD': 0.05,
            'alphaF': 0.0
        }
    else:
        if isinstance(detection_param.test_alpha, dict):
            test_alpha = detection_param.test_alpha
        else:
            test_alpha = detection_param.test_alpha.__dict__

    # Get visualization option
    visual = getattr(detection_param, 'visual', False)

    # Check whether to do MMF
    do_mmf = getattr(detection_param, 'do_mmf', True)

    # Get camera bit depth
    bit_depth = getattr(detection_param, 'bit_depth', 16)

    # Get alpha-value for local maxima detection
    alpha_loc_max = getattr(detection_param, 'alpha_loc_max', 0.05)
    if not isinstance(alpha_loc_max, (list, np.ndarray)):
        alpha_loc_max = [alpha_loc_max]
    num_alpha_loc_max = len(alpha_loc_max)

    # Check whether to estimate PSF sigma from the data
    num_sigma_iter = getattr(detection_param, 'num_sigma_iter', 10)

    # Get integration time window
    integ_window = getattr(detection_param, 'integ_window', 0)
    if not isinstance(integ_window, (list, np.ndarray)):
        integ_window = [integ_window]
    num_integ_window = len(integ_window)

    # Make sure that alpha_loc_max is the same size as integ_window
    if num_integ_window > num_alpha_loc_max:
        alpha_loc_max.extend([alpha_loc_max[0]] * (num_integ_window - num_alpha_loc_max))

    # Get background information if supplied
    abs_bg = hasattr(detection_param, 'background') and detection_param.background is not None
    if abs_bg:
        bg_image_dir = detection_param.background.image_dir
        bg_image_base = detection_param.background.filename_base
        alpha_loc_max_abs = detection_param.background.alpha_loc_max_abs

    # Get mask information if any
    mask_flag = False
    mask_image = None
    if hasattr(detection_param, 'roi_mask') and detection_param.roi_mask is not None:
        mask_flag = True
        mask_image = detection_param.roi_mask
        if len(mask_image.shape) == 3:
            mask_image = mask_image[:, :, 0]  # Assume ROI remains constant
    elif hasattr(detection_param, 'mask_loc') and detection_param.mask_loc:
        mask_flag = True
        mask_image = io.imread(detection_param.mask_loc).astype(float)

    # Determine where to save results
    if save_results is None:
        save_res_dir = os.getcwd()
        save_res_file = 'detectedFeatures.mat'
    elif isinstance(save_results, dict) or hasattr(save_results, '__dict__'):
        save_res_dir = getattr(save_results, 'dir', os.getcwd())
        save_res_file = getattr(save_results, 'filename', 'detectedFeatures.mat')
    else:
        save_results = None

    # Create enumeration strings for image filenames
    if has_image_dir:
        enum_strings = []
        format_string = f"{{:0{digits_4_enum}d}}"
        for i in range(1, last_image_num + 1):
            enum_strings.append(format_string.format(i))

    # Initialize some variables
    empty_frames = []
    frames_failed_loc_max = []
    frames_failed_mmf = []

    # Turn warnings off
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Get image indices and number of images
        image_indices = list(range(first_image_num, last_image_num + 1))
        num_images_raw = last_image_num - first_image_num + 1
        num_images_integ = [num_images_raw - 2 * iw for iw in integ_window]

        # Read first image and get image size
        if has_image_dir:
            first_image_path = os.path.join(
                image_dir,
                f"{filename_base}{enum_strings[image_indices[0] - 1]}.tif"
            )
            if os.path.exists(first_image_path):
                image_tmp = io.imread(first_image_path)
            else:
                print('First image does not exist! Exiting ...')
                return movie_info, exceptions, local_maxima, background, psf_sigma
            image_size_y, image_size_x = image_tmp.shape  # height, width = rows, columns
        else:
            # Using channel (MultiPaneTiffChannel)
            image_size_y, image_size_x = channel.imSize_  # Note: imSize_ is [height, width]

        # Make mask of ones if no mask is supplied
        if not mask_flag:
            mask_image = np.ones((image_size_y, image_size_x))

        # Check which images exist
        image_exists = np.ones(num_images_raw, dtype=bool)
        if has_image_dir:
            for i in range(num_images_raw):
                image_path = os.path.join(
                    image_dir,
                    f"{filename_base}{enum_strings[image_indices[i] - 1]}.tif"
                )
                if not os.path.exists(image_path):
                    image_exists[i] = False

        # Calculate background properties at movie end
        last_5_start = max(num_images_raw - 4, 1)
        image_last_5 = np.full((image_size_y, image_size_x, 5), np.nan)

        for i, image_idx in enumerate(range(last_5_start, num_images_raw + 1)):
            if image_exists[image_idx - 1]:
                if has_image_dir:
                    image_path = os.path.join(
                        image_dir,
                        f"{filename_base}{enum_strings[image_indices[image_idx - 1] - 1]}.tif"
                    )
                    image_last_5[:, :, i] = io.imread(image_path).astype(float)
                else:
                    image_last_5[:, :, i] = channel.loadImage(image_indices[image_idx - 1]).astype(float)

        # Apply mask
        image_last_5 = image_last_5 * mask_image[:, :, np.newaxis]

        # Normalize images
        image_last_5 = image_last_5 / (2**bit_depth - 1)
        image_last_5[image_last_5 == 0] = np.nan

        # Calculate background statistics using enhanced functions
        bg_mean_raw, bg_std_raw = spatial_move_ave_bg(image_last_5, image_size_y, image_size_x)

        # Get size of absolute background images if supplied
        if abs_bg:
            bg_image_path = os.path.join(
                bg_image_dir,
                f"{bg_image_base}{enum_strings[image_indices[0] - 1]}.tif"
            )
            image_tmp = io.imread(bg_image_path)
            bg_size_x, bg_size_y = image_tmp.shape

        # LOCAL MAXIMA DETECTION
        # Initialize output structure
        local_maxima = [LocalMaximaFrame() for _ in range(num_images_raw)]

        for i_window in range(num_integ_window):

            if verbose:
                progress_text(0, f'Detecting local maxima with integration window = {integ_window[i_window]}')

            for i_image in range(num_images_integ[i_window]):

                # Store raw images in array (keep original values)
                window_size = 1 + 2 * integ_window[i_window]
                image_raw = np.full((image_size_y, image_size_x, window_size), np.nan)

                for j_image in range(window_size):
                    img_idx = j_image + i_image
                    if has_image_dir and image_exists[img_idx]:
                        image_path = os.path.join(
                            image_dir,
                            f"{filename_base}{enum_strings[image_indices[img_idx] - 1]}.tif"
                        )
                        image_raw[:, :, j_image] = io.imread(image_path).astype(float)
                    elif not has_image_dir:
                        image_raw[:, :, j_image] = channel.loadImage(image_indices[img_idx]).astype(float)

                # Apply mask
                image_raw = image_raw * mask_image[:, :, np.newaxis]

                # Replace zeros with NaNs
                image_raw[image_raw == 0] = np.nan

                # Store original raw images BEFORE normalization for amplitude extraction
                image_raw_orig = image_raw.copy()

                # Normalize images for filtering and processing
                image_raw = image_raw / (2**bit_depth - 1)

                # Integrate images (normalized for processing)
                image_integ = robust_nanmean(image_raw, axis=2)

                # Integrate original images (for meaningful amplitudes)
                image_integ_orig = robust_nanmean(image_raw_orig, axis=2)

                # Filter integrated image (use lighter filtering)
                image_integ_f = filter_gauss_2d(image_integ, 0.5)  # Much lighter filtering

                # Get integrated image background noise statistics (use original values)
                try:
                    bg_mean_integ, bg_std_integ = spatial_move_ave_bg(
                        image_integ_orig, image_size_y, image_size_x
                    )
                    # Debug background stats
                    if verbose and i_image == 0:
                        print(f"  Debug: Background estimation - mean: {np.mean(bg_mean_integ):.3f}, std: {np.mean(bg_std_integ):.3f}")
                except Exception as e:
                    if verbose and i_image == 0:
                        print(f"  Debug: Background estimation failed: {e}")
                    # Fallback: use simple background estimation on original values
                    bg_mean_integ = np.full_like(image_integ, robust_nanmean(image_integ_orig))
                    bg_std_integ = np.full_like(image_integ, robust_nanstd(image_integ_orig))
                    if verbose and i_image == 0:
                        print(f"  Debug: Using fallback background - mean: {np.mean(bg_mean_integ):.3f}, std: {np.mean(bg_std_integ):.3f}")

                # Calculate absolute background mean and std if supplied
                if abs_bg:
                    bg_raw = np.full((bg_size_x, bg_size_y, window_size), np.nan)
                    for j_image in range(window_size):
                        img_idx = j_image + i_image
                        if image_exists[img_idx]:
                            bg_path = os.path.join(
                                bg_image_dir,
                                f"{bg_image_base}{enum_strings[image_indices[img_idx] - 1]}.tif"
                            )
                            bg_raw[:, :, j_image] = io.imread(bg_path).astype(float)

                    bg_raw[bg_raw == 0] = np.nan
                    bg_raw = bg_raw / (2**bit_depth - 1)
                    bg_integ = robust_nanmean(bg_raw, axis=2)
                    bg_abs_mean_integ = np.full((image_size_y, image_size_x), robust_nanmean(bg_integ))
                    bg_abs_std_integ = np.full((image_size_y, image_size_x), robust_nanstd(bg_integ))

                try:
                    # Call locmax2d to get local maxima in filtered image
                    # Use enhanced locmax2d function
                    f_img = locmax2d(image_integ_f, mask_size=5, threshold=1)

                    # Get positions and amplitudes of local maxima
                    local_max_pos_y, local_max_pos_x = np.where(f_img)
                    # Extract amplitudes from ORIGINAL (non-normalized) image for meaningful values
                    local_max_amp = image_integ_orig[local_max_pos_y, local_max_pos_x]
                    local_max_1d_idx = np.where(f_img.ravel())[0]

                    # Get background values corresponding to local maxima
                    bg_mean_max_f = bg_mean_integ.ravel()[local_max_1d_idx]
                    bg_std_max_f = bg_std_integ.ravel()[local_max_1d_idx]
                    # Ensure std is not zero (add small value to prevent division by zero)
                    bg_std_max_f = np.maximum(bg_std_max_f, 1e-6)
                    bg_mean_max = bg_mean_raw.ravel()[local_max_1d_idx]

                    # Calculate p-values for local maxima amplitudes using enhanced normcdf
                    p_value = 1 - normcdf(local_max_amp, bg_mean_max_f, bg_std_max_f)

                    # Debug information
                    if verbose and i_image == 0:  # Only print for first image
                        print(f"  Debug: Found {len(local_max_amp)} raw local maxima")
                        if len(local_max_amp) > 0:
                            print(f"  Debug: Amplitude range: {np.min(local_max_amp):.2f} to {np.max(local_max_amp):.2f}")
                            print(f"  Debug: Background mean: {np.mean(bg_mean_max_f):.2f}")
                            print(f"  Debug: Background std: {np.mean(bg_std_max_f):.2f}")
                            print(f"  Debug: P-value range: {np.min(p_value):.4f} to {np.max(p_value):.4f}")
                            print(f"  Debug: Using alpha_loc_max = {alpha_loc_max[i_window]}")

                    if abs_bg:
                        bg_abs_mean_max_f = bg_abs_mean_integ.ravel()[local_max_1d_idx]
                        bg_abs_std_max_f = bg_abs_std_integ.ravel()[local_max_1d_idx]
                        p_value_abs = 1 - normcdf(
                            local_max_amp, bg_abs_mean_max_f, bg_abs_std_max_f
                        )

                    # Retain only maxima with significant amplitude
                    if abs_bg:
                        keep_max = np.where(
                            (p_value < alpha_loc_max[i_window]) &
                            (p_value_abs < alpha_loc_max_abs)
                        )[0]
                    else:
                        keep_max = np.where(p_value < alpha_loc_max[i_window])[0]

                    # Debug information
                    if verbose and i_image == 0:
                        print(f"  Debug: Kept {len(keep_max)} maxima after p-value filter")

                    local_max_pos_x = local_max_pos_x[keep_max]
                    local_max_pos_y = local_max_pos_y[keep_max]
                    local_max_amp = local_max_amp[keep_max]
                    bg_mean_max = bg_mean_max[keep_max]
                    p_value = p_value[keep_max]
                    num_local_max = len(keep_max)

                    # Construct candidates structure
                    if num_local_max == 0:
                        cands = []
                    else:
                        cands = []
                        for i_max in range(num_local_max):
                            cand = CandidateFeature(
                                status=1,
                                IBkg=bg_mean_max[i_max],
                                Lmax=[local_max_pos_x[i_max], local_max_pos_y[i_max]],
                                amp=local_max_amp[i_max],
                                pValue=p_value[i_max]
                            )
                            cands.append(cand)

                    # Add candidates to the appropriate frame
                    frame_idx = i_image + integ_window[i_window]
                    local_maxima[frame_idx].cands.extend(cands)

                except Exception:
                    # If local maxima detection fails, continue
                    pass

                # Display progress
                if verbose:
                    progress_text(
                        (i_image + 1) / num_images_integ[i_window],
                        f'Detecting local maxima with integration window = {integ_window[i_window]}'
                    )

            # Assign local maxima for frames left out due to time integration
            for i_image in range(integ_window[i_window]):
                local_maxima[i_image].cands.extend(
                    local_maxima[integ_window[i_window]].cands
                )

            for i_image in range(num_images_raw - integ_window[i_window], num_images_raw):
                local_maxima[i_image].cands.extend(
                    local_maxima[num_images_raw - integ_window[i_window] - 1].cands
                )

        # Delete local maxima found in non-existent frames
        for i_frame in range(num_images_raw):
            if not image_exists[i_frame]:
                local_maxima[i_frame].cands = []

        # Remove redundant candidates and register empty frames
        if verbose:
            progress_text(0, 'Removing redundant local maxima')

        for i_image in range(num_images_raw):
            cands_current = local_maxima[i_image].cands

            if verbose and i_image < 5:  # Debug first 5 frames
                print(f"  Debug frame {i_image + 1}: {len(cands_current)} candidates before redundancy removal")

            if not cands_current:
                empty_frames.append(i_image + 1)  # 1-indexed like MATLAB
            else:
                # Get unique local maxima positions
                max_pos = np.array([cand.Lmax for cand in cands_current])

                if verbose and i_image < 5:
                    print(f"  Debug frame {i_image + 1}: positions shape = {max_pos.shape}")

                try:
                    _, unique_idx = np.unique(max_pos, axis=0, return_index=True)
                    cands_current = [cands_current[i] for i in unique_idx]
                    local_maxima[i_image].cands = cands_current

                    if verbose and i_image < 5:
                        print(f"  Debug frame {i_image + 1}: {len(cands_current)} candidates after redundancy removal")

                    # Check if frame became empty after redundancy removal
                    if not cands_current:
                        empty_frames.append(i_image + 1)

                except Exception as e:
                    if verbose and i_image < 5:
                        print(f"  Debug frame {i_image + 1}: Error in unique removal: {e}")
                    # Keep original candidates if unique fails
                    local_maxima[i_image].cands = cands_current

            if verbose:
                progress_text((i_image + 1) / num_images_raw, 'Removing redundant local maxima')

        # Make list of images that have local maxima
        good_images = [i for i in range(num_images_raw) if i + 1 not in empty_frames]
        num_good_images = len(good_images)

        if verbose:
            print(f"Empty frames: {empty_frames}")
            print(f"Good images: {good_images}")
            print(f"Number of good images: {num_good_images}")

        # PSF SIGMA ESTIMATION (enhanced with better handling)
        if num_sigma_iter > 0:
            fit_parameters = ['X1', 'X2', 'A', 'Sxy', 'B']
            psf_sigma_in = psf_sigma
            psf_sigma_0 = psf_sigma  # Initialize to current value instead of 0
            accept_calc = True
            num_iter = 0

            while (num_iter <= num_sigma_iter and accept_calc and
                   (num_iter == 0 or abs(psf_sigma - psf_sigma_0) / max(psf_sigma_0, 1e-10) > 0.05)):

                num_iter += 1
                psf_sigma_0 = psf_sigma
                sigma_estimates = []

                psf_sigma_5 = int(np.ceil(5 * psf_sigma_0))

                if verbose:
                    if num_iter == 1:
                        progress_text(0, 'Estimating PSF sigma')
                    else:
                        progress_text(0, 'Repeating PSF sigma estimation')

                # Use first 50 good images for sigma estimation
                images_2_use = good_images[:min(50, num_good_images)]
                images_2_use = [img for img in images_2_use if img >= integ_window[0]]

                for img_idx, i_image in enumerate(images_2_use):

                    # Read raw image
                    if has_image_dir:
                        image_path = os.path.join(
                            image_dir,
                            f"{filename_base}{enum_strings[image_indices[i_image] - 1]}.tif"
                        )
                        image_raw_single = io.imread(image_path).astype(float)
                    else:
                        image_raw_single = channel.loadImage(image_indices[i_image]).astype(float)

                    image_raw_single = image_raw_single / (2**bit_depth - 1)
                    image_raw_single = image_raw_single * mask_image

                    # Get feature information
                    cands = local_maxima[i_image].cands
                    if not cands:
                        continue

                    feat_pos = np.array([cand.Lmax for cand in cands])
                    feat_amp = np.array([cand.amp for cand in cands])
                    feat_bg = np.array([cand.IBkg for cand in cands])
                    feat_pv = np.array([cand.pValue for cand in cands])

                    # Retain features away from boundaries
                    valid_features = np.where(
                        (feat_pos[:, 0] > psf_sigma_5) &
                        (feat_pos[:, 0] < image_size_y - psf_sigma_5) &
                        (feat_pos[:, 1] > psf_sigma_5) &
                        (feat_pos[:, 1] < image_size_x - psf_sigma_5)
                    )[0]

                    if len(valid_features) == 0:
                        continue

                    feat_pos = feat_pos[valid_features]
                    feat_amp = feat_amp[valid_features]
                    feat_bg = feat_bg[valid_features]
                    feat_pv = feat_pv[valid_features]

                    # Find isolated features if more than one
                    if len(valid_features) > 1:
                        dist_matrix = create_distance_matrix(feat_pos, feat_pos)
                        np.fill_diagonal(dist_matrix, np.inf)
                        nn_dist = np.min(dist_matrix, axis=1)
                        isolated_features = np.where(nn_dist > 10 * psf_sigma_0)[0]

                        if len(isolated_features) == 0:
                            continue

                        feat_pos = feat_pos[isolated_features]
                        feat_amp = feat_amp[isolated_features]
                        feat_bg = feat_bg[isolated_features]
                        feat_pv = feat_pv[isolated_features]

                        # Use features with p-values between 25th and 75th percentiles
                        percentile_25 = np.percentile(feat_pv, 25)
                        percentile_75 = np.percentile(feat_pv, 75)
                        good_pvalue = np.where(
                            (feat_pv > percentile_25) & (feat_pv < percentile_75)
                        )[0]

                        if len(good_pvalue) == 0:
                            continue

                        feat_pos = feat_pos[good_pvalue]
                        feat_amp = feat_amp[good_pvalue]
                        feat_bg = feat_bg[good_pvalue]

                    # Estimate sigma for each selected feature
                    for i_feat in range(len(feat_pos)):
                        lower_bound = feat_pos[i_feat].astype(int) - psf_sigma_5
                        upper_bound = feat_pos[i_feat].astype(int) + psf_sigma_5

                        # Ensure bounds are within image
                        lower_bound = np.maximum(lower_bound, [0, 0])
                        upper_bound = np.minimum(upper_bound, [image_size_y-1, image_size_x-1])

                        image_cropped = image_raw_single[
                            lower_bound[0]:upper_bound[0]+1,
                            lower_bound[1]:upper_bound[1]+1
                        ]

                        if not np.any(np.isnan(image_cropped)):
                            init_guess = [
                                psf_sigma_5 + 1,  # X1
                                psf_sigma_5 + 1,  # X2
                                feat_amp[i_feat],  # A
                                psf_sigma_0,      # Sxy
                                feat_bg[i_feat]   # B
                            ]

                            try:
                                parameters = gauss_fit_nd(
                                    image_cropped, None, fit_parameters, init_guess
                                )
                                if parameters is not None and len(parameters) > 3:
                                    sigma_estimates.append(parameters[3])
                            except Exception:
                                continue

                    if verbose:
                        if num_iter == 1:
                            progress_text(
                                (img_idx + 1) / len(images_2_use),
                                'Estimating PSF sigma'
                            )
                        else:
                            progress_text(
                                (img_idx + 1) / len(images_2_use),
                                'Repeating PSF sigma estimation'
                            )

                # Estimate PSF sigma as robust mean
                sigma_estimates = [s for s in sigma_estimates if not np.isnan(s)]
                num_calcs = len(sigma_estimates)

                if num_calcs > 0:
                    psf_sigma_new, _, inlier_idx = robust_mean(
                        np.array(sigma_estimates), None, 3, 0, True
                    )
                    num_inlier_idx = np.sum(inlier_idx)

                    # Accept new sigma based on criteria
                    accept_calc = (
                        (num_calcs >= 100 and num_inlier_idx >= 0.7 * num_calcs) or
                        (num_calcs >= 50 and num_inlier_idx >= 0.9 * num_calcs) or
                        (num_calcs >= 10 and num_inlier_idx == num_calcs)
                    )

                    if accept_calc:
                        psf_sigma = psf_sigma_new
                        print(f'PSF sigma = {psf_sigma:.3f} ({num_inlier_idx} inliers out of {num_calcs} observations)')
                    else:
                        psf_sigma = psf_sigma_in
                        print('Not enough observations to change PSF sigma, using input PSF sigma')
                else:
                    accept_calc = False
                    psf_sigma = psf_sigma_in

            # Check convergence
            if (num_iter == num_sigma_iter + 1 and accept_calc and
                psf_sigma_0 > 0 and abs(psf_sigma - psf_sigma_0) / psf_sigma_0 > 0.05):
                psf_sigma = psf_sigma_in
                print('Estimation terminated (no convergence), using input PSF sigma')

        # MIXTURE-MODEL FITTING
        movie_info = [MovieInfo() for _ in range(num_images_raw)]

        if verbose:
            if calc_method == 'g':
                progress_text(0, 'Mixture-model fitting')
            elif calc_method == 'gv':
                progress_text(0, 'Mixture-model fitting with variable sigma')
            else:
                progress_text(0, 'Centroid calculation')

        for img_idx, i_image in enumerate(good_images):
            if verbose:
                print(f"  Debug: Processing frame {i_image + 1} (good image {img_idx + 1}/{len(good_images)})")

            # Read raw image
            if has_image_dir:
                image_path = os.path.join(
                    image_dir,
                    f"{filename_base}{enum_strings[image_indices[i_image] - 1]}.tif"
                )
                image_raw_single = io.imread(image_path).astype(float)
            else:
                image_raw_single = channel.loadImage(image_indices[i_image]).astype(float)

            image_raw_single = image_raw_single / (2**bit_depth - 1)
            image_raw_single = image_raw_single * mask_image

            try:
                # Debug: Check candidates before MMF
                cands = local_maxima[i_image].cands
                if verbose:
                    if cands:
                        print(f"  Debug: Sending {len(cands)} candidates to MMF")
                        print(f"  Debug: Sample candidate - pos: {cands[0].Lmax}, amp: {cands[0].amp:.2f}, bg: {cands[0].IBkg:.2f}")
                    else:
                        print("  Debug: No candidates for MMF")

                # Fit with mixture models or calculate centroids
                # Use original (non-normalized) image for meaningful amplitudes
                image_raw_original = image_raw_single * (2**bit_depth - 1)  # Convert back to original range

                if verbose:
                    print(f"  Debug: MMF image range: {np.nanmin(image_raw_original):.1f} to {np.nanmax(image_raw_original):.1f}")

                if calc_method == 'g':
                    if verbose:
                        print(f"  Debug: Calling detect_sub_res_features_2d_v2...")
                    features_info = detect_sub_res_features_2d_v2(
                        image_raw_original,  # Use original values
                        local_maxima[i_image].cands,
                        psf_sigma,
                        test_alpha,
                        visual,
                        do_mmf,
                        bit_depth,  # Pass actual bit depth
                        False,
                        np.mean(bg_std_raw) * (2**bit_depth - 1)  # Scale background std back too
                    )
                elif calc_method == 'gv':
                    if verbose:
                        print(f"  Debug: Calling detect_sub_res_features_2d_v2 with variable sigma...")
                    features_info = detect_sub_res_features_2d_v2(
                        image_raw_original,  # Use original values
                        local_maxima[i_image].cands,
                        psf_sigma,
                        test_alpha,
                        visual,
                        do_mmf,
                        bit_depth,  # Pass actual bit depth
                        False,
                        np.mean(bg_std_raw) * (2**bit_depth - 1),  # Scale background std back too
                        True  # Variable sigma
                    )
                else:
                    if verbose:
                        print(f"  Debug: Calling centroid_sub_res_features_2d...")
                    features_info = centroid_sub_res_features_2d(
                        image_raw_original,  # Use original values
                        local_maxima[i_image].cands,
                        psf_sigma,
                        visual,
                        bit_depth,
                        False
                    )

                if verbose:
                    print(f"  Debug: MMF function returned, checking results...")
                    if hasattr(features_info, 'xCoord') and features_info.xCoord is not None:
                        print(f"  Debug: MMF returned {len(features_info.xCoord)} features")
                        if len(features_info.xCoord) > 0:
                            # With this safe version:
                            if hasattr(features_info, 'xCoord') and len(features_info.xCoord) > 0:
                                x_val = features_info.xCoord[0]
                                y_val = features_info.yCoord[0]
                                print(f"  Debug: First feature - x:{x_val}, y:{y_val}")
                    else:
                        print("  Debug: MMF returned no features (None or empty)")
                        print(f"  Debug: features_info type: {type(features_info)}")
                        print(f"  Debug: features_info attributes: {dir(features_info)}")

                # Save results
                movie_info[i_image] = features_info

                # Check if frame is empty
                if hasattr(features_info, 'xCoord') and len(features_info.xCoord) == 0:
                    empty_frames.append(i_image + 1)

            except Exception as e:
                if verbose:
                    print(f"  Debug: MMF failed with error: {e}")
                    import traceback
                    traceback.print_exc()
                # If detection fails
                empty_frames.append(i_image + 1)
                frames_failed_mmf.append(i_image + 1)

            # Display progress
            if verbose:
                if calc_method == 'g':
                    progress_text((img_idx + 1) / len(good_images), 'Mixture-model fitting')
                elif calc_method == 'gv':
                    progress_text((img_idx + 1) / len(good_images), 'Mixture-model fitting with variable sigma')
                else:
                    progress_text((img_idx + 1) / len(good_images), 'Centroid calculation')

        # POST-PROCESSING

        # Sort list of empty frames
        empty_frames = sorted(list(set(empty_frames)))

        # Store exceptions
        exceptions = Exceptions(
            emptyFrames=empty_frames,
            framesFailedLocMax=frames_failed_loc_max,
            framesFailedMMF=frames_failed_mmf
        )

        # Adjust movie_info indexing to match MATLAB (1-indexed)
        movie_info_adjusted = [MovieInfo() for _ in range(last_image_num)]
        for i, info in enumerate(movie_info):
            if first_image_num + i - 1 < len(movie_info_adjusted):
                movie_info_adjusted[first_image_num + i - 1] = info

        # Validate movie info structure
        validation_issues = validate_movie_info(movie_info_adjusted)
        if validation_issues and verbose:
            print("Movie info validation issues:")
            for issue in validation_issues[:5]:  # Show first 5 issues
                print(f"  - {issue}")

        # Save results if requested
        if save_results is not None:
            save_path = os.path.join(save_res_dir, save_res_file)

            # Use enhanced save function
            additional_data = {
                'movieParam': movie_param.__dict__ if hasattr(movie_param, '__dict__') else str(movie_param),
                'detectionParam': detection_param.__dict__ if hasattr(detection_param, '__dict__') else str(detection_param),
                'exceptions': exceptions.__dict__,
                'localMaxima': [{'cands': [c.__dict__ for c in lm.cands]} for lm in local_maxima],
                'background': background.__dict__,
                'psfSigma': psf_sigma
            }

            try:
                save_movie_info_matlab(movie_info_adjusted, save_path, additional_data)
                print(f"Results saved to: {save_path}")
            except Exception as e:
                print(f"Warning: Could not save results to {save_path}: {e}")
                print("Detection completed successfully but results not saved.")

    return movie_info_adjusted, exceptions, local_maxima, background, psf_sigma
