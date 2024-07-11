#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 14:42:27 2024

@author: george
"""

import numpy as np
import logging
from typing import Optional, Tuple
from scipy.ndimage import gaussian_filter
from skimage import exposure
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

class VolumeProcessor:
    def __init__(self, data: np.ndarray):
        self.data = data

    def process_volume(self, slices_per_volume: int, frames_to_delete: int = 0) -> np.ndarray:
        """
        Process the volume data.

        Args:
            slices_per_volume: Number of slices in each volume.
            frames_to_delete: Number of frames to delete from the beginning.

        Returns:
            Processed numpy array.
        """
        try:
            reshaped_data = self._reshape_data(slices_per_volume)
            if frames_to_delete > 0:
                reshaped_data = self._delete_frames(reshaped_data, frames_to_delete)
            return reshaped_data
        except ValueError as e:
            logging.error(f"Error processing volume: {e}")
            raise

    def _reshape_data(self, slices_per_volume: int) -> np.ndarray:
        """
        Reshape the data into volumes.

        Args:
            slices_per_volume: Number of slices in each volume.

        Returns:
            Reshaped numpy array.
        """
        total_frames, x, y = self.data.shape
        volumes = total_frames // slices_per_volume
        return self.data[:volumes*slices_per_volume].reshape(volumes, slices_per_volume, x, y)

    def _delete_frames(self, data: np.ndarray, frames_to_delete: int) -> np.ndarray:
        return data[frames_to_delete:, :, :, :]

    def reshape_to_4d(self, data: np.ndarray, frames_per_vol: int) -> Optional[np.ndarray]:
        """
        Reshape 3D data to 4D.

        Args:
            data: Input 3D numpy array.
            frames_per_vol: Number of frames per volume.

        Returns:
            Reshaped 4D numpy array or None if reshaping fails.
        """
        try:
            total_frames, height, width = data.shape
            n_vols = total_frames // frames_per_vol

            if total_frames % frames_per_vol != 0:
                logging.warning(f"Total frames ({total_frames}) is not evenly divisible by frames per volume ({frames_per_vol}). Some frames will be discarded.")

            frames_to_use = n_vols * frames_per_vol
            reshaped_data = data[:frames_to_use].reshape(n_vols, frames_per_vol, height, width)

            logging.info(f"Reshaped data to shape: {reshaped_data.shape}")
            return reshaped_data
        except Exception as e:
            logging.error(f"Error in reshape_to_4d: {str(e)}")
            return None

    def delete_frames(self, data: np.ndarray, frames_to_delete: int) -> np.ndarray:
        """
        Delete frames from the beginning of each volume.

        Args:
            data: Input 4D numpy array.
            frames_to_delete: Number of frames to delete.

        Returns:
            Numpy array with frames deleted.
        """
        if frames_to_delete >= data.shape[1]:
            raise ValueError("frames_to_delete must be less than the number of frames per volume")
        return data[:, frames_to_delete:, :, :]

    def subtract_baseline(self, data: np.ndarray, baseline: float) -> np.ndarray:
        """
        Subtract baseline from the data.

        Args:
            data: Input numpy array.
            baseline: Baseline value to subtract.

        Returns:
            Numpy array with baseline subtracted.
        """
        return np.maximum(data - baseline, 0)  # Ensure non-negative values

    def calculate_df_f0(self, data: np.ndarray, f0_start: int, f0_end: int, vol_start: int, vol_end: int) -> np.ndarray:
        """
        Calculate delta F / F0.

        Args:
            data: Input 4D numpy array.
            f0_start, f0_end: Start and end indices for F0 calculation.
            vol_start, vol_end: Start and end indices for volume calculation.

        Returns:
            Numpy array of delta F / F0 values.
        """
        ratio_vol = data[:, f0_start:f0_end, :, :]
        ratio_vol_mean = np.mean(ratio_vol, axis=1, keepdims=True)
        vols_to_ratio = data[:, vol_start:vol_end, :, :]
        return np.divide(vols_to_ratio - ratio_vol_mean, ratio_vol_mean, out=np.zeros_like(vols_to_ratio), where=ratio_vol_mean!=0)


    def apply_gaussian_filter(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """
        Apply Gaussian filter to the data.

        Args:
            data: Input 4D numpy array.
            sigma: Standard deviation for Gaussian kernel.

        Returns:
            Filtered numpy array.
        """
        return gaussian_filter(data, sigma=(0, sigma, sigma, sigma))

    def get_volume_stats(self, data: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calculate statistics for each volume in the data.

        Args:
            data: Input 4D numpy array.

        Returns:
            Tuple of (mean, std, min, max) across all volumes.
        """
        mean = np.mean(data, axis=(1, 2, 3))
        std = np.std(data, axis=(1, 2, 3))
        min_val = np.min(data, axis=(1, 2, 3))
        max_val = np.max(data, axis=(1, 2, 3))
        return mean, std, min_val, max_val


    def multiply_by_factor(self, data, factor):
        return data * factor

    def convert_dtype(self, data, new_dtype):
        return data.astype(new_dtype)

    def average_volumes(self, data):
        return np.mean(data, axis=1)

    def safe_operation(self, operation: callable, *args, **kwargs) -> Optional[np.ndarray]:
        """
        Safely perform an operation on the data.

        Args:
            operation: Function to perform.
            *args, **kwargs: Arguments for the operation.

        Returns:
            Result of the operation or None if it fails.
        """
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in operation {operation.__name__}: {str(e)}")
            return None

    def is_data_valid(self, data):
        return data is not None and data.size > 0

    def normalize_data(self, data: np.ndarray, method: str = 'minmax') -> np.ndarray:
        """
        Normalize the data using specified method.

        Args:
            data: Input numpy array.
            method: Normalization method ('minmax' or 'zscore').

        Returns:
            Normalized numpy array.
        """
        if method == 'minmax':
            return (data - np.min(data)) / (np.max(data) - np.min(data))
        elif method == 'zscore':
            return (data - np.mean(data)) / np.std(data)
        else:
            raise ValueError("Unsupported normalization method. Use 'minmax' or 'zscore'.")



    def equalize_histogram(self, data: np.ndarray) -> np.ndarray:
        """
        Perform histogram equalization on the data.

        Args:
            data: Input numpy array.

        Returns:
            Histogram equalized numpy array.
        """
        return exposure.equalize_hist(data)

    def correct_motion(self, data: np.ndarray, reference_frame: int = 0) -> np.ndarray:
        """
        Perform simple motion correction using phase correlation.

        Args:
            data: Input 4D numpy array (t, z, y, x).
            reference_frame: Index of the reference frame.

        Returns:
            Motion corrected numpy array.
        """
        corrected_data = np.zeros_like(data)
        reference = data[reference_frame]

        for i in range(data.shape[0]):
            shift, _, _ = phase_cross_correlation(reference, data[i])
            corrected_data[i] = shift(data[i], shift, mode='constant', cval=0)

        return corrected_data

    def max_intensity_projection(self, data: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Create maximum intensity projection along specified axis.

        Args:
            data: Input 4D numpy array (t, z, y, x).
            axis: Axis along which to project (default is z-axis).

        Returns:
            3D numpy array of maximum intensity projections.
        """
        return np.max(data, axis=axis)

    def average_intensity_projection(self, data: np.ndarray, axis: int = 1) -> np.ndarray:
        """
        Create average intensity projection along specified axis.

        Args:
            data: Input 4D numpy array (t, z, y, x).
            axis: Axis along which to project (default is z-axis).

        Returns:
            3D numpy array of average intensity projections.
        """
        return np.mean(data, axis=axis)

    def apply_threshold(self, data: np.ndarray, threshold: float, mode: str = 'absolute') -> np.ndarray:
        """
        Apply threshold to the data.

        Args:
            data: Input numpy array.
            threshold: Threshold value.
            mode: 'absolute' or 'percentile'.

        Returns:
            Thresholded numpy array.
        """
        if mode == 'absolute':
            return np.where(data > threshold, data, 0)
        elif mode == 'percentile':
            return np.where(data > np.percentile(data, threshold), data, 0)
        else:
            raise ValueError("Unsupported threshold mode. Use 'absolute' or 'percentile'.")

    def crop_data(self, data: np.ndarray, x_range: Tuple[int, int], y_range: Tuple[int, int], z_range: Tuple[int, int]) -> np.ndarray:
        """
        Crop the data to specified ranges.

        Args:
            data: Input 4D numpy array (t, z, y, x).
            x_range: Tuple of (start, end) for x-axis.
            y_range: Tuple of (start, end) for y-axis.
            z_range: Tuple of (start, end) for z-axis.

        Returns:
            Cropped numpy array.
        """
        return data[:, z_range[0]:z_range[1], y_range[0]:y_range[1], x_range[0]:x_range[1]]

    def temporal_downsample(self, data: np.ndarray, factor: int) -> np.ndarray:
        """
        Downsample the data in the temporal dimension.

        Args:
            data: Input 4D numpy array (t, z, y, x).
            factor: Downsampling factor.

        Returns:
            Temporally downsampled numpy array.
        """
        return data[::factor]
