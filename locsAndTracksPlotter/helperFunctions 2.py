#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper Functions for FLIKA Tracking Plugin

This module contains utility functions for data manipulation, mathematical operations,
and curve fitting used throughout the tracking analysis pipeline.

Created on Fri Jun  2 14:48:13 2023
@author: george
"""

import logging
from typing import List, Union, Dict, Any
import numpy as np
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)


def dictFromList(lst: List[Any]) -> Dict[str, str]:
    """
    Create a dictionary from a list where keys and values are the same.

    This is commonly used for creating ComboBox items in PyQt applications.

    Args:
        lst: List of items to convert to dictionary

    Returns:
        Dictionary with string keys and values

    Example:
        >>> dictFromList([1, 2, 3])
        {'1': '1', '2': '2', '3': '3'}
    """
    try:
        return {str(x): str(x) for x in lst}
    except Exception as e:
        logger.error(f"Error creating dictionary from list: {e}")
        return {}


def exp_dec(x: np.ndarray, A1: float, tau: float) -> np.ndarray:
    """
    Single exponential decay function for curve fitting.

    Formula: 1 + A1 * exp(-x / tau)

    Args:
        x: Independent variable (time points)
        A1: Amplitude parameter
        tau: Time constant parameter

    Returns:
        Exponential decay values
    """
    try:
        return 1 + A1 * np.exp(-x / tau)
    except Exception as e:
        logger.error(f"Error in exponential decay calculation: {e}")
        return np.zeros_like(x)


def exp_dec_2(x: np.ndarray, A1: float, tau1: float, tau2: float) -> np.ndarray:
    """
    Double exponential decay function for curve fitting.

    Formula: 1 + A1 * exp(-x / tau1) + A2 * exp(-x / tau2)
    where A2 = -1 - A1

    Args:
        x: Independent variable (time points)
        A1: First amplitude parameter
        tau1: First time constant
        tau2: Second time constant

    Returns:
        Double exponential decay values
    """
    try:
        A2 = -1 - A1
        return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)
    except Exception as e:
        logger.error(f"Error in double exponential decay calculation: {e}")
        return np.zeros_like(x)


def exp_dec_3(x: np.ndarray, A1: float, A2: float, tau1: float,
              tau2: float, tau3: float) -> np.ndarray:
    """
    Triple exponential decay function for curve fitting.

    Formula: 1 + A1 * exp(-x / tau1) + A2 * exp(-x / tau2) + A3 * exp(-x / tau3)
    where A3 = -1 - A1 - A2

    Args:
        x: Independent variable (time points)
        A1: First amplitude parameter
        A2: Second amplitude parameter
        tau1: First time constant
        tau2: Second time constant
        tau3: Third time constant

    Returns:
        Triple exponential decay values
    """
    try:
        A3 = -1 - A1 - A2
        return (1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) +
                A3 * np.exp(-x / tau3))
    except Exception as e:
        logger.error(f"Error in triple exponential decay calculation: {e}")
        return np.zeros_like(x)


def rollingFunc(arr: Union[List, np.ndarray], window_size: int = 6,
                func_type: str = 'mean') -> List[float]:
    """
    Apply a rolling window function to an array.

    Args:
        arr: Input array
        window_size: Size of the rolling window (default: 6)
        func_type: Type of function to apply ('mean', 'std', 'variance')

    Returns:
        List of rolling function values

    Raises:
        ValueError: If func_type is not supported
    """
    try:
        logger.debug(f"Applying rolling {func_type} with window size {window_size}")

        series = pd.Series(arr)
        windows = series.rolling(window_size)

        if func_type == 'mean':
            moving_averages = windows.mean()
        elif func_type == 'std':
            moving_averages = windows.std()
        elif func_type == 'variance':
            moving_averages = windows.var()
        else:
            raise ValueError(f"Invalid func_type: {func_type}. "
                           "Must be 'mean', 'std', or 'variance'.")

        # Return values excluding NaN entries from the beginning
        final_list = moving_averages[window_size - 1:].tolist()

        logger.debug(f"Rolling function completed. Output length: {len(final_list)}")
        return final_list

    except Exception as e:
        logger.error(f"Error in rolling function calculation: {e}")
        return []


def gammaCorrect(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to an image.

    Gamma correction adjusts the brightness and contrast of an image
    using a power-law transformation.

    Args:
        img: Input image array
        gamma: Gamma value for correction (>1 darkens, <1 brightens)

    Returns:
        Gamma-corrected image array

    Raises:
        ValueError: If gamma is <= 0
    """
    try:
        if gamma <= 0:
            raise ValueError("Gamma value must be positive")

        logger.debug(f"Applying gamma correction with gamma={gamma}")

        gammaCorrection = 1 / gamma
        maxIntensity = np.max(img)

        if maxIntensity == 0:
            logger.warning("Image has zero maximum intensity")
            return img

        corrected_img = np.array(maxIntensity * (img / maxIntensity) ** gammaCorrection)

        logger.debug("Gamma correction completed successfully")
        return corrected_img

    except Exception as e:
        logger.error(f"Error in gamma correction: {e}")
        return img  # Return original image on error


# Convenience functions for common operations
def safe_divide(numerator: np.ndarray, denominator: np.ndarray,
                default_value: float = 0.0) -> np.ndarray:
    """
    Safely divide two arrays, handling division by zero.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        default_value: Value to use when denominator is zero

    Returns:
        Result of division with safe handling of zero denominators
    """
    try:
        result = np.divide(numerator, denominator,
                          out=np.full_like(numerator, default_value, dtype=float),
                          where=(denominator != 0))
        return result
    except Exception as e:
        logger.error(f"Error in safe division: {e}")
        return np.full_like(numerator, default_value, dtype=float)


def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize an array using different methods.

    Args:
        arr: Input array to normalize
        method: Normalization method ('minmax', 'zscore', 'unit')

    Returns:
        Normalized array
    """
    try:
        if method == 'minmax':
            min_val, max_val = np.min(arr), np.max(arr)
            if max_val == min_val:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)
        elif method == 'zscore':
            mean_val, std_val = np.mean(arr), np.std(arr)
            if std_val == 0:
                return np.zeros_like(arr)
            return (arr - mean_val) / std_val
        elif method == 'unit':
            norm = np.linalg.norm(arr)
            if norm == 0:
                return np.zeros_like(arr)
            return arr / norm
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    except Exception as e:
        logger.error(f"Error in array normalization: {e}")
        return arr
