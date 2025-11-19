#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:48:13 2023

@author: george
"""

from typing import List, Dict, Union, Callable
import numpy as np
import pandas as pd
import logging

# Set up logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def dictFromList(lst: List[Union[str, int]]) -> Dict[str, str]:
    """
    Create a dictionary from a list where each element is both the key and value.

    Args:
        lst (List[Union[str, int]]): Input list

    Returns:
        Dict[str, str]: Dictionary with list elements as both keys and values
    """
    logger.debug(f"Creating dictionary from list of length {len(lst)}")
    result = {str(x): str(x) for x in lst}
    logger.debug(f"Dictionary created with {len(result)} key-value pairs")
    return result

def exp_dec(x: Union[float, np.ndarray], A1: float, tau: float) -> Union[float, np.ndarray]:
    """
    Calculate exponential decay.

    Args:
        x (float or np.ndarray): Input value(s)
        A1 (float): Amplitude
        tau (float): Time constant

    Returns:
        float or np.ndarray: Result of exponential decay calculation

    Raises:
        ValueError: If tau is zero
    """
    logger.debug(f"Calculating exponential decay with A1={A1}, tau={tau}")
    if tau == 0:
        logger.error("Time constant (tau) is zero")
        raise ValueError("Time constant (tau) must be non-zero.")
    return 1 + A1 * np.exp(-x / tau)

def exp_dec_2(x: Union[float, np.ndarray], A1: float, tau1: float, tau2: float) -> Union[float, np.ndarray]:
    """
    Calculate double exponential decay.

    Args:
        x (float or np.ndarray): Input value(s)
        A1 (float): Amplitude 1
        tau1 (float): Time constant 1
        tau2 (float): Time constant 2

    Returns:
        float or np.ndarray: Result of double exponential decay calculation

    Raises:
        ValueError: If tau1 or tau2 is zero
    """
    logger.debug(f"Calculating double exponential decay with A1={A1}, tau1={tau1}, tau2={tau2}")
    if tau1 == 0 or tau2 == 0:
        logger.error("Time constant (tau1 or tau2) is zero")
        raise ValueError("Time constants (tau1 and tau2) must be non-zero.")

    A2 = -1 - A1
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2)

def exp_dec_3(x: Union[float, np.ndarray], A1: float, A2: float, tau1: float, tau2: float, tau3: float) -> Union[float, np.ndarray]:
    """
    Calculate triple exponential decay.

    Args:
        x (float or np.ndarray): Input value(s)
        A1 (float): Amplitude 1
        A2 (float): Amplitude 2
        tau1 (float): Time constant 1
        tau2 (float): Time constant 2
        tau3 (float): Time constant 3

    Returns:
        float or np.ndarray: Result of triple exponential decay calculation

    Raises:
        ValueError: If any of tau1, tau2, or tau3 is zero
    """
    logger.debug(f"Calculating triple exponential decay with A1={A1}, A2={A2}, tau1={tau1}, tau2={tau2}, tau3={tau3}")
    if tau1 == 0 or tau2 == 0 or tau3 == 0:
        logger.error("Time constant (tau1, tau2, or tau3) is zero")
        raise ValueError("Time constants (tau1, tau2, and tau3) must be non-zero.")

    A3 = -1 - A1 - A2
    return 1 + A1 * np.exp(-x / tau1) + A2 * np.exp(-x / tau2) + A3 * np.exp(-x / tau3)

def rollingFunc(arr: np.ndarray, window_size: int = 6, func_type: str = 'mean') -> List[float]:
    """
    Perform a rolling window operation on an array.

    Args:
        arr (np.ndarray): Input array
        window_size (int): Size of the rolling window
        func_type (str): Type of function to apply ('mean', 'std', or 'variance')

    Returns:
        List[float]: Result of the rolling window operation

    Raises:
        ValueError: If an invalid func_type is provided
    """
    logger.debug(f"Performing rolling function with window_size={window_size}, func_type={func_type}")
    series = pd.Series(arr)
    windows = series.rolling(window_size)

    func_map: Dict[str, Callable] = {
        'mean': windows.mean,
        'std': windows.std,
        'variance': windows.var
    }

    if func_type not in func_map:
        logger.error(f"Invalid func_type: {func_type}")
        raise ValueError("Invalid func_type. Must be 'mean', 'std', or 'variance'.")

    moving_averages = func_map[func_type]()
    result = moving_averages[window_size - 1:].tolist()
    logger.debug(f"Rolling function completed, returned list of length {len(result)}")
    return result

def gammaCorrect(img: np.ndarray, gamma: float) -> np.ndarray:
    """
    Apply gamma correction to an image.

    Args:
        img (np.ndarray): Input image
        gamma (float): Gamma correction factor

    Returns:
        np.ndarray: Gamma-corrected image

    Raises:
        ValueError: If gamma is zero or negative
    """
    logger.debug(f"Applying gamma correction with gamma={gamma}")
    if gamma <= 0:
        logger.error(f"Invalid gamma value: {gamma}")
        raise ValueError("Gamma must be a positive number.")

    gamma_correction = 1 / gamma
    max_intensity = np.max(img)
    result = max_intensity * np.power(img / max_intensity, gamma_correction)
    logger.debug(f"Gamma correction completed, returned array of shape {result.shape}")
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Running example usage and testing")

    test_list = [1, 2, 3, 4, 5]
    dict_result = dictFromList(test_list)
    logger.info(f"dictFromList result: {dict_result}")

    x_values = np.linspace(0, 10, 100)
    decay_result = exp_dec(x_values, 1.0, 2.0)
    logger.info(f"exp_dec result shape: {decay_result.shape}")

    test_arr = np.random.rand(20)
    rolling_mean = rollingFunc(test_arr, window_size=5, func_type='mean')
    logger.info(f"rollingFunc result: {rolling_mean}")

    test_img = np.random.rand(10, 10)
    corrected_img = gammaCorrect(test_img, 2.2)
    logger.info(f"gammaCorrect result shape: {corrected_img.shape}")

    logger.info("Example usage and testing completed")


