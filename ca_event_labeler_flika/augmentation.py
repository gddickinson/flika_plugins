#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Augmentation for Calcium Event Training
=============================================

Augmentation techniques for calcium imaging data.

Transforms:
- Geometric: rotation, flipping, elastic deformation
- Intensity: scaling, gamma, noise
- Temporal: time warping, frame dropping

Author: George Stuyt (with Claude)
Date: 2024-12-26
"""

import numpy as np
from scipy import ndimage
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class AugmentationEngine:
    """
    Data augmentation for calcium imaging patches.
    """
    
    def __init__(self, random_seed: int = None):
        """Initialize augmentation engine."""
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def augment_patch(self,
                     image: np.ndarray,
                     mask: np.ndarray,
                     p_rotate: float = 0.5,
                     p_flip: float = 0.5,
                     p_elastic: float = 0.3,
                     p_intensity: float = 0.7,
                     p_noise: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations to image and mask.
        
        Parameters
        ----------
        image : ndarray
            Input image (T, H, W)
        mask : ndarray
            Label mask (T, H, W)
        p_rotate : float
            Probability of rotation
        p_flip : float
            Probability of flipping
        p_elastic : float
            Probability of elastic deformation
        p_intensity : float
            Probability of intensity transformation
        p_noise : float
            Probability of adding noise
            
        Returns
        -------
        aug_image : ndarray
            Augmented image
        aug_mask : ndarray
            Augmented mask
        """
        aug_image = image.copy()
        aug_mask = mask.copy()
        
        # Rotation (spatial only)
        if np.random.random() < p_rotate:
            aug_image, aug_mask = self.rotate(aug_image, aug_mask)
        
        # Flipping
        if np.random.random() < p_flip:
            aug_image, aug_mask = self.flip(aug_image, aug_mask)
        
        # Elastic deformation
        if np.random.random() < p_elastic:
            aug_image, aug_mask = self.elastic_deform(aug_image, aug_mask)
        
        # Intensity transformations (image only)
        if np.random.random() < p_intensity:
            aug_image = self.intensity_transform(aug_image)
        
        # Add noise (image only)
        if np.random.random() < p_noise:
            aug_image = self.add_noise(aug_image)
        
        return aug_image, aug_mask
    
    def rotate(self,
              image: np.ndarray,
              mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate by 90, 180, or 270 degrees.
        """
        k = np.random.choice([1, 2, 3])  # 90, 180, 270 degrees
        
        # Rotate each frame
        rotated_image = np.rot90(image, k=k, axes=(1, 2))
        rotated_mask = np.rot90(mask, k=k, axes=(1, 2))
        
        return rotated_image, rotated_mask
    
    def flip(self,
            image: np.ndarray,
            mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Flip horizontally or vertically.
        """
        axis = np.random.choice([1, 2])  # Flip along H or W
        
        flipped_image = np.flip(image, axis=axis)
        flipped_mask = np.flip(mask, axis=axis)
        
        return flipped_image.copy(), flipped_mask.copy()
    
    def elastic_deform(self,
                      image: np.ndarray,
                      mask: np.ndarray,
                      alpha: float = 10.0,
                      sigma: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply elastic deformation.
        
        Parameters
        ----------
        alpha : float
            Deformation strength
        sigma : float
            Smoothness of deformation
        """
        T, H, W = image.shape
        
        # Generate random displacement fields
        dx = ndimage.gaussian_filter(
            (np.random.random((H, W)) * 2 - 1), sigma
        ) * alpha
        dy = ndimage.gaussian_filter(
            (np.random.random((H, W)) * 2 - 1), sigma
        ) * alpha
        
        # Create coordinate grids
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # Apply displacement
        indices_y = np.reshape(y + dy, (-1, 1))
        indices_x = np.reshape(x + dx, (-1, 1))
        
        # Deform each frame
        deformed_image = np.zeros_like(image)
        deformed_mask = np.zeros_like(mask)
        
        for t in range(T):
            deformed_image[t] = ndimage.map_coordinates(
                image[t], [indices_y, indices_x], order=1, mode='reflect'
            ).reshape(H, W)
            
            deformed_mask[t] = ndimage.map_coordinates(
                mask[t], [indices_y, indices_x], order=0, mode='reflect'
            ).reshape(H, W)
        
        return deformed_image, deformed_mask
    
    def intensity_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply intensity transformations.
        
        Randomly applies:
        - Brightness adjustment
        - Contrast adjustment
        - Gamma correction
        """
        transformed = image.copy()
        
        # Brightness (additive)
        if np.random.random() < 0.5:
            delta = np.random.uniform(-0.1, 0.1) * (image.max() - image.min())
            transformed = transformed + delta
        
        # Contrast (multiplicative)
        if np.random.random() < 0.5:
            factor = np.random.uniform(0.8, 1.2)
            mean = transformed.mean()
            transformed = (transformed - mean) * factor + mean
        
        # Gamma
        if np.random.random() < 0.5:
            gamma = np.random.uniform(0.8, 1.2)
            # Normalize to 0-1
            img_norm = (transformed - transformed.min()) / (transformed.max() - transformed.min() + 1e-8)
            img_gamma = np.power(img_norm, gamma)
            # Scale back
            transformed = img_gamma * (transformed.max() - transformed.min()) + transformed.min()
        
        return transformed
    
    def add_noise(self, image: np.ndarray, noise_level: float = 0.05) -> np.ndarray:
        """
        Add Gaussian noise.
        
        Parameters
        ----------
        noise_level : float
            Noise standard deviation as fraction of signal range
        """
        signal_range = image.max() - image.min()
        noise_std = noise_level * signal_range
        
        noise = np.random.normal(0, noise_std, image.shape)
        noisy = image + noise
        
        return noisy
    
    def temporal_subsample(self,
                          image: np.ndarray,
                          mask: np.ndarray,
                          factor: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Temporal subsampling (reduce frame rate).
        """
        subsampled_image = image[::factor]
        subsampled_mask = mask[::factor]
        
        return subsampled_image, subsampled_mask
    
    def time_warp(self,
                 image: np.ndarray,
                 mask: np.ndarray,
                 warp_strength: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply temporal warping (non-uniform time scaling).
        
        This simulates variations in event dynamics.
        """
        T, H, W = image.shape
        
        # Generate smooth random time mapping
        t_orig = np.arange(T)
        t_warp = t_orig + np.random.normal(0, warp_strength * T, T)
        t_warp = np.clip(t_warp, 0, T-1)
        t_warp = np.sort(t_warp)  # Maintain monotonicity
        
        # Interpolate
        warped_image = np.zeros_like(image)
        warped_mask = np.zeros_like(mask)
        
        for h in range(H):
            for w in range(W):
                warped_image[:, h, w] = np.interp(t_warp, t_orig, image[:, h, w])
                warped_mask[:, h, w] = np.interp(t_warp, t_orig, mask[:, h, w])
        
        warped_mask = np.round(warped_mask).astype(mask.dtype)
        
        return warped_image, warped_mask
