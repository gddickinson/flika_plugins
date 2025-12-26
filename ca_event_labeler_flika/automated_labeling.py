#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Label Generation for Calcium Events
==============================================

Automatically generate training labels based on signal analysis.

Methods:
- Intensity thresholding
- Temporal filtering
- Morphological operations
- Connected component analysis
- Size-based classification
- Duration-based classification

Author: George Stuyt (with Claude)
Date: 2024-12-26
"""

import numpy as np
from scipy import ndimage
from skimage import morphology, measure
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AutomatedLabeler:
    """
    Automated label generation for calcium imaging data.
    
    Uses signal processing techniques to detect and classify events.
    """
    
    # Classification criteria (pixels and frames)
    SPARK_MAX_SIZE = 50      # Max size for sparks
    PUFF_MIN_SIZE = 30       # Min size for puffs
    PUFF_MAX_SIZE = 200      # Max size for puffs
    WAVE_MIN_SIZE = 150      # Min size for waves
    
    SPARK_MAX_DURATION = 10  # Max duration for sparks
    PUFF_MIN_DURATION = 5    # Min duration for puffs
    PUFF_MAX_DURATION = 30   # Max duration for puffs
    WAVE_MIN_DURATION = 20   # Min duration for waves
    
    def __init__(self):
        """Initialize automated labeler."""
        pass
    
    def generate_labels(self,
                       image: np.ndarray,
                       intensity_threshold: float = 0.3,
                       temporal_filter_size: int = 3,
                       min_event_size: int = 10,
                       min_event_duration: int = 2,
                       auto_classify: bool = True) -> np.ndarray:
        """
        Generate automated labels for calcium imaging data.
        
        Parameters
        ----------
        image : ndarray
            Input image (T, H, W)
        intensity_threshold : float
            Threshold for event detection (0-1, normalized)
        temporal_filter_size : int
            Size of temporal median filter
        min_event_size : int
            Minimum event size in pixels
        min_event_duration : int
            Minimum event duration in frames
        auto_classify : bool
            Automatically classify events by size/duration
            
        Returns
        -------
        labels : ndarray
            Label mask (T, H, W) with values 0=bg, 1=spark, 2=puff, 3=wave
        """
        logger.info("Starting automated label generation...")
        
        T, H, W = image.shape
        
        # Step 1: Normalize image
        logger.info("Normalizing image...")
        image_norm = self.normalize_image(image)
        
        # Step 2: Apply temporal filtering
        logger.info(f"Applying temporal filter (size={temporal_filter_size})...")
        image_filtered = self.temporal_filter(image_norm, temporal_filter_size)
        
        # Step 3: Threshold to detect events
        logger.info(f"Thresholding at {intensity_threshold}...")
        binary = image_filtered > intensity_threshold
        
        # Step 4: Morphological cleanup
        logger.info("Morphological cleanup...")
        binary_clean = self.morphological_cleanup(binary)
        
        # Step 5: Connected component labeling
        logger.info("Connected component labeling...")
        labeled, num_events = self.connected_components_3d(binary_clean)
        
        logger.info(f"Found {num_events} initial events")
        
        # Step 6: Filter by size and duration
        logger.info("Filtering by size and duration...")
        labels = self.filter_events(
            labeled, 
            min_size=min_event_size,
            min_duration=min_event_duration
        )
        
        # Step 7: Classify events
        if auto_classify:
            logger.info("Auto-classifying events...")
            labels = self.classify_events(labels)
        else:
            # All events are class 1 (spark) by default
            labels = (labels > 0).astype(np.uint8)
        
        num_final = len(np.unique(labels)) - 1  # Subtract background
        logger.info(f"Generated {num_final} labeled events")
        
        return labels
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image to 0-1 range.
        
        Uses percentile-based normalization to be robust to outliers.
        """
        p_low = np.percentile(image, 1)
        p_high = np.percentile(image, 99)
        
        image_norm = (image - p_low) / (p_high - p_low + 1e-8)
        image_norm = np.clip(image_norm, 0, 1)
        
        return image_norm.astype(np.float32)
    
    def temporal_filter(self, image: np.ndarray, filter_size: int) -> np.ndarray:
        """
        Apply temporal median filter to reduce noise.
        
        Parameters
        ----------
        image : ndarray
            Input image (T, H, W)
        filter_size : int
            Size of median filter in time dimension
            
        Returns
        -------
        filtered : ndarray
            Filtered image
        """
        if filter_size <= 1:
            return image
        
        # Apply median filter along time axis
        filtered = ndimage.median_filter(image, size=(filter_size, 1, 1))
        
        return filtered
    
    def morphological_cleanup(self, binary: np.ndarray) -> np.ndarray:
        """
        Clean up binary mask using morphological operations.
        
        Removes small isolated pixels and fills small holes.
        """
        # Process each frame
        cleaned = np.zeros_like(binary)
        
        for t in range(binary.shape[0]):
            frame = binary[t]
            
            # Remove small objects
            frame = morphology.remove_small_objects(frame, min_size=5)
            
            # Close small holes
            frame = morphology.binary_closing(frame, morphology.disk(2))
            
            cleaned[t] = frame
        
        return cleaned
    
    def connected_components_3d(self, binary: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Label connected components in 3D (time + space).
        
        Parameters
        ----------
        binary : ndarray
            Binary mask (T, H, W)
            
        Returns
        -------
        labeled : ndarray
            Labeled image where each event has unique ID
        num_events : int
            Number of detected events
        """
        # Define 3D connectivity (6-connectivity: time + 4-spatial)
        structure = np.zeros((3, 3, 3), dtype=bool)
        structure[1, 1, :] = True  # Spatial connectivity
        structure[:, 1, 1] = True  # Temporal connectivity
        
        labeled, num_events = ndimage.label(binary, structure=structure)
        
        return labeled, num_events
    
    def filter_events(self, 
                     labeled: np.ndarray,
                     min_size: int = 10,
                     min_duration: int = 2) -> np.ndarray:
        """
        Filter events by size and duration criteria.
        
        Parameters
        ----------
        labeled : ndarray
            Labeled image
        min_size : int
            Minimum size in voxels
        min_duration : int
            Minimum duration in frames
            
        Returns
        -------
        filtered : ndarray
            Filtered labeled image
        """
        filtered = np.zeros_like(labeled)
        
        unique_labels = np.unique(labeled)
        unique_labels = unique_labels[unique_labels > 0]  # Skip background
        
        new_label = 1
        
        for label_id in unique_labels:
            mask = labeled == label_id
            
            # Check size
            size = np.sum(mask)
            if size < min_size:
                continue
            
            # Check duration
            frames_with_event = np.any(mask, axis=(1, 2))
            duration = np.sum(frames_with_event)
            if duration < min_duration:
                continue
            
            # Keep this event
            filtered[mask] = new_label
            new_label += 1
        
        return filtered
    
    def classify_events(self, labeled: np.ndarray) -> np.ndarray:
        """
        Classify events as sparks (1), puffs (2), or waves (3).
        
        Classification based on:
        - Size (spatial extent)
        - Duration (temporal extent)
        
        Criteria:
        - Sparks: Small (<50 pixels), short (<10 frames)
        - Puffs: Medium (30-200 pixels), medium (5-30 frames)
        - Waves: Large (>150 pixels), long (>20 frames)
        
        Parameters
        ----------
        labeled : ndarray
            Instance labeled image
            
        Returns
        -------
        classified : ndarray
            Class labeled image (0=bg, 1=spark, 2=puff, 3=wave)
        """
        classified = np.zeros_like(labeled, dtype=np.uint8)
        
        unique_labels = np.unique(labeled)
        unique_labels = unique_labels[unique_labels > 0]
        
        logger.info(f"Classifying {len(unique_labels)} events...")
        
        for label_id in unique_labels:
            mask = labeled == label_id
            
            # Calculate properties
            size = np.sum(mask)
            
            # Get temporal extent
            frames_with_event = np.any(mask, axis=(1, 2))
            duration = np.sum(frames_with_event)
            
            # Get spatial extent (max area in any frame)
            max_area = 0
            for t in range(labeled.shape[0]):
                if frames_with_event[t]:
                    area = np.sum(mask[t])
                    max_area = max(max_area, area)
            
            # Classify based on size and duration
            if max_area <= self.SPARK_MAX_SIZE and duration <= self.SPARK_MAX_DURATION:
                event_class = 1  # Spark
            elif max_area >= self.WAVE_MIN_SIZE and duration >= self.WAVE_MIN_DURATION:
                event_class = 3  # Wave
            elif (self.PUFF_MIN_SIZE <= max_area <= self.PUFF_MAX_SIZE and
                  self.PUFF_MIN_DURATION <= duration <= self.PUFF_MAX_DURATION):
                event_class = 2  # Puff
            else:
                # Default to puff for ambiguous cases
                event_class = 2
            
            classified[mask] = event_class
        
        # Count events per class
        for class_id, class_name in [(1, 'sparks'), (2, 'puffs'), (3, 'waves')]:
            count = np.sum(np.any(classified == class_id, axis=(1, 2)))
            logger.info(f"  {class_name}: {count}")
        
        return classified
    
    def refine_labels(self,
                     labels: np.ndarray,
                     image: np.ndarray,
                     refinement_iterations: int = 3) -> np.ndarray:
        """
        Refine labels using active contours or watershed.
        
        This is an advanced feature for improving label boundaries.
        
        Parameters
        ----------
        labels : ndarray
            Initial labels
        image : ndarray
            Original image
        refinement_iterations : int
            Number of refinement iterations
            
        Returns
        -------
        refined : ndarray
            Refined labels
        """
        # Placeholder for advanced refinement
        # Could implement:
        # - Active contours
        # - Watershed segmentation
        # - Level sets
        # - GrabCut
        
        logger.info("Label refinement not yet implemented")
        return labels
    
    def detect_by_template_matching(self,
                                   image: np.ndarray,
                                   template: np.ndarray,
                                   threshold: float = 0.8) -> np.ndarray:
        """
        Detect events using template matching.
        
        Useful for detecting events with consistent morphology.
        
        Parameters
        ----------
        image : ndarray
            Input image
        template : ndarray
            Template event (T, H, W)
        threshold : float
            Correlation threshold
            
        Returns
        -------
        detections : ndarray
            Binary detection mask
        """
        from scipy.signal import correlate
        
        # Normalize
        image_norm = self.normalize_image(image)
        template_norm = self.normalize_image(template)
        
        # Correlate (this is computationally expensive for 3D)
        correlation = correlate(image_norm, template_norm, mode='same')
        
        # Threshold
        detections = correlation > threshold
        
        return detections
    
    def estimate_noise_level(self, image: np.ndarray) -> float:
        """
        Estimate noise level in image.
        
        Uses median absolute deviation of background regions.
        
        Returns
        -------
        noise_std : float
            Estimated noise standard deviation
        """
        # Use bottom 20% of intensity values as background
        sorted_vals = np.sort(image.flatten())
        n_bg = int(len(sorted_vals) * 0.2)
        background = sorted_vals[:n_bg]
        
        # MAD estimator
        mad = np.median(np.abs(background - np.median(background)))
        noise_std = 1.4826 * mad  # Convert MAD to std
        
        return noise_std
