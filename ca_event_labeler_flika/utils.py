#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility Functions for Calcium Event Labeler
============================================

Common utility functions for the labeling plugin.

Author: George Stuyt (with Claude)
Date: 2024-12-26
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)


def normalize_image(image: np.ndarray,
                   method: str = 'percentile',
                   **kwargs) -> np.ndarray:
    """
    Normalize image to 0-1 range.

    Parameters
    ----------
    image : ndarray
        Input image
    method : str
        'minmax', 'percentile', or 'zscore'
    **kwargs
        Additional arguments for specific methods

    Returns
    -------
    normalized : ndarray
        Normalized image
    """
    if method == 'minmax':
        vmin = image.min()
        vmax = image.max()
    elif method == 'percentile':
        p_low = kwargs.get('p_low', 1)
        p_high = kwargs.get('p_high', 99)
        vmin = np.percentile(image, p_low)
        vmax = np.percentile(image, p_high)
    elif method == 'zscore':
        mean = image.mean()
        std = image.std()
        return (image - mean) / (std + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")

    normalized = (image - vmin) / (vmax - vmin + 1e-8)
    normalized = np.clip(normalized, 0, 1)

    return normalized.astype(np.float32)


def convert_to_instance_mask(class_mask: np.ndarray) -> np.ndarray:
    """
    Convert class mask to instance mask with unique IDs.

    Parameters
    ----------
    class_mask : ndarray
        Class labels (T, H, W)

    Returns
    -------
    instance_mask : ndarray
        Instance labels with unique IDs
    """
    from scipy import ndimage

    instance_mask = np.zeros_like(class_mask, dtype=np.uint16)

    # Define 3D connectivity
    structure = np.zeros((3, 3, 3), dtype=bool)
    structure[1, 1, :] = True
    structure[:, 1, 1] = True

    instance_id = 1

    for class_id in [1, 2, 3]:  # For each event class
        class_regions = class_mask == class_id
        labeled, num = ndimage.label(class_regions, structure=structure)

        for region_id in range(1, num + 1):
            region_mask = labeled == region_id
            instance_mask[region_mask] = instance_id
            instance_id += 1

    return instance_mask


def extract_event_properties(instance_mask: np.ndarray,
                             class_mask: np.ndarray,
                             pixel_size: float = 0.2,
                             frame_rate: float = 6.79) -> List[Dict]:
    """
    Extract properties of each event.

    Parameters
    ----------
    instance_mask : ndarray
        Instance labels
    class_mask : ndarray
        Class labels
    pixel_size : float
        Pixel size in micrometers
    frame_rate : float
        Frame rate in milliseconds

    Returns
    -------
    properties : list of dict
        Event properties
    """
    properties = []

    unique_instances = np.unique(instance_mask)
    unique_instances = unique_instances[unique_instances > 0]

    for instance_id in unique_instances:
        mask = instance_mask == instance_id

        # Get class
        class_id = int(np.median(class_mask[mask]))

        # Temporal extent
        t_coords = np.where(np.any(mask, axis=(1, 2)))[0]
        if len(t_coords) == 0:
            continue

        t_start = t_coords[0]
        t_end = t_coords[-1]
        duration_frames = len(t_coords)
        duration_ms = duration_frames * frame_rate

        # Spatial extent
        coords = np.where(mask)
        y_center = np.mean(coords[1])
        x_center = np.mean(coords[2])

        y_size = (coords[1].max() - coords[1].min() + 1) * pixel_size
        x_size = (coords[2].max() - coords[2].min() + 1) * pixel_size

        # Volume
        volume = np.sum(mask)

        properties.append({
            'instance_id': int(instance_id),
            'class_id': int(class_id),
            'class_name': ['bg', 'spark', 'puff', 'wave'][class_id],
            't_start': int(t_start),
            't_end': int(t_end),
            'duration_frames': int(duration_frames),
            'duration_ms': float(duration_ms),
            'y_center': float(y_center),
            'x_center': float(x_center),
            'y_size_um': float(y_size),
            'x_size_um': float(x_size),
            'volume_voxels': int(volume)
        })

    return properties


def load_annotation_file(filepath: Path) -> Tuple[np.ndarray, Dict]:
    """
    Load annotation file with metadata.

    Supports TIFF, NPY, and JSON formats.

    Returns
    -------
    labels : ndarray
        Label array
    metadata : dict
        Metadata
    """
    filepath = Path(filepath)

    if filepath.suffix == '.npy':
        labels = np.load(filepath)
        metadata = {}

        # Try to load metadata if exists
        metadata_file = filepath.with_suffix('.json')
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

    elif filepath.suffix == '.json':
        with open(filepath, 'r') as f:
            data = json.load(f)
        labels = np.array(data['labels'], dtype=np.uint8)
        metadata = {k: v for k, v in data.items() if k != 'labels'}

    else:
        # Assume TIFF
        from tifffile import imread
        labels = imread(filepath)
        metadata = {}

    return labels, metadata


def save_annotation_file(labels: np.ndarray,
                        filepath: Path,
                        metadata: Optional[Dict] = None,
                        format: str = 'tiff'):
    """
    Save annotation file with metadata.

    Parameters
    ----------
    labels : ndarray
        Label array
    filepath : Path
        Output path
    metadata : dict, optional
        Metadata to save
    format : str
        'tiff', 'npy', or 'json'
    """
    filepath = Path(filepath)

    if format == 'npy':
        np.save(filepath, labels)

        if metadata:
            with open(filepath.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f, indent=2, default=str)

    elif format == 'json':
        data = {'labels': labels.tolist()}
        if metadata:
            data.update(metadata)

        with open(filepath, 'w') as f:
            json.dump(data, f, default=str)

    else:  # TIFF
        from tifffile import imwrite
        imwrite(filepath, labels)


def merge_annotations(annotations: List[np.ndarray],
                     method: str = 'majority') -> np.ndarray:
    """
    Merge multiple annotations.

    Parameters
    ----------
    annotations : list of ndarray
        List of annotation arrays
    method : str
        'majority', 'union', or 'intersection'

    Returns
    -------
    merged : ndarray
        Merged annotations
    """
    if len(annotations) == 0:
        raise ValueError("No annotations to merge")

    if len(annotations) == 1:
        return annotations[0]

    # Check shapes match
    shape = annotations[0].shape
    for ann in annotations[1:]:
        if ann.shape != shape:
            raise ValueError("All annotations must have same shape")

    if method == 'majority':
        # Stack and take mode
        stacked = np.stack(annotations, axis=0)
        from scipy.stats import mode
        merged, _ = mode(stacked, axis=0, keepdims=False)
        return merged.astype(annotations[0].dtype)

    elif method == 'union':
        # Take maximum label
        merged = np.maximum.reduce(annotations)
        return merged

    elif method == 'intersection':
        # Take minimum non-zero label
        merged = annotations[0].copy()
        for ann in annotations[1:]:
            # Where both are non-zero, take minimum
            both_nonzero = (merged > 0) & (ann > 0)
            merged[both_nonzero] = np.minimum(merged[both_nonzero], ann[both_nonzero])
            # Where one is zero, set to zero
            either_zero = (merged == 0) | (ann == 0)
            merged[either_zero] = 0
        return merged

    else:
        raise ValueError(f"Unknown method: {method}")


def visualize_labels(image: np.ndarray,
                    labels: np.ndarray,
                    alpha: float = 0.5) -> np.ndarray:
    """
    Create RGB visualization with label overlay.

    Parameters
    ----------
    image : ndarray
        Grayscale image (H, W)
    labels : ndarray
        Label mask (H, W)
    alpha : float
        Overlay transparency

    Returns
    -------
    vis : ndarray
        RGB visualization (H, W, 3)
    """
    # Normalize image
    img_norm = normalize_image(image) * 255

    # Create RGB
    rgb = np.stack([img_norm] * 3, axis=-1).astype(np.uint8)

    # Color map
    colors = {
        0: (0, 0, 0),           # Background - black (transparent)
        1: (0, 255, 0),         # Spark - green
        2: (255, 165, 0),       # Puff - orange
        3: (255, 0, 0)          # Wave - red
    }

    # Apply overlays
    for class_id in [1, 2, 3]:
        mask = labels == class_id
        if np.any(mask):
            color = np.array(colors[class_id])
            rgb[mask] = (rgb[mask] * (1 - alpha) + color * alpha).astype(np.uint8)

    return rgb
