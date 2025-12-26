#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Label Quality Assessment for Calcium Event Training Data
=========================================================

Tools for assessing the quality of training labels.

Metrics:
- Class distribution and balance
- Spatial coverage
- Temporal coverage
- Label consistency
- Annotation completeness
- Inter-annotator agreement
- Boundary quality

Author: George Stuyt (with Claude)
Date: 2024-12-26
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tifffile import imread
import logging

logger = logging.getLogger(__name__)


class LabelQualityChecker:
    """
    Assess quality of training labels for calcium imaging.
    """
    
    CLASS_NAMES = {
        0: 'background',
        1: 'spark',
        2: 'puff',
        3: 'wave'
    }
    
    def __init__(self):
        """Initialize quality checker."""
        pass
    
    def assess_quality(self,
                      data_dir: Path,
                      check_distribution: bool = True,
                      check_coverage: bool = True,
                      check_consistency: bool = True,
                      check_completeness: bool = True) -> Dict:
        """
        Comprehensive quality assessment of labeled data.
        
        Parameters
        ----------
        data_dir : Path
            Directory containing images and masks subdirectories
        check_distribution : bool
            Check class distribution
        check_coverage : bool
            Check spatial/temporal coverage
        check_consistency : bool
            Check label consistency
        check_completeness : bool
            Check annotation completeness
            
        Returns
        -------
        results : dict
            Quality assessment results
        """
        results = {}
        
        # Find all mask files
        mask_dir = data_dir / 'masks'
        if not mask_dir.exists():
            logger.error(f"Masks directory not found: {mask_dir}")
            return {'error': 'Masks directory not found'}
        
        mask_files = list(mask_dir.glob('*.tif*'))
        
        if len(mask_files) == 0:
            logger.error("No mask files found!")
            return {'error': 'No mask files found'}
        
        logger.info(f"Found {len(mask_files)} mask files")
        
        # Class distribution
        if check_distribution:
            logger.info("Checking class distribution...")
            results['distribution'] = self.check_class_distribution(mask_files)
        
        # Spatial/temporal coverage
        if check_coverage:
            logger.info("Checking spatial and temporal coverage...")
            results['coverage'] = self.check_coverage(mask_files)
        
        # Label consistency
        if check_consistency:
            logger.info("Checking label consistency...")
            results['consistency'] = self.check_consistency(mask_files)
        
        # Annotation completeness
        if check_completeness:
            logger.info("Checking annotation completeness...")
            results['completeness'] = self.check_completeness(mask_files, data_dir)
        
        return results
    
    def check_class_distribution(self, mask_files: List[Path]) -> Dict:
        """
        Check distribution of classes across dataset.
        
        Returns class counts, percentages, and balance metrics.
        """
        class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        total_pixels = 0
        file_counts = {0: 0, 1: 0, 2: 0, 3: 0}  # Files containing each class
        
        for mask_file in mask_files:
            try:
                mask = imread(mask_file)
                total_pixels += mask.size
                
                for class_id in [0, 1, 2, 3]:
                    count = np.sum(mask == class_id)
                    class_counts[class_id] += count
                    
                    if count > 0:
                        file_counts[class_id] += 1
                        
            except Exception as e:
                logger.warning(f"Error reading {mask_file.name}: {e}")
                continue
        
        # Calculate percentages
        class_percentages = {}
        for class_id, count in class_counts.items():
            pct = 100 * count / total_pixels if total_pixels > 0 else 0
            class_percentages[class_id] = pct
        
        # Calculate balance metrics
        event_pixels = class_counts[1] + class_counts[2] + class_counts[3]
        bg_pixels = class_counts[0]
        
        imbalance_ratio = bg_pixels / event_pixels if event_pixels > 0 else float('inf')
        
        # Spark:puff:wave ratios
        class_ratios = {}
        if event_pixels > 0:
            class_ratios = {
                'spark': class_counts[1] / event_pixels,
                'puff': class_counts[2] / event_pixels,
                'wave': class_counts[3] / event_pixels
            }
        
        results = {
            'num_files': len(mask_files),
            'total_pixels': total_pixels,
            'class_counts': class_counts,
            'class_percentages': class_percentages,
            'files_per_class': file_counts,
            'imbalance_ratio': imbalance_ratio,
            'event_class_ratios': class_ratios,
            'warnings': []
        }
        
        # Add warnings
        if imbalance_ratio > 100:
            results['warnings'].append(f"Severe class imbalance: {imbalance_ratio:.1f}:1")
        
        # Check if any class is missing
        for class_id in [1, 2, 3]:
            if class_counts[class_id] == 0:
                results['warnings'].append(f"No {self.CLASS_NAMES[class_id]} events labeled!")
        
        # Check if classes are very unbalanced
        if class_ratios:
            max_ratio = max(class_ratios.values())
            min_ratio = min(class_ratios.values())
            if min_ratio > 0 and max_ratio / min_ratio > 10:
                results['warnings'].append("Event classes are very unbalanced")
        
        return results
    
    def check_coverage(self, mask_files: List[Path]) -> Dict:
        """
        Check spatial and temporal coverage of annotations.
        """
        spatial_coverages = []
        temporal_coverages = []
        
        for mask_file in mask_files:
            try:
                mask = imread(mask_file)
                T, H, W = mask.shape
                
                # Spatial coverage (% of frames with any labels)
                frames_with_labels = np.any(mask > 0, axis=(1, 2))
                spatial_coverage = np.sum(frames_with_labels) / T
                spatial_coverages.append(spatial_coverage)
                
                # Temporal coverage (% of spatial area covered)
                any_frame = np.any(mask > 0, axis=0)
                temporal_coverage = np.sum(any_frame) / (H * W)
                temporal_coverages.append(temporal_coverage)
                
            except Exception as e:
                logger.warning(f"Error reading {mask_file.name}: {e}")
                continue
        
        results = {
            'mean_spatial_coverage': np.mean(spatial_coverages) if spatial_coverages else 0,
            'std_spatial_coverage': np.std(spatial_coverages) if spatial_coverages else 0,
            'mean_temporal_coverage': np.mean(temporal_coverages) if temporal_coverages else 0,
            'std_temporal_coverage': np.std(temporal_coverages) if temporal_coverages else 0,
            'warnings': []
        }
        
        # Add warnings
        if results['mean_spatial_coverage'] < 0.1:
            results['warnings'].append("Very sparse temporal annotations (<10% of frames)")
        
        if results['mean_temporal_coverage'] < 0.05:
            results['warnings'].append("Very limited spatial coverage (<5% of area)")
        
        return results
    
    def check_consistency(self, mask_files: List[Path]) -> Dict:
        """
        Check consistency of labels.
        
        Checks for:
        - Isolated single-pixel labels
        - Temporal discontinuities
        - Unusual size distributions
        """
        issues = []
        isolated_pixels = []
        temporal_gaps = []
        size_distributions = {1: [], 2: [], 3: []}
        
        for mask_file in mask_files:
            try:
                mask = imread(mask_file)
                
                # Check for isolated pixels
                from scipy import ndimage
                for t in range(mask.shape[0]):
                    frame = mask[t]
                    for class_id in [1, 2, 3]:
                        class_mask = frame == class_id
                        labeled, num = ndimage.label(class_mask)
                        
                        for region_id in range(1, num + 1):
                            region = labeled == region_id
                            size = np.sum(region)
                            
                            if size == 1:
                                isolated_pixels.append(mask_file.name)
                                break
                            
                            # Track size distribution
                            size_distributions[class_id].append(size)
                
                # Check temporal continuity
                for class_id in [1, 2, 3]:
                    class_mask_3d = mask == class_id
                    labeled_3d, num_events = ndimage.label(class_mask_3d)
                    
                    for event_id in range(1, num_events + 1):
                        event_mask = labeled_3d == event_id
                        frames_with_event = np.where(np.any(event_mask, axis=(1, 2)))[0]
                        
                        if len(frames_with_event) > 1:
                            gaps = np.diff(frames_with_event)
                            if np.any(gaps > 5):  # Gap larger than 5 frames
                                temporal_gaps.append(mask_file.name)
                                break
                
            except Exception as e:
                logger.warning(f"Error checking {mask_file.name}: {e}")
                continue
        
        # Calculate size statistics
        size_stats = {}
        for class_id in [1, 2, 3]:
            if size_distributions[class_id]:
                sizes = size_distributions[class_id]
                size_stats[self.CLASS_NAMES[class_id]] = {
                    'mean': np.mean(sizes),
                    'std': np.std(sizes),
                    'median': np.median(sizes),
                    'min': np.min(sizes),
                    'max': np.max(sizes)
                }
        
        results = {
            'num_files_with_isolated_pixels': len(set(isolated_pixels)),
            'num_files_with_temporal_gaps': len(set(temporal_gaps)),
            'size_statistics': size_stats,
            'warnings': []
        }
        
        if results['num_files_with_isolated_pixels'] > 0:
            results['warnings'].append(
                f"{results['num_files_with_isolated_pixels']} files have isolated single pixels"
            )
        
        if results['num_files_with_temporal_gaps'] > 0:
            results['warnings'].append(
                f"{results['num_files_with_temporal_gaps']} files have temporal discontinuities"
            )
        
        return results
    
    def check_completeness(self, mask_files: List[Path], data_dir: Path) -> Dict:
        """
        Check if annotations are complete.
        
        Checks:
        - Do all images have corresponding masks?
        - Are there orphan masks without images?
        """
        image_dir = data_dir / 'images'
        
        if not image_dir.exists():
            return {'error': 'Images directory not found'}
        
        image_files = list(image_dir.glob('*.tif*'))
        
        # Create mapping
        image_names = {f.stem for f in image_files}
        mask_names = {f.stem.replace('_class', '').replace('_mask', '') for f in mask_files}
        
        # Find mismatches
        images_without_masks = image_names - mask_names
        masks_without_images = mask_names - image_names
        
        results = {
            'num_images': len(image_files),
            'num_masks': len(mask_files),
            'num_images_without_masks': len(images_without_masks),
            'num_masks_without_images': len(masks_without_images),
            'warnings': []
        }
        
        if images_without_masks:
            results['warnings'].append(
                f"{len(images_without_masks)} images missing corresponding masks"
            )
            results['images_without_masks'] = list(images_without_masks)[:10]  # Show first 10
        
        if masks_without_images:
            results['warnings'].append(
                f"{len(masks_without_images)} masks without corresponding images"
            )
            results['masks_without_images'] = list(masks_without_images)[:10]
        
        return results
    
    def compare_annotators(self,
                          mask_files_1: List[Path],
                          mask_files_2: List[Path]) -> Dict:
        """
        Compare annotations from two different annotators.
        
        Calculates inter-annotator agreement metrics.
        
        Parameters
        ----------
        mask_files_1 : list
            Masks from annotator 1
        mask_files_2 : list
            Masks from annotator 2
            
        Returns
        -------
        agreement : dict
            Agreement metrics including IoU, Dice, pixel accuracy
        """
        from sklearn.metrics import accuracy_score, cohen_kappa_score
        
        # Match files by name
        names_1 = {f.stem: f for f in mask_files_1}
        names_2 = {f.stem: f for f in mask_files_2}
        
        common_names = set(names_1.keys()) & set(names_2.keys())
        
        if not common_names:
            return {'error': 'No matching files between annotators'}
        
        all_ious = []
        all_dices = []
        all_accuracies = []
        all_kappas = []
        
        for name in common_names:
            try:
                mask1 = imread(names_1[name])
                mask2 = imread(names_2[name])
                
                # Ensure same shape
                if mask1.shape != mask2.shape:
                    logger.warning(f"Shape mismatch for {name}, skipping")
                    continue
                
                # Flatten for metrics
                m1_flat = mask1.flatten()
                m2_flat = mask2.flatten()
                
                # Pixel accuracy
                acc = accuracy_score(m1_flat, m2_flat)
                all_accuracies.append(acc)
                
                # Cohen's kappa
                kappa = cohen_kappa_score(m1_flat, m2_flat)
                all_kappas.append(kappa)
                
                # IoU and Dice per class
                for class_id in [1, 2, 3]:
                    m1_class = (m1_flat == class_id)
                    m2_class = (m2_flat == class_id)
                    
                    intersection = np.sum(m1_class & m2_class)
                    union = np.sum(m1_class | m2_class)
                    
                    if union > 0:
                        iou = intersection / union
                        dice = 2 * intersection / (np.sum(m1_class) + np.sum(m2_class))
                        
                        all_ious.append(iou)
                        all_dices.append(dice)
                
            except Exception as e:
                logger.warning(f"Error comparing {name}: {e}")
                continue
        
        results = {
            'num_compared': len(common_names),
            'mean_iou': np.mean(all_ious) if all_ious else 0,
            'mean_dice': np.mean(all_dices) if all_dices else 0,
            'mean_accuracy': np.mean(all_accuracies) if all_accuracies else 0,
            'mean_kappa': np.mean(all_kappas) if all_kappas else 0,
        }
        
        # Interpretation
        kappa = results['mean_kappa']
        if kappa < 0.4:
            agreement = "poor"
        elif kappa < 0.6:
            agreement = "moderate"
        elif kappa < 0.8:
            agreement = "substantial"
        else:
            agreement = "excellent"
        
        results['agreement_level'] = agreement
        
        return results
