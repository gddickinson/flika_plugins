#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataset Creator for Calcium Event Training Data
================================================

Create training datasets from labeled calcium imaging data.

Features:
- Extract patches from full frames
- Train/validation/test splitting
- Class balancing via oversampling
- Data augmentation
- Multiple output formats

Author: George Stuyt (with Claude)
Date: 2024-12-26
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from tifffile import imread, imwrite
import shutil
import logging

logger = logging.getLogger(__name__)


class DatasetCreator:
    """
    Create training datasets from labeled calcium imaging data.
    """
    
    def __init__(self):
        """Initialize dataset creator."""
        pass
    
    def create_dataset(self,
                      input_dir: Path,
                      output_dir: Path,
                      patch_size: Tuple[int, int, int] = (16, 64, 64),
                      stride: Optional[Tuple[int, int, int]] = None,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      test_ratio: float = 0.15,
                      augment: bool = True,
                      balance_classes: bool = True,
                      random_seed: int = 42,
                      progress_callback: Optional[Callable] = None) -> Dict:
        """
        Create complete training dataset.
        
        Parameters
        ----------
        input_dir : Path
            Directory with images/ and masks/ subdirectories
        output_dir : Path
            Output directory for dataset
        patch_size : tuple
            (T, H, W) size of patches
        stride : tuple, optional
            Stride for patch extraction (defaults to patch_size // 2)
        train_ratio : float
            Proportion for training set
        val_ratio : float
            Proportion for validation set
        test_ratio : float
            Proportion for test set
        augment : bool
            Apply data augmentation
        balance_classes : bool
            Balance classes via oversampling
        random_seed : int
            Random seed for reproducibility
        progress_callback : callable, optional
            Callback(progress: float) for progress updates
            
        Returns
        -------
        stats : dict
            Dataset statistics
        """
        np.random.seed(random_seed)
        
        logger.info(f"Creating dataset from {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create output directories
        output_dir = Path(output_dir)
        for split in ['train', 'val', 'test']:
            (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (output_dir / split / 'masks').mkdir(parents=True, exist_ok=True)
        
        # Step 1: Extract patches
        if progress_callback:
            progress_callback(0.1)
        
        logger.info("Extracting patches...")
        patches = self.extract_patches(
            input_dir,
            patch_size=patch_size,
            stride=stride
        )
        
        if progress_callback:
            progress_callback(0.3)
        
        # Step 2: Split into train/val/test
        logger.info("Splitting into train/val/test...")
        split_patches = self.split_dataset(
            patches,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        if progress_callback:
            progress_callback(0.5)
        
        # Step 3: Balance classes if requested
        if balance_classes:
            logger.info("Balancing classes...")
            split_patches['train'] = self.balance_classes(split_patches['train'])
        
        if progress_callback:
            progress_callback(0.6)
        
        # Step 4: Apply augmentation if requested
        if augment:
            logger.info("Applying augmentation to training set...")
            from .augmentation import AugmentationEngine
            aug_engine = AugmentationEngine()
            
            augmented_patches = []
            for patch in split_patches['train']:
                augmented_patches.append(patch)
                
                # Create 2-3 augmented versions
                for _ in range(2):
                    aug_patch = aug_engine.augment_patch(
                        patch['image'],
                        patch['mask']
                    )
                    augmented_patches.append({
                        'image': aug_patch[0],
                        'mask': aug_patch[1],
                        'source': patch['source'] + '_aug'
                    })
            
            split_patches['train'] = augmented_patches
        
        if progress_callback:
            progress_callback(0.8)
        
        # Step 5: Save patches
        logger.info("Saving patches...")
        stats = self.save_patches(split_patches, output_dir)
        
        if progress_callback:
            progress_callback(1.0)
        
        logger.info("Dataset creation complete!")
        return stats
    
    def extract_patches(self,
                       input_dir: Path,
                       patch_size: Tuple[int, int, int] = (16, 64, 64),
                       stride: Optional[Tuple[int, int, int]] = None) -> List[Dict]:
        """
        Extract patches from full frames.
        
        Parameters
        ----------
        input_dir : Path
            Directory with images/ and masks/
        patch_size : tuple
            (T, H, W) size
        stride : tuple, optional
            Stride for extraction
            
        Returns
        -------
        patches : list of dict
            List of {'image': array, 'mask': array, 'source': str}
        """
        if stride is None:
            # Default: 50% overlap
            stride = tuple(s // 2 for s in patch_size)
        
        T_patch, H_patch, W_patch = patch_size
        T_stride, H_stride, W_stride = stride
        
        image_dir = input_dir / 'images'
        mask_dir = input_dir / 'masks'
        
        image_files = sorted(image_dir.glob('*.tif*'))
        
        all_patches = []
        
        for img_file in image_files:
            # Find corresponding mask
            mask_file = mask_dir / img_file.name.replace('_video', '_class')
            if not mask_file.exists():
                mask_file = mask_dir / img_file.name
            
            if not mask_file.exists():
                logger.warning(f"No mask for {img_file.name}, skipping")
                continue
            
            try:
                image = imread(img_file)
                mask = imread(mask_file)
                
                if image.shape != mask.shape:
                    logger.warning(f"Shape mismatch for {img_file.name}, skipping")
                    continue
                
                # Extract patches from this file
                T, H, W = image.shape
                
                for t in range(0, T - T_patch + 1, T_stride):
                    for h in range(0, H - H_patch + 1, H_stride):
                        for w in range(0, W - W_patch + 1, W_stride):
                            patch_img = image[t:t+T_patch, h:h+H_patch, w:w+W_patch]
                            patch_mask = mask[t:t+T_patch, h:h+H_patch, w:w+W_patch]
                            
                            # Only keep patches with some signal
                            if np.max(patch_img) > np.mean(patch_img) * 1.5:
                                all_patches.append({
                                    'image': patch_img,
                                    'mask': patch_mask,
                                    'source': f"{img_file.stem}_t{t}_h{h}_w{w}"
                                })
                
            except Exception as e:
                logger.error(f"Error processing {img_file.name}: {e}")
                continue
        
        logger.info(f"Extracted {len(all_patches)} patches")
        return all_patches
    
    def split_dataset(self,
                     patches: List[Dict],
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Dict[str, List[Dict]]:
        """
        Split patches into train/val/test sets.
        
        Uses stratified splitting to maintain class balance.
        """
        # Shuffle patches
        indices = np.random.permutation(len(patches))
        
        # Calculate split points
        n_train = int(len(patches) * train_ratio)
        n_val = int(len(patches) * val_ratio)
        
        # Split
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        
        split_patches = {
            'train': [patches[i] for i in train_idx],
            'val': [patches[i] for i in val_idx],
            'test': [patches[i] for i in test_idx]
        }
        
        logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
        
        return split_patches
    
    def balance_classes(self, patches: List[Dict]) -> List[Dict]:
        """
        Balance classes by oversampling minority classes.
        
        Counts pixels per class and oversamples patches with
        underrepresented classes.
        """
        # Count class representation in each patch
        patch_scores = []
        for patch in patches:
            mask = patch['mask']
            scores = {}
            for class_id in [1, 2, 3]:
                scores[class_id] = np.sum(mask == class_id)
            patch_scores.append(scores)
        
        # Find global class counts
        global_counts = {1: 0, 2: 0, 3: 0}
        for scores in patch_scores:
            for class_id in [1, 2, 3]:
                global_counts[class_id] += scores[class_id]
        
        # Find max count
        max_count = max(global_counts.values())
        
        # Oversample to balance
        balanced_patches = list(patches)  # Start with original
        
        for class_id in [1, 2, 3]:
            if global_counts[class_id] == 0:
                continue
            
            # How many times to duplicate patches with this class?
            target_count = max_count
            current_count = global_counts[class_id]
            
            if current_count < target_count:
                # Find patches with this class
                patches_with_class = [
                    (i, patch) for i, (patch, scores) in enumerate(zip(patches, patch_scores))
                    if scores[class_id] > 0
                ]
                
                # How many patches to add?
                n_to_add = int((target_count - current_count) / (current_count / len(patches_with_class)))
                
                # Oversample
                for _ in range(n_to_add):
                    idx, patch = patches_with_class[np.random.randint(len(patches_with_class))]
                    balanced_patches.append(patch)
        
        logger.info(f"Balanced from {len(patches)} to {len(balanced_patches)} patches")
        
        return balanced_patches
    
    def save_patches(self,
                    split_patches: Dict[str, List[Dict]],
                    output_dir: Path) -> Dict:
        """
        Save patches to disk.
        
        Returns statistics about saved dataset.
        """
        stats = {
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0,
            'total_patches': 0
        }
        
        for split_name, patches in split_patches.items():
            split_dir = output_dir / split_name
            
            for i, patch in enumerate(patches):
                # Save image
                img_path = split_dir / 'images' / f"{split_name}_{i:06d}.tif"
                imwrite(img_path, patch['image'])
                
                # Save mask
                mask_path = split_dir / 'masks' / f"{split_name}_{i:06d}.tif"
                imwrite(mask_path, patch['mask'])
            
            stats[f'{split_name}_samples'] = len(patches)
            stats['total_patches'] += len(patches)
            
            logger.info(f"Saved {len(patches)} {split_name} patches")
        
        # Save metadata
        metadata = {
            'patch_size': patches[0]['image'].shape if patches else None,
            'splits': stats,
            'class_names': {0: 'background', 1: 'spark', 2: 'puff', 3: 'wave'}
        }
        
        import json
        with open(output_dir / 'dataset_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return stats
