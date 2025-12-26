#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcium Event Detector Diagnostics Module
=========================================

Comprehensive diagnostics for model training, inference, and data quality.

Features:
- Data integrity checking (iCloud stubs, corrupted files)
- Model diagnostics (checkpoint inspection, parameter counts)
- Training diagnostics (loss tracking, convergence analysis)
- Inference diagnostics (prediction quality, threshold testing)
- System diagnostics (GPU/MPS availability, memory usage)

Author: George Stuyt (with Claude)
Date: 2024-12-26
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# Data Integrity Diagnostics
# ============================================================================

def verify_data_integrity(data_dir: Path, num_samples: int = 10) -> Tuple[bool, List[str]]:
    """
    Verify that TIFF files are actually downloaded and readable.
    
    Prevents iCloud stub file issues where training crashes because
    files are not actually downloaded yet.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing images/ and masks/ subdirectories
    num_samples : int
        Number of random files to check
        
    Returns
    -------
    is_valid : bool
        True if all sampled files are readable
    corrupted_files : list
        List of corrupted/undownloaded file paths
    """
    data_dir = Path(data_dir)
    
    logger.info(f"Verifying data integrity in {data_dir}")
    logger.info(f"Checking random sample of {num_samples} files...")
    
    # Get image and mask files
    image_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    
    if not image_dir.exists() or not mask_dir.exists():
        logger.error(f"Missing directories: images/ or masks/ not found in {data_dir}")
        return False, []
    
    image_files = list(image_dir.glob('*.tif'))
    mask_files = list(mask_dir.glob('*.tif'))
    
    if len(image_files) == 0:
        logger.error("No image files found!")
        return False, []
    
    if len(mask_files) == 0:
        logger.error("No mask files found!")
        return False, []
    
    logger.info(f"Found {len(image_files)} image files, {len(mask_files)} mask files")
    
    # Sample random files
    import random
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    sample_masks = random.sample(mask_files, min(num_samples, len(mask_files)))
    
    corrupted_files = []
    
    # Check images
    for img_path in sample_images:
        try:
            from tifffile import imread
            img = imread(img_path)
            
            # Check if actually loaded (not iCloud stub)
            if img.size == 0 or img.nbytes == 0:
                corrupted_files.append(str(img_path))
                logger.warning(f"Empty file (iCloud stub?): {img_path.name}")
            else:
                logger.info(f"✓ {img_path.name}: OK ({img.shape})")
        except Exception as e:
            corrupted_files.append(str(img_path))
            logger.error(f"Failed to load {img_path.name}: {e}")
    
    # Check masks
    for mask_path in sample_masks:
        try:
            from tifffile import imread
            mask = imread(mask_path)
            
            if mask.size == 0 or mask.nbytes == 0:
                corrupted_files.append(str(mask_path))
                logger.warning(f"Empty file (iCloud stub?): {mask_path.name}")
            else:
                # Verify mask values are valid
                unique_values = np.unique(mask)
                valid_values = {0, 1, 2, 3, -100}  # bg, spark, puff, wave, uncertain
                if not set(unique_values).issubset(valid_values):
                    logger.warning(f"Invalid mask values in {mask_path.name}: {unique_values}")
                logger.info(f"✓ {mask_path.name}: OK ({mask.shape}, classes={unique_values})")
        except Exception as e:
            corrupted_files.append(str(mask_path))
            logger.error(f"Failed to load {mask_path.name}: {e}")
    
    if len(corrupted_files) > 0:
        logger.error(f"Found {len(corrupted_files)} corrupted/undownloaded files!")
        logger.error("This is likely an iCloud issue. To fix:")
        logger.error("1. Go to data folder in Finder")
        logger.error("2. Right-click → 'Download Now' or disable iCloud")
        logger.error("3. Wait for all files to download")
        logger.error("4. Re-run this check")
        return False, corrupted_files
    
    logger.info("✅ Data integrity check passed!")
    logger.info("All sampled files are readable")
    return True, []


def check_data_balance(data_dir: Path) -> Dict:
    """
    Check class balance in training data.
    
    Parameters
    ----------
    data_dir : Path
        Directory containing masks/ subdirectory
        
    Returns
    -------
    stats : dict
        Statistics about class distribution
    """
    from tifffile import imread
    
    mask_dir = data_dir / 'masks'
    mask_files = list(mask_dir.glob('*.tif'))
    
    if len(mask_files) == 0:
        logger.error("No mask files found!")
        return {}
    
    logger.info(f"Analyzing class balance across {len(mask_files)} mask files...")
    
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    total_pixels = 0
    
    for mask_file in tqdm(mask_files, desc="Reading masks"):
        try:
            mask = imread(mask_file)
            total_pixels += mask.size
            
            for class_id in range(4):
                class_counts[class_id] += np.sum(mask == class_id)
        except Exception as e:
            logger.warning(f"Skipping {mask_file.name}: {e}")
    
    # Calculate percentages
    class_names = {0: 'background', 1: 'spark', 2: 'puff', 3: 'wave'}
    stats = {
        'total_pixels': total_pixels,
        'class_counts': class_counts,
        'class_percentages': {},
        'num_files': len(mask_files)
    }
    
    for class_id, count in class_counts.items():
        pct = 100 * count / total_pixels if total_pixels > 0 else 0
        stats['class_percentages'][class_names[class_id]] = pct
        logger.info(f"{class_names[class_id]}: {count:,} pixels ({pct:.2f}%)")
    
    # Check for severe imbalance
    event_pixels = class_counts[1] + class_counts[2] + class_counts[3]
    bg_pixels = class_counts[0]
    
    if event_pixels > 0:
        imbalance_ratio = bg_pixels / event_pixels
        stats['imbalance_ratio'] = imbalance_ratio
        
        if imbalance_ratio > 100:
            logger.warning(f"Severe class imbalance detected! Ratio: {imbalance_ratio:.1f}:1")
            logger.warning("Consider using weighted loss or oversampling")
        else:
            logger.info(f"Class balance ratio (bg:events): {imbalance_ratio:.1f}:1")
    
    return stats


# ============================================================================
# Model Diagnostics
# ============================================================================

def inspect_checkpoint(checkpoint_path: Path) -> Dict:
    """
    Inspect checkpoint file and extract information.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file
        
    Returns
    -------
    info : dict
        Checkpoint information
    """
    logger.info(f"Inspecting checkpoint: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        info = {
            'file': str(checkpoint_path),
            'iteration': checkpoint.get('iteration', 'unknown'),
            'best_val_loss': checkpoint.get('best_val_loss', 'unknown'),
            'has_model': 'model_state_dict' in checkpoint,
            'has_optimizer': 'optimizer_state_dict' in checkpoint,
            'has_scheduler': 'scheduler_state_dict' in checkpoint,
            'has_config': 'config' in checkpoint,
        }
        
        # Try to get model info
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            num_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            info['num_parameters'] = num_params
        
        # Get config info
        if 'config' in checkpoint:
            config = checkpoint['config']
            if isinstance(config, dict):
                info['batch_size'] = config.get('batch_size', 'unknown')
                info['learning_rate'] = config.get('learning_rate', 'unknown')
        
        logger.info(f"Checkpoint info:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")
        
        return info
        
    except Exception as e:
        logger.error(f"Error loading checkpoint: {e}")
        return {'error': str(e)}


def find_latest_checkpoint_by_iteration(checkpoint_dir: Path) -> Optional[Path]:
    """
    Find the MOST RECENT checkpoint by iteration number.
    
    Checks ALL checkpoint files and returns the one with highest iteration.
    Critical fix: prevents using old interrupted.pth when newer checkpoints exist.
    
    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing checkpoints
        
    Returns
    -------
    latest_path : Path or None
        Path to most recent checkpoint
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        logger.warning(f"Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    # Find ALL checkpoint files
    all_checkpoints = []
    
    # Standard checkpoints
    all_checkpoints.extend(checkpoint_dir.glob('checkpoint_iter_*.pth'))
    
    # Best model checkpoints
    all_checkpoints.extend(checkpoint_dir.glob('best_model_iter_*.pth'))
    
    # Special named checkpoints
    for name in ['interrupted.pth', 'latest.pth', 'emergency.pth', 'best_model.pth']:
        ckpt_path = checkpoint_dir / name
        if ckpt_path.exists():
            all_checkpoints.append(ckpt_path)
    
    # Emergency checkpoints
    all_checkpoints.extend(checkpoint_dir.glob('emergency_iter_*.pth'))
    
    if not all_checkpoints:
        logger.info("No checkpoints found")
        return None
    
    logger.info(f"Found {len(all_checkpoints)} checkpoint(s), checking iterations...")
    
    latest_ckpt = None
    max_iter = -1
    
    for ckpt_path in all_checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            iteration = ckpt.get('iteration', 0)
            
            logger.info(f"  {ckpt_path.name}: iteration {iteration:,}")
            
            if iteration > max_iter:
                max_iter = iteration
                latest_ckpt = ckpt_path
        except Exception as e:
            logger.warning(f"  {ckpt_path.name}: Error loading ({e})")
            continue
    
    if latest_ckpt:
        logger.info(f"\n✅ Most recent: {latest_ckpt.name} at iteration {max_iter:,}")
        return latest_ckpt
    
    return None


def count_model_parameters(model: torch.nn.Module) -> Dict:
    """
    Count model parameters.
    
    Parameters
    ----------
    model : Module
        PyTorch model
        
    Returns
    -------
    stats : dict
        Parameter statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    stats = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'total_mb': total_params * 4 / (1024**2),  # Assuming float32
    }
    
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    logger.info(f"  Size: {stats['total_mb']:.2f} MB")
    
    return stats


# ============================================================================
# Training Diagnostics
# ============================================================================

def analyze_training_history(checkpoint_dir: Path) -> Dict:
    """
    Analyze training history from saved metrics.
    
    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing training logs
        
    Returns
    -------
    analysis : dict
        Training analysis results
    """
    metrics_file = checkpoint_dir / 'metrics.json'
    
    if not metrics_file.exists():
        logger.warning(f"No metrics file found at {metrics_file}")
        return {}
    
    import json
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Extract training losses
    train_losses = metrics.get('train_losses', [])
    val_losses = metrics.get('val_losses', [])
    
    if not train_losses:
        logger.warning("No training losses found in metrics")
        return {}
    
    analysis = {
        'num_iterations': len(train_losses),
        'num_validations': len(val_losses),
        'final_train_loss': train_losses[-1] if train_losses else None,
        'final_val_loss': val_losses[-1][1] if val_losses else None,
        'best_val_loss': min(v[1] for v in val_losses) if val_losses else None,
    }
    
    # Check for overfitting
    if val_losses and train_losses:
        recent_train = np.mean(train_losses[-100:])
        recent_val = val_losses[-1][1]
        
        if recent_val > recent_train * 1.5:
            analysis['overfitting_warning'] = True
            logger.warning(f"Possible overfitting detected!")
            logger.warning(f"Train loss: {recent_train:.4f}, Val loss: {recent_val:.4f}")
        else:
            analysis['overfitting_warning'] = False
    
    # Check for convergence
    if len(train_losses) > 1000:
        recent_trend = np.polyfit(range(100), train_losses[-100:], 1)[0]
        
        if abs(recent_trend) < 1e-5:
            analysis['converged'] = True
            logger.info("Training appears to have converged (flat loss)")
        else:
            analysis['converged'] = False
            logger.info(f"Training still improving (slope: {recent_trend:.2e})")
    
    logger.info(f"Training analysis:")
    for key, value in analysis.items():
        logger.info(f"  {key}: {value}")
    
    return analysis


# ============================================================================
# System Diagnostics
# ============================================================================

def check_system_capabilities() -> Dict:
    """
    Check system capabilities for training/inference.
    
    Returns
    -------
    capabilities : dict
        System capability information
    """
    capabilities = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
        'cpu_count': torch.get_num_threads(),
    }
    
    if capabilities['cuda_available']:
        capabilities['cuda_device_count'] = torch.cuda.device_count()
        capabilities['cuda_device_name'] = torch.cuda.get_device_name(0)
        capabilities['cuda_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if capabilities['mps_available']:
        try:
            capabilities['mps_memory_allocated_gb'] = torch.mps.current_allocated_memory() / (1024**3)
        except:
            pass
    
    # Determine best device
    if capabilities['cuda_available']:
        capabilities['recommended_device'] = 'cuda'
    elif capabilities['mps_available']:
        capabilities['recommended_device'] = 'mps'
    else:
        capabilities['recommended_device'] = 'cpu'
    
    logger.info("System capabilities:")
    for key, value in capabilities.items():
        logger.info(f"  {key}: {value}")
    
    return capabilities


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Parameters
    ----------
    seconds : float
        Time in seconds
        
    Returns
    -------
    formatted : str
        Human-readable time string
    """
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m {int(seconds % 60)}s"


# ============================================================================
# Inference Diagnostics
# ============================================================================

def test_inference_quality(model, test_image: np.ndarray, 
                          ground_truth: Optional[np.ndarray] = None) -> Dict:
    """
    Test inference quality on a sample image.
    
    Parameters
    ----------
    model : Module
        Trained model
    test_image : ndarray
        Test image (T, H, W)
    ground_truth : ndarray, optional
        Ground truth mask for comparison
        
    Returns
    -------
    results : dict
        Inference quality metrics
    """
    from ca_event_detector.inference.detect import CalciumEventDetector
    
    # Run inference
    detector = CalciumEventDetector(model=model)
    results = detector.detect(test_image)
    
    class_mask = results['class_mask']
    instance_mask = results['instance_mask']
    
    # Basic statistics
    stats = {
        'num_sparks': int(np.sum(class_mask == 1)),
        'num_puffs': int(np.sum(class_mask == 2)),
        'num_waves': int(np.sum(class_mask == 3)),
        'num_instances': int(instance_mask.max()),
    }
    
    # Compare with ground truth if available
    if ground_truth is not None:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        # Flatten for metrics
        pred_flat = class_mask.flatten()
        gt_flat = ground_truth.flatten()
        
        # Overall accuracy
        stats['accuracy'] = accuracy_score(gt_flat, pred_flat)
        
        # Per-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            gt_flat, pred_flat, average=None, labels=[1, 2, 3], zero_division=0
        )
        
        stats['spark_precision'] = float(precision[0])
        stats['spark_recall'] = float(recall[0])
        stats['spark_f1'] = float(f1[0])
        
        stats['puff_precision'] = float(precision[1])
        stats['puff_recall'] = float(recall[1])
        stats['puff_f1'] = float(f1[1])
        
        stats['wave_precision'] = float(precision[2])
        stats['wave_recall'] = float(recall[2])
        stats['wave_f1'] = float(f1[2])
        
        logger.info(f"Inference quality metrics:")
        logger.info(f"  Accuracy: {stats['accuracy']:.3f}")
        logger.info(f"  Spark F1: {stats['spark_f1']:.3f}")
        logger.info(f"  Puff F1: {stats['puff_f1']:.3f}")
        logger.info(f"  Wave F1: {stats['wave_f1']:.3f}")
    
    return stats
