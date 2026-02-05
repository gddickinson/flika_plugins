#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcium Event Detector Testing Module
=====================================

Comprehensive testing suite for model validation and diagnostics.

Features:
- Model testing with ground truth comparison
- Inference speed benchmarking
- Threshold optimization
- Cross-validation
- Memory profiling

Author: George Stuyt (with Claude)
Date: 2024-12-26
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import torch
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Ground Truth Testing
# ============================================================================

def test_with_ground_truth(model_path: Path, test_data_dir: Path, 
                          config=None) -> Dict:
    """
    Test model against ground truth annotations.
    
    Parameters
    ----------
    model_path : Path
        Path to trained model
    test_data_dir : Path
        Directory with test images and masks
    config : Config, optional
        Detection configuration
        
    Returns
    -------
    metrics : dict
        Performance metrics
    """
    from ca_event_detector.inference.detect import CalciumEventDetector
    from tifffile import imread
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)
    
    logger.info(f"Testing model: {model_path}")
    logger.info(f"Test data: {test_data_dir}")
    
    # Create detector
    detector = CalciumEventDetector(str(model_path), config)
    
    # Find test files
    image_dir = test_data_dir / 'images'
    mask_dir = test_data_dir / 'masks'
    
    image_files = sorted(image_dir.glob('*.tif'))
    
    if len(image_files) == 0:
        logger.error(f"No test images found in {image_dir}")
        return {}
    
    logger.info(f"Found {len(image_files)} test files")
    
    # Collect predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    for image_file in tqdm(image_files, desc="Testing"):
        # Find corresponding mask
        mask_file = mask_dir / image_file.name.replace('_video', '_class')
        if not mask_file.exists():
            # Try alternate naming
            mask_file = mask_dir / image_file.name
        
        if not mask_file.exists():
            logger.warning(f"No mask found for {image_file.name}, skipping")
            continue
        
        try:
            # Load data
            image = imread(image_file)
            ground_truth = imread(mask_file)
            
            # Run detection
            results = detector.detect(image)
            predictions = results['class_mask']
            
            # Collect flattened results
            all_predictions.append(predictions.flatten())
            all_ground_truth.append(ground_truth.flatten())
            
        except Exception as e:
            logger.error(f"Error processing {image_file.name}: {e}")
            continue
    
    if len(all_predictions) == 0:
        logger.error("No valid test results!")
        return {}
    
    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions)
    all_ground_truth = np.concatenate(all_ground_truth)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    
    # Overall accuracy
    accuracy = accuracy_score(all_ground_truth, all_predictions)
    
    # Per-class metrics (ignoring background and uncertain)
    # Only evaluate on pixels that are clearly labeled (0,1,2,3)
    valid_mask = (all_ground_truth != -100)
    valid_predictions = all_predictions[valid_mask]
    valid_ground_truth = all_ground_truth[valid_mask]
    
    # Calculate precision, recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(
        valid_ground_truth, valid_predictions, 
        average=None, labels=[0, 1, 2, 3], zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(valid_ground_truth, valid_predictions, labels=[0, 1, 2, 3])
    
    metrics = {
        'accuracy': float(accuracy),
        'background': {
            'precision': float(precision[0]),
            'recall': float(recall[0]),
            'f1': float(f1[0]),
            'support': int(support[0])
        },
        'spark': {
            'precision': float(precision[1]),
            'recall': float(recall[1]),
            'f1': float(f1[1]),
            'support': int(support[1])
        },
        'puff': {
            'precision': float(precision[2]),
            'recall': float(recall[2]),
            'f1': float(f1[2]),
            'support': int(support[2])
        },
        'wave': {
            'precision': float(precision[3]),
            'recall': float(recall[3]),
            'f1': float(f1[3]),
            'support': int(support[3])
        },
        'confusion_matrix': cm.tolist(),
        'num_test_files': len(all_predictions)
    }
    
    # Log results
    logger.info(f"\n{'='*60}")
    logger.info(f"Test Results ({len(image_files)} files)")
    logger.info(f"{'='*60}")
    logger.info(f"Overall Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"\nPer-class F1 scores:")
    logger.info(f"  Background: {metrics['background']['f1']:.3f}")
    logger.info(f"  Ca²⁺ Sparks: {metrics['spark']['f1']:.3f}")
    logger.info(f"  Ca²⁺ Puffs: {metrics['puff']['f1']:.3f}")
    logger.info(f"  Ca²⁺ Waves: {metrics['wave']['f1']:.3f}")
    logger.info(f"{'='*60}")
    
    return metrics


# ============================================================================
# Threshold Optimization
# ============================================================================

def optimize_threshold(model_path: Path, val_data_dir: Path,
                      threshold_range: Tuple[float, float] = (0.1, 0.9),
                      num_steps: int = 17) -> Dict:
    """
    Find optimal probability threshold for detection.
    
    Parameters
    ----------
    model_path : Path
        Path to trained model
    val_data_dir : Path
        Validation data directory
    threshold_range : tuple
        Range of thresholds to test
    num_steps : int
        Number of thresholds to test
        
    Returns
    -------
    results : dict
        Threshold optimization results
    """
    from ca_event_detector.inference.detect import CalciumEventDetector
    from ca_event_detector.configs.config import Config
    from tifffile import imread
    from sklearn.metrics import f1_score
    
    logger.info("Optimizing probability threshold...")
    logger.info(f"Range: {threshold_range}, Steps: {num_steps}")
    
    # Get test data
    image_dir = val_data_dir / 'images'
    mask_dir = val_data_dir / 'masks'
    
    image_files = list(image_dir.glob('*.tif'))[:5]  # Use subset
    
    if len(image_files) == 0:
        logger.error("No validation images found!")
        return {}
    
    # Test thresholds
    thresholds = np.linspace(threshold_range[0], threshold_range[1], num_steps)
    results_list = []
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        # Create detector with this threshold
        config = Config()
        config.inference.probability_threshold = threshold
        
        detector = CalciumEventDetector(str(model_path), config)
        
        # Test on validation set
        predictions = []
        ground_truths = []
        
        for image_file in image_files:
            mask_file = mask_dir / image_file.name.replace('_video', '_class')
            if not mask_file.exists():
                continue
            
            try:
                image = imread(image_file)
                ground_truth = imread(mask_file)
                
                result = detector.detect(image)
                pred = result['class_mask']
                
                # Only evaluate on labeled pixels
                valid_mask = (ground_truth != -100)
                predictions.extend(pred[valid_mask].flatten())
                ground_truths.extend(ground_truth[valid_mask].flatten())
                
            except Exception as e:
                logger.warning(f"Error processing {image_file.name}: {e}")
                continue
        
        if len(predictions) == 0:
            continue
        
        # Calculate F1 for events (class 1, 2, 3)
        # Convert to binary: 0 = background, 1 = any event
        pred_binary = np.array(predictions) > 0
        gt_binary = np.array(ground_truths) > 0
        
        f1 = f1_score(gt_binary, pred_binary)
        
        results_list.append({
            'threshold': threshold,
            'f1_score': f1
        })
        
        logger.info(f"Threshold {threshold:.2f}: F1 = {f1:.3f}")
    
    if not results_list:
        logger.error("No valid results!")
        return {}
    
    # Find best threshold
    df = pd.DataFrame(results_list)
    best_idx = df['f1_score'].idxmax()
    best_threshold = df.loc[best_idx, 'threshold']
    best_f1 = df.loc[best_idx, 'f1_score']
    
    results = {
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'all_results': results_list
    }
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Optimal threshold: {best_threshold:.3f}")
    logger.info(f"Best F1 score: {best_f1:.3f}")
    logger.info(f"{'='*60}")
    
    return results


# ============================================================================
# Performance Benchmarking
# ============================================================================

def benchmark_inference_speed(model_path: Path, test_image_sizes: List[Tuple] = None,
                             num_runs: int = 10, device: str = 'mps') -> Dict:
    """
    Benchmark inference speed on different image sizes.
    
    Parameters
    ----------
    model_path : Path
        Path to model
    test_image_sizes : list
        List of (T, H, W) tuples to test
    num_runs : int
        Number of runs per size
    device : str
        Device to use
        
    Returns
    -------
    benchmarks : dict
        Timing results
    """
    from ca_event_detector.inference.detect import CalciumEventDetector
    from ca_event_detector.configs.config import Config
    
    if test_image_sizes is None:
        test_image_sizes = [
            (100, 256, 256),    # Small
            (500, 512, 512),    # Medium
            (1000, 512, 512),   # Large
        ]
    
    logger.info(f"Benchmarking inference speed on {device}")
    
    config = Config()
    config.device = device
    
    detector = CalciumEventDetector(str(model_path), config)
    
    benchmarks = []
    
    for size in test_image_sizes:
        T, H, W = size
        logger.info(f"\nTesting size: {T}x{H}x{W}")
        
        # Generate random test image
        test_image = np.random.randint(0, 255, (T, H, W), dtype=np.uint8)
        
        # Warmup
        _ = detector.detect(test_image)
        
        # Benchmark
        times = []
        for _ in tqdm(range(num_runs), desc="Running"):
            start = time.time()
            _ = detector.detect(test_image)
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = T / avg_time  # Frames per second
        
        result = {
            'size': f"{T}x{H}x{W}",
            'avg_time': avg_time,
            'std_time': std_time,
            'fps': fps,
            'runs': num_runs
        }
        
        benchmarks.append(result)
        
        logger.info(f"Average time: {avg_time:.2f}s ± {std_time:.2f}s")
        logger.info(f"Processing speed: {fps:.1f} frames/sec")
    
    return {'device': device, 'results': benchmarks}


# ============================================================================
# Memory Profiling
# ============================================================================

def profile_memory_usage(model_path: Path, test_image_size: Tuple = (500, 512, 512),
                        device: str = 'mps') -> Dict:
    """
    Profile memory usage during inference.
    
    Parameters
    ----------
    model_path : Path
        Path to model
    test_image_size : tuple
        Test image size (T, H, W)
    device : str
        Device to use
        
    Returns
    -------
    profile : dict
        Memory usage statistics
    """
    from ca_event_detector.inference.detect import CalciumEventDetector
    from ca_event_detector.configs.config import Config
    
    config = Config()
    config.device = device
    
    logger.info(f"Profiling memory usage on {device}")
    logger.info(f"Test size: {test_image_size}")
    
    # Get baseline memory
    if device == 'cuda':
        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated() / (1024**3)
    elif device == 'mps':
        baseline_memory = torch.mps.current_allocated_memory() / (1024**3)
    else:
        baseline_memory = 0
    
    logger.info(f"Baseline memory: {baseline_memory:.2f} GB")
    
    # Create detector
    detector = CalciumEventDetector(str(model_path), config)
    
    # Get memory after model load
    if device == 'cuda':
        model_memory = torch.cuda.memory_allocated() / (1024**3)
    elif device == 'mps':
        model_memory = torch.mps.current_allocated_memory() / (1024**3)
    else:
        model_memory = 0
    
    logger.info(f"After model load: {model_memory:.2f} GB")
    
    # Generate test image
    T, H, W = test_image_size
    test_image = np.random.randint(0, 255, (T, H, W), dtype=np.uint8)
    
    # Run inference
    _ = detector.detect(test_image)
    
    # Get peak memory
    if device == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        current_memory = torch.cuda.memory_allocated() / (1024**3)
    elif device == 'mps':
        peak_memory = torch.mps.driver_allocated_memory() / (1024**3)
        current_memory = torch.mps.current_allocated_memory() / (1024**3)
    else:
        peak_memory = 0
        current_memory = 0
    
    profile = {
        'device': device,
        'test_size': test_image_size,
        'baseline_memory_gb': baseline_memory,
        'model_memory_gb': model_memory,
        'peak_memory_gb': peak_memory,
        'current_memory_gb': current_memory,
        'inference_memory_gb': peak_memory - model_memory
    }
    
    logger.info(f"\nMemory Profile:")
    logger.info(f"  Model: {model_memory:.2f} GB")
    logger.info(f"  Peak: {peak_memory:.2f} GB")
    logger.info(f"  Inference overhead: {profile['inference_memory_gb']:.2f} GB")
    
    return profile


# ============================================================================
# Model Comparison
# ============================================================================

def compare_models(model_paths: List[Path], test_data_dir: Path) -> pd.DataFrame:
    """
    Compare multiple models on same test set.
    
    Parameters
    ----------
    model_paths : list
        List of model paths to compare
    test_data_dir : Path
        Test data directory
        
    Returns
    -------
    comparison : DataFrame
        Comparison results
    """
    logger.info(f"Comparing {len(model_paths)} models")
    
    results = []
    
    for model_path in tqdm(model_paths, desc="Testing models"):
        logger.info(f"\nTesting: {model_path.name}")
        
        try:
            # Test with ground truth
            metrics = test_with_ground_truth(model_path, test_data_dir)
            
            # Add model info
            metrics['model'] = model_path.name
            metrics['model_path'] = str(model_path)
            
            results.append(metrics)
            
        except Exception as e:
            logger.error(f"Error testing {model_path.name}: {e}")
            continue
    
    if not results:
        logger.error("No valid results!")
        return pd.DataFrame()
    
    # Create comparison dataframe
    df = pd.DataFrame(results)
    
    # Sort by F1 score
    if 'spark' in df.columns:
        df = df.sort_values('spark.f1', ascending=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("Model Comparison Results")
    logger.info(f"{'='*60}")
    logger.info(df[['model', 'accuracy', 'spark.f1', 'puff.f1', 'wave.f1']])
    logger.info(f"{'='*60}")
    
    return df
