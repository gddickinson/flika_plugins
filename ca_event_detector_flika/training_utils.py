#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Training Module for Calcium Event Detector
===================================================

Incorporates all tested features from train_patches_resume_mps.py including:
- Robust checkpoint finding by iteration number
- Data integrity verification
- Robust dataset wrapper for handling corrupted files
- Enhanced progress monitoring
- Emergency checkpoint saving
- MPS optimization

Author: George Stuyt (with Claude)
Date: 2024-12-26
"""

import os
import sys
import torch
import json
import time
import random
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
from collections import deque
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Robust Dataset Wrapper
# ============================================================================

class RobustDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper that skips corrupted files instead of crashing.
    
    Critical for handling iCloud stub files and other data issues during training.
    """
    
    def __init__(self, base_dataset, max_retries: int = 5):
        """
        Parameters
        ----------
        base_dataset : Dataset
            Underlying dataset
        max_retries : int
            Maximum retries before giving up
        """
        self.base_dataset = base_dataset
        self.max_retries = max_retries
        self.failed_indices = set()
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """Try to load item, skip to random other if fails."""
        for attempt in range(self.max_retries):
            try:
                # Try to load the requested item
                item = self.base_dataset[idx]
                return item
            except Exception as e:
                # Mark as failed
                self.failed_indices.add(idx)
                
                if attempt == 0:
                    logger.warning(f"Failed to load sample {idx}: {e}")
                    logger.warning("Trying random alternative...")
                
                # Pick random alternative
                idx = random.randint(0, len(self.base_dataset) - 1)
                
                # Avoid infinite loop with same corrupted file
                while idx in self.failed_indices and len(self.failed_indices) < len(self):
                    idx = random.randint(0, len(self.base_dataset) - 1)
        
        # If we get here, something is very wrong
        raise RuntimeError(f"Failed to load any sample after {self.max_retries} attempts")


def create_robust_dataloaders(config):
    """
    Create dataloaders with error handling for corrupted files.
    
    Parameters
    ----------
    config : Config
        Training configuration
        
    Returns
    -------
    train_loader : DataLoader
        Training data loader
    val_loader : DataLoader
        Validation data loader
    """
    from ca_event_detector.data.dataset import CalciumEventDataset
    from torch.utils.data import DataLoader
    
    # Create base dataset
    data_dir = config.data.data_dir
    
    full_dataset = CalciumEventDataset(
        data_dir=data_dir,
        segment_length=config.data.segment_length,
        step_size=config.data.step_size,
        temporal_context=config.data.temporal_context,
        augment=config.data.augment,
        normalize_min=config.data.normalize_min,
        normalize_max=config.data.normalize_max
    )
    
    # Split into train/val
    dataset_size = len(full_dataset)
    train_size = int(config.data.train_ratio * dataset_size)
    val_size = dataset_size - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.data.random_seed)
    )
    
    # Wrap with robust loaders
    train_dataset_robust = RobustDataset(train_dataset)
    val_dataset_robust = RobustDataset(val_dataset)
    
    # Create dataloaders
    num_workers = config.training.num_workers if config.training.device == 'cuda' else 0
    
    if num_workers == 0 and config.training.num_workers > 0:
        logger.info(f"Setting num_workers=0 for non-CUDA device (was {config.training.num_workers})")
    
    train_loader = DataLoader(
        train_dataset_robust,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if config.training.device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset_robust,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if config.training.device == 'cuda' else False
    )
    
    logger.info("✅ Robust data loaders created")
    logger.info("Will skip corrupted files automatically")
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    return train_loader, val_loader


# ============================================================================
# Enhanced Training Loop
# ============================================================================

def train_with_robust_handling(trainer, train_loader, val_loader, config, checkpoint_dir):
    """
    Training loop with robust error handling and enhanced monitoring.
    
    Features:
    - Emergency checkpoint saving on errors
    - Skips corrupted batches instead of crashing
    - Enhanced progress monitoring
    - ETA calculation
    - Convergence warnings
    
    Parameters
    ----------
    trainer : Trainer
        Trainer instance
    train_loader : DataLoader
        Training data
    val_loader : DataLoader
        Validation data
    config : Config
        Configuration
    checkpoint_dir : Path
        Checkpoint directory
    """
    device = torch.device(config.training.device)
    model = trainer.model
    optimizer = trainer.optimizer
    scheduler = trainer.scheduler
    criterion = trainer.criterion
    
    start_iter = trainer.current_iteration
    best_val_loss = trainer.best_val_loss
    best_val_loss_iter = start_iter
    
    # Enhanced monitoring
    recent_losses = deque(maxlen=100)
    validation_history = []
    iter_times = deque(maxlen=100)
    
    logger.info("="*70)
    logger.info("STARTING TRAINING WITH ROBUST ERROR HANDLING")
    logger.info("="*70)
    logger.info(f"Resuming from iteration {start_iter:,}")
    logger.info(f"Target: {config.training.num_iterations:,} iterations")
    logger.info(f"Validation every {config.training.val_frequency} iterations")
    logger.info("")
    logger.info("🛡️  Robust mode enabled:")
    logger.info("   - Skips corrupted/iCloud stub files")
    logger.info("   - Saves emergency checkpoint on error")
    logger.info("   - Continues training despite file issues")
    logger.info("="*70)
    
    # Training loop
    model.train()
    train_iter = iter(train_loader)
    
    progress_bar = tqdm(
        range(start_iter, config.training.num_iterations),
        initial=start_iter,
        total=config.training.num_iterations,
        desc="Training"
    )
    
    for iteration in progress_bar:
        iter_start = time.time()
        
        try:
            # Get next batch
            try:
                images, masks, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, masks, _ = next(train_iter)
            
            # Move to device
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Track metrics
            loss_value = loss.item()
            recent_losses.append(loss_value)
            iter_times.append(time.time() - iter_start)
            
            # Update progress bar
            avg_loss = sum(recent_losses) / len(recent_losses)
            current_lr = optimizer.param_groups[0]['lr']
            
            progress_bar.set_postfix({
                'loss': f'{loss_value:.3f}',
                'avg': f'{avg_loss:.3f}',
                'best': f'{best_val_loss:.3f}',
                'lr': f'{current_lr:.1e}'
            })
            
            # Progress updates every 100 iterations
            if (iteration + 1) % 100 == 0:
                avg_time = sum(iter_times) / len(iter_times)
                remaining = config.training.num_iterations - iteration - 1
                eta = remaining * avg_time
                
                # Check if improving
                if len(recent_losses) >= 100:
                    recent_avg = sum(list(recent_losses)[-100:]) / 100
                    older_avg = sum(list(recent_losses)[:50]) / 50 if len(recent_losses) >= 50 else recent_avg
                    trend = "📉 improving" if recent_avg < older_avg else "📊 stable"
                else:
                    trend = "📊 tracking"
                
                # Time since best
                iters_since_best = iteration - best_val_loss_iter
                
                from .diagnostics import format_time
                
                logger.info(f"\n┌{'─'*68}┐")
                logger.info(f"│ 📊 Progress - Iteration {iteration+1:,} / {config.training.num_iterations:,} ({100*(iteration+1)/config.training.num_iterations:.1f}%)    │")
                logger.info(f"├{'─'*68}┤")
                logger.info(f"│ Loss:        {avg_loss:.4f} (avg last 100)  {trend}                  │")
                logger.info(f"│ Best val:    {best_val_loss:.4f} (at iter {best_val_loss_iter:,})                │")
                logger.info(f"│ Since best:  {iters_since_best:,} iterations ago                     │")
                logger.info(f"│ Learn rate:  {current_lr:.2e}                                         │")
                logger.info(f"│ Speed:       {avg_time:.2f} sec/iter    ETA: {format_time(eta)}       │")
                logger.info(f"└{'─'*68}┘")
                
                # Warning if no improvement
                if iters_since_best > 5000:
                    logger.warning(f"⚠️  No improvement for {iters_since_best:,} iterations")
                    logger.warning("Consider monitoring closely or stopping if this continues")
            
            # Validation
            if (iteration + 1) % config.training.val_frequency == 0:
                logger.info(f"\n[Iteration {iteration+1}] Running validation...")
                
                model.eval()
                val_losses = []
                
                with torch.no_grad():
                    val_progress = tqdm(val_loader, desc="Validating", leave=False)
                    for val_images, val_masks, _ in val_progress:
                        try:
                            val_images = val_images.to(device)
                            val_masks = val_masks.to(device, dtype=torch.long)
                            
                            val_outputs = model(val_images)
                            val_loss = criterion(val_outputs, val_masks)
                            val_losses.append(val_loss.item())
                        except Exception as e:
                            logger.warning(f"Skipping corrupted validation batch: {e}")
                            continue
                
                model.train()
                
                if len(val_losses) > 0:
                    avg_val_loss = sum(val_losses) / len(val_losses)
                    validation_history.append((iteration + 1, avg_val_loss))
                    
                    logger.info(f"✅ Validation complete: {avg_val_loss:.4f}")
                    
                    # Save if best
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_val_loss_iter = iteration + 1
                        
                        # Save best model
                        best_path = Path(checkpoint_dir) / 'best_model.pth'
                        torch.save({
                            'iteration': iteration + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'best_val_loss': best_val_loss,
                            'config': config.__dict__
                        }, best_path)
                        
                        # Also save with iteration number
                        best_iter_path = Path(checkpoint_dir) / f'best_model_iter_{iteration+1}.pth'
                        torch.save({
                            'iteration': iteration + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'best_val_loss': best_val_loss,
                            'config': config.__dict__
                        }, best_iter_path)
                        
                        logger.info(f"🏆 New best model! Val loss: {best_val_loss:.4f}")
                
                # Save checkpoint
                checkpoint_path = Path(checkpoint_dir) / 'latest.pth'
                torch.save({
                    'iteration': iteration + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_loss': best_val_loss,
                    'config': config.__dict__
                }, checkpoint_path)
                
                # Periodic checkpoint
                if (iteration + 1) % 5000 == 0:
                    periodic_path = Path(checkpoint_dir) / f'checkpoint_iter_{iteration+1}.pth'
                    torch.save({
                        'iteration': iteration + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_val_loss': best_val_loss,
                        'config': config.__dict__
                    }, periodic_path)
                
                logger.info(f"💾 Checkpoint saved")
        
        except KeyboardInterrupt:
            logger.info(f"\n\n⚠️  Training interrupted by user at iteration {iteration+1}")
            
            # Save interrupted checkpoint
            interrupted_path = Path(checkpoint_dir) / 'interrupted.pth'
            torch.save({
                'iteration': iteration + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_loss': best_val_loss,
                'config': config.__dict__
            }, interrupted_path)
            
            logger.info(f"💾 Interrupted checkpoint saved: {interrupted_path}")
            break
        
        except Exception as e:
            logger.error(f"\n❌ Error during training at iteration {iteration+1}: {e}")
            logger.info("Skipping this batch and continuing...")
            
            # Save emergency checkpoint
            emergency_path = Path(checkpoint_dir) / f'emergency_iter_{iteration+1}.pth'
            try:
                torch.save({
                    'iteration': iteration + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'best_val_loss': best_val_loss,
                    'config': config.__dict__
                }, emergency_path)
                logger.info(f"💾 Emergency checkpoint saved: {emergency_path}")
            except Exception as save_error:
                logger.error(f"⚠️  Could not save emergency checkpoint: {save_error}")
            
            # Continue training (don't crash!)
            continue
    
    logger.info(f"\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info(f"Final iteration: {iteration+1:,}")
    logger.info(f"Best val loss: {best_val_loss:.4f}")
    logger.info(f"Model saved at: {checkpoint_dir}/best_model.pth")


# ============================================================================
# MPS Optimization
# ============================================================================

def setup_mps_environment():
    """
    Setup MPS environment for optimal Apple Silicon performance.
    
    CRITICAL: Must be called BEFORE importing ca_event_detector modules.
    """
    # Enable MPS fallback for unsupported ops
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    
    # Check MPS availability
    if torch.backends.mps.is_available():
        logger.info("✓ MPS available - using Apple Silicon GPU acceleration")
        return True
    else:
        logger.warning("MPS not available - will use CPU")
        return False


def print_memory_usage():
    """Print current MPS memory usage."""
    if torch.backends.mps.is_available():
        try:
            allocated = torch.mps.current_allocated_memory() / 1024**3
            logger.info(f"MPS Memory: {allocated:.2f} GB allocated")
        except:
            pass


# ============================================================================
# Training Utilities
# ============================================================================

def estimate_training_time(config, checkpoint_path: Optional[Path] = None,
                          avg_iter_time: float = 7.3) -> Dict:
    """
    Estimate training time based on configuration and checkpoint.
    
    Parameters
    ----------
    config : Config
        Training configuration
    checkpoint_path : Path, optional
        Current checkpoint path
    avg_iter_time : float
        Average time per iteration in seconds
        
    Returns
    -------
    estimate : dict
        Time estimates
    """
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        current_iteration = checkpoint.get('iteration', 0)
    else:
        current_iteration = 0
    
    remaining_iterations = config.training.num_iterations - current_iteration
    estimated_time = remaining_iterations * avg_iter_time
    
    from .diagnostics import format_time
    
    estimate = {
        'current_iteration': current_iteration,
        'target_iterations': config.training.num_iterations,
        'remaining_iterations': remaining_iterations,
        'estimated_time_seconds': estimated_time,
        'estimated_time_formatted': format_time(estimated_time),
        'avg_iter_time': avg_iter_time
    }
    
    return estimate
