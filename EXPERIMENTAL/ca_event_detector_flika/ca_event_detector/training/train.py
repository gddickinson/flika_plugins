"""
Training script for calcium event detection model.

UPDATED VERSION - Matches paper configuration:
- Fixed ignore_index bug (4 instead of -100)
- Learning rate scheduler support
- Better logging and monitoring
- Validation every 1000 iterations
"""

import os
import torch
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import numpy as np
import argparse
import json
import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, AttributeError) as e:
    TENSORBOARD_AVAILABLE = False
    print(f"Warning: TensorBoard not available ({e}). Training will proceed without TensorBoard logging.")

from ca_event_detector.models.unet3d import UNet3D
from ca_event_detector.training.losses import get_loss_function
from ca_event_detector.data.dataset import create_dataloaders
from ca_event_detector.configs.config import Config


class MetricsLogger:
    """Simple JSON-based metrics logger."""

    def __init__(self, log_path: str):
        self.log_path = Path(log_path)
        self.metrics = []

    def log(self, iteration: int, metrics: dict):
        """Log metrics for an iteration."""
        entry = {'iteration': iteration, 'timestamp': time.time()}
        entry.update(metrics)
        self.metrics.append(entry)

    def save(self):
        """Save metrics to JSON file."""
        with open(self.log_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)


class Trainer:
    """
    Trainer class for calcium event detection model.

    UPDATED with paper-matched configuration.
    """

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.training.device)

        # Set MPS fallback if using MPS device
        if self.device.type == 'mps':
            os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
            print(f"Using Apple MPS (Metal Performance Shaders)")
            print(f"Note: Some operations may fall back to CPU")
        elif self.device.type == 'cuda':
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"Using CPU (training will be slow)")

        # Create model
        self.model = UNet3D(
            in_channels=config.model.in_channels,
            num_classes=config.model.num_classes,
            base_channels=config.model.base_channels
        ).to(self.device)

        # Print model info
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {num_params:,}")
        print(f"Base channels: {config.model.base_channels}")

        # Load pretrained weights if specified
        if config.training.pretrained_weights:
            self.load_pretrained_weights(config.training.pretrained_weights)

        # Create loss function with CORRECT ignore_index
        # CRITICAL FIX: Use ignore_index=4 to properly ignore class 4 (undefined/artifact events)
        self.criterion = get_loss_function(
            loss_type=config.training.loss_type,
            ignore_index=4  # Ignore class 4 (undefined/artifact class in ground truth masks)
        )

        # Create optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate
        )

        # Create learning rate scheduler (NEW - from paper)
        self.scheduler = None
        if config.training.use_scheduler and config.training.scheduler_type == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.training.scheduler_step_size,
                gamma=config.training.scheduler_gamma
            )
            print(f"Using StepLR scheduler: step_size={config.training.scheduler_step_size}, gamma={config.training.scheduler_gamma}")

        # Training state
        self.current_iteration = 0
        self.best_val_loss = float('inf')

        # Create directories
        self.checkpoint_dir = Path(config.training.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Create tensorboard writer if available
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=self.checkpoint_dir / 'logs')
        else:
            self.writer = None

        # Create metrics logger (NEW)
        self.metrics_logger = MetricsLogger(self.checkpoint_dir / 'metrics.json')

        # Resume from checkpoint if specified
        if config.training.resume_from:
            self.load_checkpoint(config.training.resume_from)

    def load_pretrained_weights(self, weights_path: str):
        """Load pretrained model weights."""
        print(f"Loading pretrained weights from {weights_path}")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load weights
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("Successfully loaded all pretrained weights")
        except RuntimeError as e:
            print(f"Warning: Could not load all weights strictly. Error: {e}")
            print("Attempting to load compatible weights...")

            # Try loading with strict=False
            missing_keys, unexpected_keys = self.model.load_state_dict(
                state_dict, strict=False
            )

            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")

    def save_checkpoint(self, filename: str = None):
        """Save training checkpoint."""
        if filename is None:
            filename = f"checkpoint_iter_{self.current_iteration}.pth"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint = {
            'iteration': self.current_iteration,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save scheduler state if using scheduler
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Also save as 'latest.pth'
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)

        # Save metrics
        self.metrics_logger.save()

    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint to resume training."""
        print(f"Loading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_iteration = checkpoint.get('iteration', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        # Load scheduler state if present
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Resumed from iteration {self.current_iteration}")

    def train_iteration(self, images, masks):
        """
        Perform a single training iteration.

        Args:
            images: (B, C, T, H, W) input images
            masks: (B, T, H, W) ground truth masks

        Returns:
            loss value
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        images = images.to(self.device)
        masks = masks.to(self.device)

        logits = self.model(images)

        # Ignore first and last temporal_context frames
        tc = self.config.data.temporal_context
        if tc > 0:
            logits = logits[:, :, tc:-tc]
            masks = masks[:, tc:-tc]

        # Compute loss
        loss = self.criterion(logits, masks)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def validate(self, val_loader):
        """
        Run validation.

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(self.device)
                masks = masks.to(self.device)

                logits = self.model(images)

                # Ignore first and last temporal_context frames
                tc = self.config.data.temporal_context
                if tc > 0:
                    logits = logits[:, :, tc:-tc]
                    masks = masks[:, tc:-tc]

                loss = self.criterion(logits, masks)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def train(self, train_loader, val_loader):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"\n{'='*60}")
        print(f"Starting training from iteration {self.current_iteration}")
        print(f"Target iterations: {self.config.training.num_iterations}")
        print(f"Device: {self.device}")
        print(f"Base channels: {self.config.model.base_channels}")
        print(f"Augmentation: {self.config.data.augmentation_mode}")
        print(f"Scheduler: {'Yes' if self.scheduler else 'No'}")
        print(f"{'='*60}\n")

        train_iter = iter(train_loader)
        pbar = tqdm(total=self.config.training.num_iterations - self.current_iteration,
                   initial=self.current_iteration)

        while self.current_iteration < self.config.training.num_iterations:
            try:
                images, masks, _ = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                images, masks, _ = next(train_iter)

            # Train iteration
            loss = self.train_iteration(images, masks)

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss:.4f}', 'lr': f'{current_lr:.2e}'})
            pbar.update(1)

            # Logging
            if self.current_iteration % self.config.training.log_frequency == 0:
                if self.writer is not None:
                    self.writer.add_scalar('train/loss', loss, self.current_iteration)
                    self.writer.add_scalar('train/learning_rate', current_lr, self.current_iteration)

                # Log to JSON
                self.metrics_logger.log(self.current_iteration, {
                    'train_loss': loss,
                    'learning_rate': current_lr
                })

            # Validation (NEW - every val_frequency iterations)
            if self.current_iteration % self.config.training.val_frequency == 0:
                print(f"\n[Iteration {self.current_iteration}] Running validation...")
                val_loss = self.validate(val_loader)

                if self.writer is not None:
                    self.writer.add_scalar('val/loss', val_loss, self.current_iteration)

                # Log to JSON
                self.metrics_logger.log(self.current_iteration, {
                    'val_loss': val_loss
                })

                print(f"Validation loss: {val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pth')
                    print(f"New best model saved! (loss: {val_loss:.4f})")

            # Checkpointing
            if self.current_iteration % self.config.training.save_frequency == 0:
                self.save_checkpoint()

                # Clear cache for MPS to prevent memory leaks
                if self.device.type == 'mps':
                    torch.mps.empty_cache()

            # Step scheduler if using one
            if self.scheduler is not None:
                self.scheduler.step()

            self.current_iteration += 1

        pbar.close()

        # Final validation
        print("\nRunning final validation...")
        val_loss = self.validate(val_loader)
        print(f"Final validation loss: {val_loss:.4f}")

        # Save final checkpoint
        self.save_checkpoint('final_model.pth')

        print("\nTraining completed!")
        if self.writer is not None:
            self.writer.close()


def main():
    """Main training function."""
    import sys
    from pathlib import Path

    # Add package to path if running as script
    script_dir = Path(__file__).parent.absolute()
    package_root = script_dir.parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

    parser = argparse.ArgumentParser(description='Train calcium event detection model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data_dir', type=str, help='Path to data directory')
    parser.add_argument('--pretrained', type=str, help='Path to pretrained weights')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, help='Directory to save checkpoints')

    args = parser.parse_args()

    # Load config
    if args.config:
        config = Config.load(args.config)
    else:
        config = Config()

    # Override config with command-line arguments
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.pretrained:
        config.training.pretrained_weights = args.pretrained
    if args.resume:
        config.training.resume_from = args.resume
    if args.save_dir:
        config.training.save_dir = args.save_dir

    # Save config
    config_save_path = Path(config.training.save_dir) / 'config.json'
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(str(config_save_path))
    print(f"Configuration saved to {config_save_path}")

    # Create data loaders
    print("Loading dataset...")
    train_loader, val_loader = create_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    # Create trainer and train
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
