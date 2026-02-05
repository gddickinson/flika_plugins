"""
Multi-Task Loss Functions

Implements Kendall uncertainty weighting for balancing
localization and calcium detection losses automatically.

Based on: Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
Losses for Scene Geometry and Semantics", CVPR 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationLoss(nn.Module):
    """
    Loss for puncta localization task.
    
    Components:
    - Detection loss (binary cross-entropy)
    - Offset loss (MSE for sub-pixel positions)
    - Photon loss (MSE for photon counts)
    """
    
    def __init__(self, 
                 detection_weight: float = 1.0,
                 offset_weight: float = 1.0,
                 photon_weight: float = 0.1):
        super().__init__()
        
        self.detection_weight = detection_weight
        self.offset_weight = offset_weight
        self.photon_weight = photon_weight
    
    def forward(self, pred, target):
        """
        Compute localization loss.
        
        Args:
            pred: Dict with 'prob', 'offset', 'photons'
            target: Dict with 'has_puncta', 'offset_gt', 'photons_gt'
            
        Returns:
            loss: Total localization loss
            components: Dict with individual loss components
        """
        
        # Detection loss (BCE)
        detection_loss = F.binary_cross_entropy(
            pred['prob'],
            target['has_puncta'].float()
        )
        
        # Offset loss (only where puncta exist)
        mask = target['has_puncta'] > 0.5
        if mask.sum() > 0:
            offset_diff = pred['offset'] - target['offset_gt']
            offset_loss = (offset_diff[mask]**2).mean()
        else:
            offset_loss = torch.tensor(0.0, device=pred['prob'].device)
        
        # Photon loss (only where puncta exist)
        if mask.sum() > 0:
            photon_diff = pred['photons'] - target['photons_gt']
            photon_loss = (photon_diff[mask]**2).mean()
        else:
            photon_loss = torch.tensor(0.0, device=pred['prob'].device)
        
        # Total loss
        total_loss = (
            self.detection_weight * detection_loss +
            self.offset_weight * offset_loss +
            self.photon_weight * photon_loss
        )
        
        components = {
            'detection': detection_loss.item(),
            'offset': offset_loss.item(),
            'photon': photon_loss.item()
        }
        
        return total_loss, components


class CalciumLoss(nn.Module):
    """
    Loss for calcium signal detection.
    
    Binary classification: signal vs no-signal at puncta locations.
    Uses focal loss to handle class imbalance.
    """
    
    def __init__(self, 
                 use_focal_loss: bool = True,
                 alpha: float = 0.25,
                 gamma: float = 2.0):
        super().__init__()
        
        self.use_focal_loss = use_focal_loss
        self.alpha = alpha
        self.gamma = gamma
    
    def focal_loss(self, pred_logits, target):
        """
        Focal loss for handling class imbalance.
        
        FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
        """
        # Get probabilities
        probs = F.softmax(pred_logits, dim=1)
        
        # Target probabilities
        target_probs = torch.zeros_like(probs)
        target_probs.scatter_(1, target.unsqueeze(1), 1.0)
        
        # Focal loss
        pt = (probs * target_probs).sum(dim=1)
        focal_weight = (1 - pt)**self.gamma
        
        # Cross-entropy
        ce_loss = F.cross_entropy(pred_logits, target, reduction='none')
        
        # Combined
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.mean()
    
    def forward(self, pred_logits, target):
        """
        Compute calcium detection loss.
        
        Args:
            pred_logits: (B, 2, T, H, W) class logits
            target: (B, T, H, W) class labels (0=no signal, 1=signal)
            
        Returns:
            loss: Total calcium loss
            components: Dict with metrics
        """
        
        if self.use_focal_loss:
            loss = self.focal_loss(pred_logits, target)
        else:
            loss = F.cross_entropy(pred_logits, target)
        
        # Compute metrics
        with torch.no_grad():
            preds = pred_logits.argmax(dim=1)
            accuracy = (preds == target).float().mean()
            
            # Class-specific metrics
            signal_mask = target == 1
            if signal_mask.sum() > 0:
                recall = (preds[signal_mask] == 1).float().mean()
            else:
                recall = torch.tensor(0.0)
            
            no_signal_mask = target == 0
            if no_signal_mask.sum() > 0:
                specificity = (preds[no_signal_mask] == 0).float().mean()
            else:
                specificity = torch.tensor(0.0)
        
        components = {
            'loss': loss.item(),
            'accuracy': accuracy.item(),
            'recall': recall.item(),
            'specificity': specificity.item()
        }
        
        return loss, components


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss with Kendall uncertainty weighting.
    
    Learns task-specific uncertainty parameters that automatically
    balance the localization and calcium losses.
    
    L_total = (1/2σ₁²)L_loc + (1/2σ₂²)L_ca + log(σ₁σ₂)
    
    where σ₁, σ₂ are learned noise parameters.
    """
    
    def __init__(self, 
                 use_uncertainty_weighting: bool = True,
                 initial_loc_log_var: float = 0.0,
                 initial_ca_log_var: float = 0.0):
        super().__init__()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Task-specific losses
        self.localization_loss = LocalizationLoss()
        self.calcium_loss = CalciumLoss()
        
        if use_uncertainty_weighting:
            # Learnable log-variance parameters
            self.log_var_localization = nn.Parameter(
                torch.tensor(initial_loc_log_var)
            )
            self.log_var_calcium = nn.Parameter(
                torch.tensor(initial_ca_log_var)
            )
        else:
            # Fixed weights
            self.weight_localization = 1.0
            self.weight_calcium = 1.0
    
    def forward(self, pred, target_loc, target_ca):
        """
        Compute multi-task loss.
        
        Args:
            pred: Dict with 'localization' and 'calcium' predictions
            target_loc: Localization ground truth
            target_ca: Calcium ground truth
            
        Returns:
            total_loss: Combined loss
            components: Dict with all loss components and weights
        """
        
        # Compute individual task losses
        loc_loss, loc_components = self.localization_loss(
            pred['localization'], target_loc
        )
        
        ca_loss, ca_components = self.calcium_loss(
            pred['calcium'], target_ca
        )
        
        if self.use_uncertainty_weighting:
            # Kendall uncertainty weighting
            # L = (1/2σ²)L_task + log(σ)
            
            precision_loc = torch.exp(-self.log_var_localization)
            precision_ca = torch.exp(-self.log_var_calcium)
            
            total_loss = (
                precision_loc * loc_loss + self.log_var_localization +
                precision_ca * ca_loss + self.log_var_calcium
            )
            
            # Effective weights (for monitoring)
            weight_loc = precision_loc.item()
            weight_ca = precision_ca.item()
            
        else:
            # Fixed weighting
            total_loss = (
                self.weight_localization * loc_loss +
                self.weight_calcium * ca_loss
            )
            
            weight_loc = self.weight_localization
            weight_ca = self.weight_calcium
        
        # Combine all components for monitoring
        components = {
            'total': total_loss.item(),
            'localization': loc_loss.item(),
            'calcium': ca_loss.item(),
            'weight_loc': weight_loc,
            'weight_ca': weight_ca,
            **{f'loc_{k}': v for k, v in loc_components.items()},
            **{f'ca_{k}': v for k, v in ca_components.items()}
        }
        
        if self.use_uncertainty_weighting:
            components['log_var_loc'] = self.log_var_localization.item()
            components['log_var_ca'] = self.log_var_calcium.item()
        
        return total_loss, components


# Test the losses
if __name__ == '__main__':
    # Create dummy predictions and targets
    batch_size = 2
    time_steps = 5
    height, width = 64, 64
    
    # Localization predictions
    pred_loc = {
        'prob': torch.rand(batch_size, 1, time_steps, height, width),
        'offset': torch.randn(batch_size, 2, time_steps, height, width) * 0.5,
        'photons': torch.rand(batch_size, 1, time_steps, height, width) * 1000
    }
    
    # Localization targets
    target_loc = {
        'has_puncta': torch.randint(0, 2, (batch_size, 1, time_steps, height, width)),
        'offset_gt': torch.randn(batch_size, 2, time_steps, height, width) * 0.5,
        'photons_gt': torch.rand(batch_size, 1, time_steps, height, width) * 1000
    }
    
    # Calcium predictions and targets
    pred_ca = torch.randn(batch_size, 2, time_steps, height, width)
    target_ca = torch.randint(0, 2, (batch_size, time_steps, height, width))
    
    # Test individual losses
    print("Testing individual losses:")
    
    loc_loss_fn = LocalizationLoss()
    loc_loss, loc_comp = loc_loss_fn(pred_loc, target_loc)
    print(f"Localization loss: {loc_loss.item():.4f}")
    print(f"  Components: {loc_comp}")
    
    ca_loss_fn = CalciumLoss()
    ca_loss, ca_comp = ca_loss_fn(pred_ca, target_ca)
    print(f"\nCalcium loss: {ca_loss.item():.4f}")
    print(f"  Components: {ca_comp}")
    
    # Test multi-task loss
    print("\nTesting multi-task loss:")
    
    mt_loss_fn = MultiTaskLoss(use_uncertainty_weighting=True)
    
    pred = {
        'localization': pred_loc,
        'calcium': pred_ca
    }
    
    total_loss, components = mt_loss_fn(pred, target_loc, target_ca)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Components:")
    for k, v in components.items():
        print(f"  {k}: {v:.4f}")
    
    # Test backward pass
    print("\nTesting backward pass:")
    total_loss.backward()
    print("✅ Gradients computed successfully")
    
    # Test learned weights evolution
    print("\nSimulating training (weight adaptation):")
    optimizer = torch.optim.Adam(mt_loss_fn.parameters(), lr=0.01)
    
    for step in range(10):
        optimizer.zero_grad()
        
        # Random batch
        pred_loc = {
            'prob': torch.rand(batch_size, 1, time_steps, height, width),
            'offset': torch.randn(batch_size, 2, time_steps, height, width) * 0.5,
            'photons': torch.rand(batch_size, 1, time_steps, height, width) * 1000
        }
        pred_ca = torch.randn(batch_size, 2, time_steps, height, width)
        pred = {'localization': pred_loc, 'calcium': pred_ca}
        
        total_loss, components = mt_loss_fn(pred, target_loc, target_ca)
        total_loss.backward()
        optimizer.step()
        
        if step % 3 == 0:
            print(f"Step {step}: w_loc={components['weight_loc']:.4f}, "
                  f"w_ca={components['weight_ca']:.4f}")
    
    print("\n✅ All tests passed!")
