"""
Hybrid Model Architecture

Combines PIEZO1 puncta localization with calcium signal detection
using a shared encoder and specialized decoder heads.

Architecture:
    Input (2 channels: HaloTag + Calcium)
         ↓
    Shared Encoder (from existing U-Net, 16 base channels)
         ↓
    ┌────────┴────────┐
    ↓                 ↓
Localization      Calcium
Decoder           Decoder
    ↓                 ↓
(x,y,σ,N,p)      (signal/no-signal)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class SharedEncoder(nn.Module):
    """
    Shared U-Net encoder for feature extraction.
    
    Can be initialized from your existing trained 16-channel U-Net encoder.
    """
    
    def __init__(self, in_channels: int = 2, base_channels: int = 16):
        super().__init__()
        
        # Encoder levels (matching your existing U-Net)
        self.enc1 = self._make_encoder_block(in_channels, base_channels)
        self.enc2 = self._make_encoder_block(base_channels, base_channels * 2)
        self.enc3 = self._make_encoder_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._make_encoder_block(base_channels * 4, base_channels * 8)
        
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
    
    def _make_encoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create encoder block with 2 convolutions."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning features at multiple scales.
        
        Args:
            x: (B, C, T, H, W) input
            
        Returns:
            features: Dict with keys 'enc1', 'enc2', 'enc3', 'enc4'
        """
        # Encoder path
        e1 = self.enc1(x)           # (B, 16, T, H, W)
        e2 = self.enc2(self.pool(e1))   # (B, 32, T, H/2, W/2)
        e3 = self.enc3(self.pool(e2))   # (B, 64, T, H/4, W/4)
        e4 = self.enc4(self.pool(e3))   # (B, 128, T, H/8, W/8)
        
        return {
            'enc1': e1,
            'enc2': e2,
            'enc3': e3,
            'enc4': e4
        }
    
    def load_from_unet(self, unet_checkpoint_path: str):
        """
        Load encoder weights from existing U-Net checkpoint.
        
        Args:
            unet_checkpoint_path: Path to your trained U-Net
        """
        print(f"Loading encoder weights from: {unet_checkpoint_path}")
        checkpoint = torch.load(unet_checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract encoder weights
        state_dict = checkpoint['model_state_dict']
        encoder_keys = [k for k in state_dict.keys() if k.startswith('encoder')]
        
        # Map to this encoder's keys
        new_state_dict = {}
        for key in encoder_keys:
            new_key = key.replace('encoder.', '')
            new_state_dict[new_key] = state_dict[key]
        
        # Load weights
        self.load_state_dict(new_state_dict, strict=False)
        print(f"✅ Loaded {len(new_state_dict)} encoder layers")


class LocalizationDecoder(nn.Module):
    """
    DECODE-style decoder for puncta localization.
    
    Outputs:
        - Detection probability p ∈ [0, 1]
        - Sub-pixel offsets (Δx, Δy) ∈ [-0.5, 0.5]
        - Photon count N > 0
        - Uncertainties (σx, σy, σN)
    """
    
    def __init__(self, base_channels: int = 16):
        super().__init__()
        
        # Decoder path (reverse of encoder)
        self.up1 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, 
                                      kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = self._make_decoder_block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2,
                                      kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = self._make_decoder_block(base_channels * 4, base_channels * 2)
        
        self.up3 = nn.ConvTranspose3d(base_channels * 2, base_channels,
                                      kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = self._make_decoder_block(base_channels * 2, base_channels)
        
        # Output heads
        self.prob_head = nn.Conv3d(base_channels, 1, kernel_size=1)  # Detection probability
        self.offset_head = nn.Conv3d(base_channels, 2, kernel_size=1)  # (Δx, Δy)
        self.photon_head = nn.Conv3d(base_channels, 1, kernel_size=1)  # Photon count
        self.uncertainty_head = nn.Conv3d(base_channels, 3, kernel_size=1)  # (σx, σy, σN)
    
    def _make_decoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create decoder block."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            features: Encoder features from SharedEncoder
            
        Returns:
            outputs: Dict with 'prob', 'offset', 'photons', 'uncertainty'
        """
        # Decoder path with skip connections
        d1 = self.up1(features['enc4'])
        d1 = torch.cat([d1, features['enc3']], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, features['enc2']], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, features['enc1']], dim=1)
        d3 = self.dec3(d3)
        
        # Output heads
        prob = torch.sigmoid(self.prob_head(d3))  # [0, 1]
        offset = torch.tanh(self.offset_head(d3)) * 0.5  # [-0.5, 0.5]
        photons = F.softplus(self.photon_head(d3))  # > 0
        uncertainty = F.softplus(self.uncertainty_head(d3))  # > 0
        
        return {
            'prob': prob,  # (B, 1, T, H, W)
            'offset': offset,  # (B, 2, T, H, W)
            'photons': photons,  # (B, 1, T, H, W)
            'uncertainty': uncertainty  # (B, 3, T, H, W): σx, σy, σN
        }


class CalciumDecoder(nn.Module):
    """
    Decoder for calcium signal detection at puncta locations.
    
    Outputs binary classification: signal present / not present
    """
    
    def __init__(self, base_channels: int = 16):
        super().__init__()
        
        # Lighter decoder for binary classification
        self.up1 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4,
                                      kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = self._make_decoder_block(base_channels * 8, base_channels * 4)
        
        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2,
                                      kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec2 = self._make_decoder_block(base_channels * 4, base_channels * 2)
        
        self.up3 = nn.ConvTranspose3d(base_channels * 2, base_channels,
                                      kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec3 = self._make_decoder_block(base_channels * 2, base_channels)
        
        # Binary classification head
        self.signal_head = nn.Conv3d(base_channels, 2, kernel_size=1)  # [no_signal, signal]
    
    def _make_decoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create decoder block."""
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Encoder features from SharedEncoder
            
        Returns:
            logits: (B, 2, T, H, W) class logits
        """
        # Decoder path
        d1 = self.up1(features['enc4'])
        d1 = torch.cat([d1, features['enc3']], dim=1)
        d1 = self.dec1(d1)
        
        d2 = self.up2(d1)
        d2 = torch.cat([d2, features['enc2']], dim=1)
        d2 = self.dec2(d2)
        
        d3 = self.up3(d2)
        d3 = torch.cat([d3, features['enc1']], dim=1)
        d3 = self.dec3(d3)
        
        # Classification head
        logits = self.signal_head(d3)
        
        return logits


class HybridModel(nn.Module):
    """
    Combined model for PIEZO1 localization + calcium detection.
    """
    
    def __init__(self, 
                 in_channels: int = 2,
                 base_channels: int = 16,
                 pretrained_encoder_path: str = None):
        super().__init__()
        
        # Shared encoder
        self.encoder = SharedEncoder(in_channels, base_channels)
        
        if pretrained_encoder_path:
            self.encoder.load_from_unet(pretrained_encoder_path)
        
        # Task-specific decoders
        self.localization_decoder = LocalizationDecoder(base_channels)
        self.calcium_decoder = CalciumDecoder(base_channels)
    
    def forward(self, x: torch.Tensor, 
                decode_localization: bool = True,
                decode_calcium: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (B, 2, T, H, W) dual-channel input
            decode_localization: Whether to run localization decoder
            decode_calcium: Whether to run calcium decoder
            
        Returns:
            outputs: Dict with 'localization' and 'calcium' outputs
        """
        # Shared encoder
        features = self.encoder(x)
        
        outputs = {}
        
        # Localization decoder
        if decode_localization:
            outputs['localization'] = self.localization_decoder(features)
        
        # Calcium decoder
        if decode_calcium:
            outputs['calcium'] = self.calcium_decoder(features)
        
        return outputs
    
    def freeze_encoder(self):
        """Freeze encoder weights for decoder-only training."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("🔒 Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for joint training."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("🔓 Encoder unfrozen")


# Test the model
if __name__ == '__main__':
    # Create model
    model = HybridModel(in_channels=2, base_channels=16)
    
    # Test input
    x = torch.randn(2, 2, 5, 64, 64)  # (batch=2, channels=2, time=5, H=64, W=64)
    
    # Forward pass
    outputs = model(x)
    
    print("Model architecture:")
    print(f"  Encoder params: {sum(p.numel() for p in model.encoder.parameters()):,}")
    print(f"  Loc decoder params: {sum(p.numel() for p in model.localization_decoder.parameters()):,}")
    print(f"  Ca decoder params: {sum(p.numel() for p in model.calcium_decoder.parameters()):,}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nOutput shapes:")
    print(f"  Localization prob: {outputs['localization']['prob'].shape}")
    print(f"  Localization offset: {outputs['localization']['offset'].shape}")
    print(f"  Localization photons: {outputs['localization']['photons'].shape}")
    print(f"  Localization uncertainty: {outputs['localization']['uncertainty'].shape}")
    print(f"  Calcium logits: {outputs['calcium'].shape}")
    
    print("\n✅ Model architecture validated")
