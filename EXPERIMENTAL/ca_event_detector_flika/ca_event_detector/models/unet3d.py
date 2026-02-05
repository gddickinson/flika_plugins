"""
3D U-Net architecture for calcium event detection.
Based on the paper: "A deep learning-based approach for efficient detection 
and classification of local Ca2+ release events in Full-Frame confocal imaging"

Modified to use Upsample + Conv3d instead of ConvTranspose3d for compatibility
with Apple MPS (Metal Performance Shaders) backend on Apple Silicon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive 3D convolution layers with batch normalization and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, 
                     padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, 
                     padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling then double conv.
    
    Uses Upsample + Conv3d instead of ConvTranspose3d for compatibility
    with Apple MPS backend. This maintains equivalent functionality while
    working on all GPU backends (CUDA, MPS, etc.).
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Replace ConvTranspose3d with Upsample + Conv3d
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch due to padding
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet3D(nn.Module):
    """
    3D U-Net for calcium event detection.
    
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        num_classes (int): Number of output classes (default: 4)
        base_channels (int): Number of channels in first layer (default: 64)
    """
    
    def __init__(self, in_channels=1, num_classes=4, base_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)
        
        # Decoder
        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)
        
        # Output layer
        self.outc = nn.Conv3d(base_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


def create_model(config):
    """
    Factory function to create model from config.
    
    Args:
        config: Configuration dictionary or object
        
    Returns:
        UNet3D model instance
    """
    return UNet3D(
        in_channels=getattr(config, 'in_channels', 1),
        num_classes=getattr(config, 'num_classes', 4),
        base_channels=getattr(config, 'base_channels', 64)
    )


if __name__ == '__main__':
    # Test the model
    model = UNet3D(in_channels=1, num_classes=4, base_channels=64)
    x = torch.randn(1, 1, 256, 128, 128)  # (B, C, T, H, W)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
