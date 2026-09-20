"""
Residual U-Net 3D Model.

Uses residual blocks instead of plain convolutional blocks.
Residual connections help with gradient flow in deeper networks.

Reference: He et al., "Deep Residual Learning for Image Recognition"
"""

import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """
    Residual Block for 3D volumes.
    
    output = F(x) + x
    
    Where F(x) is: Conv -> BN -> ReLU -> Conv -> BN
    
    Benefits:
    - Better gradient flow (avoids vanishing gradient)
    - Easier to train deeper networks
    - Learn residual mappings instead of direct mappings
    """
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Skip connection (1x1 conv if channels change)
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity  # Residual connection
        out = self.relu(out)
        
        return out


class ResUNet3D(nn.Module):
    """
    3D U-Net with Residual Blocks.
    
    Replaces standard conv blocks with residual blocks for:
    - Better gradient flow during training
    - Faster convergence
    - Ability to train deeper networks
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        init_features: Initial number of features
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(ResUNet3D, self).__init__()
        
        features = init_features
        
        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels, features, 3, padding=1)
        
        # Encoder
        self.encoder1 = ResidualBlock3D(features, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = ResidualBlock3D(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = ResidualBlock3D(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock3D(features * 4, features * 8)
        
        # Decoder
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, 2, 2)
        self.decoder3 = ResidualBlock3D(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, 2, 2)
        self.decoder2 = ResidualBlock3D(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, 2, 2)
        self.decoder1 = ResidualBlock3D(features * 2, features)
        
        # Output
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with skip connections
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))


class ResUNet3DDeep(nn.Module):
    """
    Deeper Residual U-Net with 4 levels and double residual blocks.
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(ResUNet3DDeep, self).__init__()
        
        features = init_features
        
        # Initial convolution
        self.init_conv = nn.Conv3d(in_channels, features, 3, padding=1)
        
        # Encoder (4 levels, each with 2 residual blocks)
        self.encoder1 = nn.Sequential(
            ResidualBlock3D(features, features),
            ResidualBlock3D(features, features)
        )
        self.pool1 = nn.MaxPool3d(2)
        
        self.encoder2 = nn.Sequential(
            ResidualBlock3D(features, features * 2),
            ResidualBlock3D(features * 2, features * 2)
        )
        self.pool2 = nn.MaxPool3d(2)
        
        self.encoder3 = nn.Sequential(
            ResidualBlock3D(features * 2, features * 4),
            ResidualBlock3D(features * 4, features * 4)
        )
        self.pool3 = nn.MaxPool3d(2)
        
        self.encoder4 = nn.Sequential(
            ResidualBlock3D(features * 4, features * 8),
            ResidualBlock3D(features * 8, features * 8)
        )
        self.pool4 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResidualBlock3D(features * 8, features * 16),
            ResidualBlock3D(features * 16, features * 16)
        )
        
        # Decoder
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, 2, 2)
        self.decoder4 = ResidualBlock3D(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, 2, 2)
        self.decoder3 = ResidualBlock3D(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, 2, 2)
        self.decoder2 = ResidualBlock3D(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, 2, 2)
        self.decoder1 = ResidualBlock3D(features * 2, features)
        
        # Output
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.init_conv(x)
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))
