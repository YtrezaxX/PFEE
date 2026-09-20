"""
Attention U-Net 3D Model.

Attention gates help the model focus on relevant features 
and suppress irrelevant ones in skip connections.

Reference: Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas"
"""

import torch
import torch.nn as nn


class AttentionGate3D(nn.Module):
    """
    Attention Gate for 3D volumes.
    
    Computes attention coefficients to weight the skip connection features.
    
    Attention(x, g) = x * σ(W_x * x + W_g * g + b)
    
    Args:
        F_g: Number of feature channels from gating signal (decoder)
        F_l: Number of feature channels from skip connection (encoder)
        F_int: Number of intermediate channels
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        
        # Transform gating signal
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        # Transform skip connection
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )
        
        # Compute attention coefficients
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        """
        Args:
            g: Gating signal from decoder (coarser level)
            x: Features from encoder (skip connection)
        
        Returns:
            Attention-weighted features
        """
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Add and apply ReLU
        psi = self.relu(g1 + x1)
        
        # Compute attention map
        psi = self.psi(psi)
        
        # Apply attention to skip connection
        return x * psi


class AttentionUNet3D(nn.Module):
    """
    3D U-Net with Attention Gates.
    
    Attention gates are added to skip connections to help the model
    focus on relevant features and suppress irrelevant background.
    
    Benefits for grain segmentation:
    - Better focus on grain centers (high distance values)
    - Suppression of background noise
    - Improved boundary detection
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        init_features: Initial number of features
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(AttentionUNet3D, self).__init__()
        
        features = init_features
        
        # Encoder
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = self._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 4, features * 8)
        
        # Decoder with attention gates
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, 2, 2)
        self.attention3 = AttentionGate3D(F_g=features * 4, F_l=features * 4, F_int=features * 2)
        self.decoder3 = self._block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, 2, 2)
        self.attention2 = AttentionGate3D(F_g=features * 2, F_l=features * 2, F_int=features)
        self.decoder2 = self._block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, 2, 2)
        self.attention1 = AttentionGate3D(F_g=features, F_l=features, F_int=features // 2)
        self.decoder1 = self._block(features * 2, features)
        
        # Output
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)
    
    @staticmethod
    def _block(in_channels, out_channels):
        """Basic convolutional block."""
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder with attention
        dec3 = self.upconv3(bottleneck)
        enc3_att = self.attention3(g=dec3, x=enc3)  # Apply attention
        dec3 = torch.cat((dec3, enc3_att), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        enc2_att = self.attention2(g=dec2, x=enc2)
        dec2 = torch.cat((dec2, enc2_att), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        enc1_att = self.attention1(g=dec1, x=enc1)
        dec1 = torch.cat((dec1, enc1_att), dim=1)
        dec1 = self.decoder1(dec1)
        
        return torch.sigmoid(self.conv(dec1))
