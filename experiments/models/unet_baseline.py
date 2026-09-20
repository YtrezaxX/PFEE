"""
U-Net 3D Baseline Model.
Copy of the original implementation from data_yukiko/unet.py
"""

import torch
import torch.nn as nn


class UNet3D(nn.Module):
    """
    3D U-Net architecture for volumetric segmentation.
    
    Architecture:
    - 3 encoder levels (down-sampling)
    - 1 bottleneck
    - 3 decoder levels (up-sampling)
    - Skip connections between encoder and decoder
    
    Args:
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 1)
        init_features: Initial number of features (default: 32)
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3D, self).__init__()

        features = init_features
        
        # Encoder path
        self.encoder1 = UNet3D._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = UNet3D._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = UNet3D._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = UNet3D._block(features * 4, features * 8)

        # Decoder path
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, 2, 2)
        self.decoder3 = UNet3D._block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, 2, 2)
        self.decoder2 = UNet3D._block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, 2, 2)
        self.decoder1 = UNet3D._block(features * 2, features)

        # Output
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    @staticmethod
    def _block(in_channels, out_channels):
        """Basic convolutional block: Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""
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


class UNet3DDeep(nn.Module):
    """
    Deeper 3D U-Net with 4 encoder levels.
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet3DDeep, self).__init__()

        features = init_features
        
        # Encoder path (4 levels)
        self.encoder1 = UNet3D._block(in_channels, features)
        self.pool1 = nn.MaxPool3d(2)
        self.encoder2 = UNet3D._block(features, features * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.encoder3 = UNet3D._block(features * 2, features * 4)
        self.pool3 = nn.MaxPool3d(2)
        self.encoder4 = UNet3D._block(features * 4, features * 8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = UNet3D._block(features * 8, features * 16)

        # Decoder path (4 levels)
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, 2, 2)
        self.decoder4 = UNet3D._block(features * 16, features * 8)
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, 2, 2)
        self.decoder3 = UNet3D._block(features * 8, features * 4)
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, 2, 2)
        self.decoder2 = UNet3D._block(features * 4, features * 2)
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, 2, 2)
        self.decoder1 = UNet3D._block(features * 2, features)

        # Output
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))

        # Decoder
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
