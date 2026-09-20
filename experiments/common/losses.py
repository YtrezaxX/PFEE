"""
Loss functions for grain segmentation experiments.
Includes MSE, Dice, Boundary, Focal, and combined losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import ndimage


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    
    Better than MSE for:
    - Handling class imbalance (lots of background)
    - Learning complete grain shapes
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1 - dice


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss that penalizes errors near grain boundaries.
    
    Crucial for separating adjacent grains in watershed.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        # Compute base MSE
        mse = self.mse(pred, target)
        
        # Compute boundary weights from target
        # Higher weight near boundaries (where gradient is high)
        with torch.no_grad():
            # Use gradient magnitude as boundary indicator
            target_np = target.cpu().numpy()
            weights = np.zeros_like(target_np)
            
            for i in range(target_np.shape[0]):
                for c in range(target_np.shape[1]):
                    # Compute gradient magnitude
                    gz = np.abs(np.diff(target_np[i, c], axis=0, prepend=0))
                    gy = np.abs(np.diff(target_np[i, c], axis=1, prepend=0))
                    gx = np.abs(np.diff(target_np[i, c], axis=2, prepend=0))
                    gradient_mag = gz + gy + gx
                    
                    # Normalize and add base weight
                    gradient_mag = gradient_mag / (gradient_mag.max() + 1e-6)
                    weights[i, c] = 1.0 + 2.0 * gradient_mag  # Higher weight at boundaries
            
            weights = torch.from_numpy(weights).to(pred.device)
        
        # Weighted MSE
        weighted_mse = (mse * weights).mean()
        return weighted_mse


class FocalLoss(nn.Module):
    """
    Focal Loss for handling hard examples (small grains).
    
    FL = -α * (1 - p)^γ * log(p)
    
    Reduces loss for easy examples, focuses on hard ones.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # Clamp predictions for numerical stability
        pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
        
        # Compute focal weights
        pt = torch.where(target > 0.5, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        
        # Compute cross-entropy like loss
        bce = -target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
        
        # Apply focal weighting
        focal_loss = self.alpha * focal_weight * bce
        
        return focal_loss.mean()


class GradientLoss(nn.Module):
    """
    Gradient loss to preserve spatial structure.
    From the original implementation.
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target):
        # Compute gradients along each axis
        grad_pred_z = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        grad_target_z = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        
        grad_pred_y = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        grad_target_y = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        
        grad_pred_x = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        grad_target_x = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        
        loss = (self.mse(grad_pred_z, grad_target_z) + 
                self.mse(grad_pred_y, grad_target_y) + 
                self.mse(grad_pred_x, grad_target_x)) / 3
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with configurable weights.
    
    Default (baseline): 70% MSE + 30% Gradient
    Recommended: 40% Dice + 30% MSE + 30% Boundary
    """
    def __init__(self, loss_type='mse_gradient', 
                 mse_weight=0.7, gradient_weight=0.3,
                 dice_weight=0.4, boundary_weight=0.3):
        super().__init__()
        self.loss_type = loss_type
        self.mse_weight = mse_weight
        self.gradient_weight = gradient_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        
        self.mse = nn.MSELoss()
        self.gradient = GradientLoss()
        self.dice = DiceLoss()
        self.boundary = BoundaryLoss()
    
    def forward(self, pred, target):
        if self.loss_type == 'mse_gradient':
            # Original baseline loss
            return self.mse_weight * self.mse(pred, target) + \
                   self.gradient_weight * self.gradient(pred, target)
        
        elif self.loss_type == 'dice_only':
            return self.dice(pred, target)
        
        elif self.loss_type == 'dice_mse':
            return 0.5 * self.dice(pred, target) + 0.5 * self.mse(pred, target)
        
        elif self.loss_type == 'dice_boundary':
            return self.dice_weight * self.dice(pred, target) + \
                   self.mse_weight * self.mse(pred, target) + \
                   self.boundary_weight * self.boundary(pred, target)
        
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
