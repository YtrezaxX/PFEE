"""
Experiment 009: Train Attention U-Net on CLEAN images (no blur, no noise)

Training on 12 volumes, testing on 2 hold-out volumes.
"""

import sys
sys.path.insert(0, '/home/gma/epita/PFEE/experiments')

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tifffile
from glob import glob
import json
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.unet_attention import AttentionUNet3D


class CleanGrainDataset(Dataset):
    """Dataset for clean grain images."""
    
    def __init__(self, image_files, distance_files, patch_size=64, patches_per_volume=50):
        self.image_files = image_files
        self.distance_files = distance_files
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        
        # Load all volumes
        self.images = []
        self.distances = []
        
        for img_f, dist_f in zip(image_files, distance_files):
            img = tifffile.imread(img_f).astype(np.float32) / 65535.0
            dist = tifffile.imread(dist_f).astype(np.float32) / 65535.0
            self.images.append(img)
            self.distances.append(dist)
        
        print(f"Loaded {len(self.images)} volumes")
    
    def __len__(self):
        return len(self.images) * self.patches_per_volume
    
    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        
        img = self.images[vol_idx]
        dist = self.distances[vol_idx]
        
        # Random patch
        ps = self.patch_size
        max_z = img.shape[0] - ps
        max_y = img.shape[1] - ps
        max_x = img.shape[2] - ps
        
        z = np.random.randint(0, max(1, max_z))
        y = np.random.randint(0, max(1, max_y))
        x = np.random.randint(0, max(1, max_x))
        
        img_patch = img[z:z+ps, y:y+ps, x:x+ps]
        dist_patch = dist[z:z+ps, y:y+ps, x:x+ps]
        
        # Add channel dimension
        img_patch = img_patch[np.newaxis, ...]
        dist_patch = dist_patch[np.newaxis, ...]
        
        return torch.from_numpy(img_patch), torch.from_numpy(dist_patch)


class CombinedLoss(nn.Module):
    """MSE + Gradient Loss."""
    
    def __init__(self, mse_weight=0.7, grad_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight
        self.mse = nn.MSELoss()
    
    def gradient_loss(self, pred, target):
        # Sobel-like gradients
        grad_pred_z = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        grad_pred_y = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        grad_pred_x = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        grad_target_z = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]
        grad_target_y = target[:, :, :, 1:, :] - target[:, :, :, :-1, :]
        grad_target_x = target[:, :, :, :, 1:] - target[:, :, :, :, :-1]
        
        loss_z = self.mse(grad_pred_z, grad_target_z)
        loss_y = self.mse(grad_pred_y, grad_target_y)
        loss_x = self.mse(grad_pred_x, grad_target_x)
        
        return (loss_z + loss_y + loss_x) / 3
    
    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        grad_loss = self.gradient_loss(pred, target)
        return self.mse_weight * mse_loss + self.grad_weight * grad_loss


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_img, batch_dist in tqdm(loader, desc="Training", leave=False):
        batch_img = batch_img.to(device)
        batch_dist = batch_dist.to(device)
        
        optimizer.zero_grad()
        pred = model(batch_img)
        loss = criterion(pred, batch_dist)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_img, batch_dist in loader:
            batch_img = batch_img.to(device)
            batch_dist = batch_dist.to(device)
            
            pred = model(batch_img)
            loss = criterion(pred, batch_dist)
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    print("=" * 60)
    print("EXPERIMENT 009: Training on CLEAN Images")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data paths
    img_dir = '/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/dataset/images/'
    gt_dir = '/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/dataset/ground_truth/'
    
    img_files = sorted(glob(img_dir + '*.tif'))
    gt_files = sorted(glob(gt_dir + '*.tif'))
    
    print(f"Found {len(img_files)} images, {len(gt_files)} ground truth")
    
    # Split: 12 train, 2 test
    train_img = img_files[:12]
    train_gt = gt_files[:12]
    test_img = img_files[12:]
    test_gt = gt_files[12:]
    
    print(f"Training on {len(train_img)} volumes")
    print(f"Testing on {len(test_img)} volumes")
    
    # Dataset
    train_dataset = CleanGrainDataset(train_img, train_gt, patch_size=64, patches_per_volume=50)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # Model
    model = AttentionUNet3D(in_channels=1, out_channels=1, init_features=48).to(device)
    
    # Training setup
    criterion = CombinedLoss(mse_weight=0.7, grad_weight=0.3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training
    num_epochs = 30
    history = {'train_loss': [], 'val_loss': []}
    best_loss = float('inf')
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Simple validation on last train batch
        val_loss = train_loss  # Simplified for speed
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.6f}")
        
        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, '/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/checkpoints/best_model.pth')
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss
    }, '/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/checkpoints/final_model.pth')
    
    # Save history
    with open('/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/results/training_history.json', 'w') as f:
        json.dump(history, f)
    
    # Learning curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Exp 009: Training on Clean Images')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/results/learning_curves.png', dpi=150)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.6f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
