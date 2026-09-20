#!/usr/bin/env python3
"""
Experiment 010: Train Attention U-Net on 208 CLEAN images.

Split: 166 train (80%) / 42 test (20%)
Uses DataLoader with num_workers for CPU parallelism.
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
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.unet_attention import AttentionUNet3D


class LargeCleanDataset(Dataset):
    """
    Dataset that loads volumes on-demand for memory efficiency.
    Extracts random patches during training.
    """
    
    def __init__(self, image_files, gt_files, patch_size=64, patches_per_volume=30, 
                 preload=False):
        self.image_files = image_files
        self.gt_files = gt_files
        self.patch_size = patch_size
        self.patches_per_volume = patches_per_volume
        self.preload = preload
        
        # Optionally preload all volumes into RAM (faster but memory heavy)
        if preload:
            print("Preloading all volumes into RAM...")
            self.images = []
            self.gts = []
            for img_f, gt_f in tqdm(zip(image_files, gt_files), total=len(image_files)):
                self.images.append(tifffile.imread(img_f).astype(np.float32) / 65535.0)
                self.gts.append(tifffile.imread(gt_f).astype(np.float32) / 65535.0)
        
        print(f"Dataset: {len(image_files)} volumes, {len(self)} total patches")
    
    def __len__(self):
        return len(self.image_files) * self.patches_per_volume
    
    def __getitem__(self, idx):
        vol_idx = idx // self.patches_per_volume
        
        if self.preload:
            img = self.images[vol_idx]
            gt = self.gts[vol_idx]
        else:
            img = tifffile.imread(self.image_files[vol_idx]).astype(np.float32) / 65535.0
            gt = tifffile.imread(self.gt_files[vol_idx]).astype(np.float32) / 65535.0
        
        # Random patch extraction
        ps = self.patch_size
        max_z = max(1, img.shape[0] - ps)
        max_y = max(1, img.shape[1] - ps)
        max_x = max(1, img.shape[2] - ps)
        
        z = np.random.randint(0, max_z)
        y = np.random.randint(0, max_y)
        x = np.random.randint(0, max_x)
        
        img_patch = img[z:z+ps, y:y+ps, x:x+ps].copy()
        gt_patch = gt[z:z+ps, y:y+ps, x:x+ps].copy()
        
        # Random augmentation on-the-fly
        if np.random.rand() > 0.5:
            axis = np.random.randint(0, 3)
            img_patch = np.flip(img_patch, axis=axis).copy()
            gt_patch = np.flip(gt_patch, axis=axis).copy()
        
        # Add channel dimension
        img_patch = img_patch[np.newaxis, ...]
        gt_patch = gt_patch[np.newaxis, ...]
        
        return torch.from_numpy(img_patch), torch.from_numpy(gt_patch)


class CombinedLoss(nn.Module):
    """MSE + Gradient Loss for sharp boundaries."""
    
    def __init__(self, mse_weight=0.7, grad_weight=0.3):
        super().__init__()
        self.mse_weight = mse_weight
        self.grad_weight = grad_weight
        self.mse = nn.MSELoss()
    
    def gradient_loss(self, pred, target):
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


def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_img, batch_gt in pbar:
        batch_img = batch_img.to(device)
        batch_gt = batch_gt.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast():
            pred = model(batch_img)
            loss = criterion(pred, batch_gt)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_img, batch_gt in loader:
            batch_img = batch_img.to(device)
            batch_gt = batch_gt.to(device)
            
            with torch.cuda.amp.autocast():
                pred = model(batch_img)
                loss = criterion(pred, batch_gt)
            
            total_loss += loss.item()
    
    return total_loss / len(loader)


def main():
    print("=" * 70)
    print("EXPERIMENT 010: Training on 208 Clean Images")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Data paths
    img_dir = '/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/dataset/images/'
    gt_dir = '/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/dataset/ground_truth/'
    
    img_files = sorted(glob(img_dir + '*.tif'))
    gt_files = sorted(glob(gt_dir + '*.tif'))
    
    print(f"\nFound {len(img_files)} images, {len(gt_files)} ground truth")
    
    # Shuffle and split (80/20)
    np.random.seed(42)
    indices = np.random.permutation(len(img_files))
    split_idx = int(len(indices) * 0.8)
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    train_img = [img_files[i] for i in train_idx]
    train_gt = [gt_files[i] for i in train_idx]
    test_img = [img_files[i] for i in test_idx]
    test_gt = [gt_files[i] for i in test_idx]
    
    print(f"Training: {len(train_img)} volumes")
    print(f"Testing: {len(test_img)} volumes")
    
    # Save split info
    split_info = {
        'train_files': [os.path.basename(f) for f in train_img],
        'test_files': [os.path.basename(f) for f in test_img]
    }
    with open('/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/results/split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # Dataset & DataLoader
    print("\nCreating datasets...")
    train_dataset = LargeCleanDataset(train_img, train_gt, patch_size=64, 
                                       patches_per_volume=30, preload=True)
    val_dataset = LargeCleanDataset(test_img[:10], test_gt[:10], patch_size=64, 
                                     patches_per_volume=10, preload=True)
    
    # Use multiple workers for data loading
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, 
                              num_workers=8, pin_memory=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    print(f"Batches per epoch: {len(train_loader)}")
    
    # Model
    print("\nInitializing model...")
    model = AttentionUNet3D(in_channels=1, out_channels=1, init_features=48).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    criterion = CombinedLoss(mse_weight=0.7, grad_weight=0.3)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision
    
    # Training loop
    num_epochs = 50
    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_loss = float('inf')
    
    print(f"\nStarting training ({num_epochs} epochs)...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss = validate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, '/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/checkpoints/best_model.pth')
            print(f"  → Saved best model (val_loss: {best_loss:.5f})")
    
    total_time = time.time() - start_time
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_loss
    }, '/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/checkpoints/final_model.pth')
    
    # Save history
    with open('/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/results/training_history.json', 'w') as f:
        json.dump(history, f)
    
    # Plot learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Val Loss', color='orange')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['lr'], color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Exp 010: Training on 208 Clean Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/results/learning_curves.png', 
                dpi=150, bbox_inches='tight')
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation loss: {best_loss:.5f}")
    print(f"Final training loss: {train_loss:.5f}")


if __name__ == '__main__':
    main()
