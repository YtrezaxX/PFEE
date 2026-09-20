"""
Training script for exp_002_dice_loss.
Tests Dice Loss instead of MSE+Gradient.
"""

import os
import sys
import time
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.unet_baseline import UNet3D
from common.datasets import GrainDistanceDataset
from common.losses import CombinedLoss
from common.utils import set_seed, save_checkpoint, get_device, count_parameters, AverageMeter


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        loss.backward()
        optimizer.step()
        
        loss_meter.update(loss.item(), images.size(0))
    
    return loss_meter.avg


def validate(model, dataloader, criterion, device):
    model.eval()
    loss_meter = AverageMeter()
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            loss_meter.update(loss.item(), images.size(0))
    
    return loss_meter.avg


def main():
    config = load_config()
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Description: {config['experiment']['description']}")
    
    set_seed(config['seed'])
    device = get_device()
    
    # Find data files
    data_dir = '/home/gma/epita/PFEE/data_yukiko/dataset'
    image_files = sorted(glob(os.path.join(data_dir, 'default', '*.tif')))
    distance_files = sorted(glob(os.path.join(data_dir, 'ground_truth', '*.tif')))
    
    if not image_files:
        print("ERROR: Dataset not found!")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Create dataset
    dataset = GrainDistanceDataset(
        image_files=image_files,
        distance_files=distance_files,
        patch_size=config['data']['patch_size'],
        num_patches_per_volume=config['data']['num_patches_per_volume']
    )
    
    # Split
    train_size = int(len(dataset) * config['data']['train_split'])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              shuffle=True, num_workers=config['hardware']['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=config['hardware']['num_workers'], pin_memory=True)
    
    # Model
    model = UNet3D(in_channels=1, out_channels=1, init_features=config['model']['init_features']).to(device)
    print(f"Parameters: {count_parameters(model):,}")
    
    # Loss - Dice + MSE
    criterion = CombinedLoss(loss_type=config['loss']['type'])
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'],
                      weight_decay=config['training']['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['training']['epochs'])
    
    # Training
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    exp_dir = os.path.dirname(__file__)
    checkpoint_dir = os.path.join(exp_dir, 'checkpoints')
    results_dir = os.path.join(exp_dir, 'results')
    
    print("\nStarting training...")
    start_time = time.time()
    
    for epoch in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch, val_loss,
                          os.path.join(checkpoint_dir, 'best_model.pth'), scheduler)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['training']['epochs']} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Time: {(time.time()-start_time)/60:.1f}min")
    
    save_checkpoint(model, optimizer, config['training']['epochs'], val_loss,
                   os.path.join(checkpoint_dir, 'final_model.pth'), scheduler)
    
    # Save history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'training_time': time.time() - start_time
    }
    with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")


if __name__ == '__main__':
    main()
