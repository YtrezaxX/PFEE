"""
Evaluation script for exp_001_baseline.

Evaluates the trained model and computes metrics.
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import tifffile
from glob import glob

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.unet_baseline import UNet3D
from common.metrics import compute_metrics, watershed_segmentation
from common.visualization import (
    plot_training_curves, plot_prediction_comparison,
    plot_segmentation_result, plot_grain_statistics
)
from common.utils import get_device


def load_config():
    """Load experiment configuration."""
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def predict_full_volume(model, image, patch_size=64, stride=32, device='cuda'):
    """
    Predict on full volume using sliding window.
    
    Args:
        model: Trained model
        image: 3D input image (D, H, W)
        patch_size: Size of patches
        stride: Stride between patches (use < patch_size for overlap)
        device: Compute device
    
    Returns:
        Full prediction volume
    """
    model.eval()
    d, h, w = image.shape
    
    # Pad image if needed
    pad_d = (patch_size - d % patch_size) % patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    image_padded = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')
    
    # Output and count arrays for averaging overlaps
    output = np.zeros_like(image_padded, dtype=np.float32)
    count = np.zeros_like(image_padded, dtype=np.float32)
    
    # Sliding window prediction
    for z in range(0, image_padded.shape[0] - patch_size + 1, stride):
        for y in range(0, image_padded.shape[1] - patch_size + 1, stride):
            for x in range(0, image_padded.shape[2] - patch_size + 1, stride):
                # Extract patch
                patch = image_padded[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                
                # Predict
                with torch.no_grad():
                    patch_tensor = torch.from_numpy(patch[np.newaxis, np.newaxis, ...]).float().to(device)
                    pred = model(patch_tensor).cpu().numpy()[0, 0]
                
                # Accumulate
                output[z:z+patch_size, y:y+patch_size, x:x+patch_size] += pred
                count[z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1
    
    # Average overlapping predictions
    output = output / np.maximum(count, 1)
    
    # Remove padding
    output = output[:d, :h, :w]
    
    return output


def main():
    # Load config
    config = load_config()
    print(f"Evaluating: {config['experiment']['name']}")
    
    device = get_device()
    
    exp_dir = os.path.dirname(__file__)
    results_dir = os.path.join(exp_dir, 'results')
    checkpoint_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pth')
    
    # Check if model exists
    if not os.path.exists(checkpoint_path):
        print("ERROR: No trained model found!")
        print("Please run train.py first.")
        return
    
    # Load model
    model = UNet3D(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels'],
        init_features=config['model']['init_features']
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Plot training curves
    history_path = os.path.join(results_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        plot_training_curves(
            history['train_losses'],
            history['val_losses'],
            save_path=os.path.join(results_dir, 'learning_curves.png')
        )
        print("Saved learning curves")
    
    # Find data files
    data_dir = '/home/gma/epita/PFEE/data_yukiko/dataset'
    image_files = sorted(glob(os.path.join(data_dir, 'default', '*.tif')))
    distance_files = sorted(glob(os.path.join(data_dir, 'ground_truth', '*.tif')))
    
    if not image_files:
        print("ERROR: No images found!")
        return
    
    # Evaluate on first volume
    print("\nEvaluating on first volume...")
    
    image = tifffile.imread(image_files[0]).astype(np.float32)
    image = image / 65535.0 if image.max() > 1 else image
    
    gt_distance = tifffile.imread(distance_files[0]).astype(np.float32)
    gt_distance = gt_distance / 65535.0 if gt_distance.max() > 1 else gt_distance
    
    # Predict
    print("Predicting full volume...")
    pred_distance = predict_full_volume(
        model, image,
        patch_size=config['data']['patch_size'],
        stride=32,
        device=device
    )
    
    # Compute metrics
    print("Computing metrics...")
    metrics = compute_metrics(pred_distance, gt_distance)
    
    print("\n=== Results ===")
    print(f"Distance Map Metrics:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"\nSegmentation Metrics:")
    print(f"  Predicted grains: {metrics['pred_grains']}")
    print(f"  Ground truth grains: {metrics['gt_grains']}")
    print(f"  Grain count accuracy: {metrics['grain_count_accuracy']*100:.1f}%")
    print(f"  IoU: {metrics['iou']:.4f}")
    print(f"  Dice: {metrics['dice']:.4f}")
    
    # Save metrics
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Prediction comparison
    plot_prediction_comparison(
        image, gt_distance, pred_distance,
        save_path=os.path.join(results_dir, 'prediction_comparison.png')
    )
    
    # Segmentation result
    labels, num_grains = watershed_segmentation(pred_distance)
    gt_labels, gt_num_grains = watershed_segmentation(gt_distance)
    
    plot_segmentation_result(
        image, labels, gt_labels,
        save_path=os.path.join(results_dir, 'segmentation_result.png')
    )
    
    # Grain statistics
    plot_grain_statistics(
        labels,
        save_path=os.path.join(results_dir, 'grain_statistics.png')
    )
    
    print(f"\nResults saved to {results_dir}")


if __name__ == '__main__':
    main()
