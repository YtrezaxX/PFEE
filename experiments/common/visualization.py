"""
Visualization functions for grain segmentation experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import os


def plot_training_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', marker='o', markersize=2)
    plt.plot(val_losses, label='Val Loss', marker='s', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_prediction_comparison(image, gt_distance, pred_distance, slice_idx=None, save_path=None):
    """
    Plot comparison of input, ground truth, and prediction.
    
    Args:
        image: Input 3D image
        gt_distance: Ground truth distance map
        pred_distance: Predicted distance map
        slice_idx: Z-slice index (default: middle)
        save_path: Path to save figure
    """
    if slice_idx is None:
        slice_idx = image.shape[0] // 2
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Input image
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth distance
    axes[1].imshow(gt_distance[slice_idx], cmap='hot')
    axes[1].set_title('Ground Truth Distance')
    axes[1].axis('off')
    
    # Predicted distance
    axes[2].imshow(pred_distance[slice_idx], cmap='hot')
    axes[2].set_title('Predicted Distance')
    axes[2].axis('off')
    
    # Error map
    error = np.abs(pred_distance[slice_idx] - gt_distance[slice_idx])
    im = axes[3].imshow(error, cmap='Reds')
    axes[3].set_title(f'Absolute Error (MAE={error.mean():.4f})')
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_segmentation_result(image, labels, gt_labels=None, slice_idx=None, save_path=None):
    """
    Plot segmentation results.
    
    Args:
        image: Input 3D image
        labels: Predicted segmentation labels
        gt_labels: Ground truth labels (optional)
        slice_idx: Z-slice index
        save_path: Path to save figure
    """
    if slice_idx is None:
        slice_idx = image.shape[0] // 2
    
    ncols = 3 if gt_labels is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    
    # Input image
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Predicted labels
    axes[1].imshow(labels[slice_idx], cmap='nipy_spectral')
    num_pred = len(np.unique(labels)) - 1  # Exclude background
    axes[1].set_title(f'Predicted ({num_pred} grains)')
    axes[1].axis('off')
    
    if gt_labels is not None:
        axes[2].imshow(gt_labels[slice_idx], cmap='nipy_spectral')
        num_gt = len(np.unique(gt_labels)) - 1
        axes[2].set_title(f'Ground Truth ({num_gt} grains)')
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_boundary_comparison(distance_map, labels, slice_idx=None, save_path=None):
    """Plot distance map with grain boundaries overlaid."""
    if slice_idx is None:
        slice_idx = distance_map.shape[0] // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distance map
    axes[0].imshow(distance_map[slice_idx], cmap='hot')
    axes[0].set_title('Distance Map')
    axes[0].axis('off')
    
    # Distance map with boundaries
    boundaries = find_boundaries(labels[slice_idx], mode='inner')
    axes[1].imshow(distance_map[slice_idx], cmap='gray')
    axes[1].imshow(boundaries, cmap='Reds', alpha=0.5)
    axes[1].set_title('Distance Map with Boundaries')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_grain_statistics(labels, save_path=None):
    """Plot grain size distribution."""
    # Compute grain volumes
    grain_ids, grain_volumes = np.unique(labels[labels > 0], return_counts=True)
    
    # Compute equivalent radii (assuming spherical)
    grain_radii = (3 * grain_volumes / (4 * np.pi)) ** (1/3)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Volume histogram
    axes[0].hist(grain_volumes, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Grain Volume (voxels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Volume Distribution (n={len(grain_volumes)})')
    axes[0].grid(True, alpha=0.3)
    
    # Radius histogram
    axes[1].hist(grain_radii, bins=50, color='coral', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Equivalent Radius (voxels)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'Radius Distribution (mean={grain_radii.mean():.2f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_metrics_comparison(metrics_dict, save_path=None):
    """
    Plot comparison of metrics across experiments.
    
    Args:
        metrics_dict: Dict of {exp_name: {metric: value}}
    """
    exp_names = list(metrics_dict.keys())
    
    # Select key metrics
    key_metrics = ['grain_count_accuracy', 'iou', 'dice', 'mae']
    available_metrics = [m for m in key_metrics if m in list(metrics_dict.values())[0]]
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=(4 * len(available_metrics), 5))
    if len(available_metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, available_metrics):
        values = [metrics_dict[exp].get(metric, 0) for exp in exp_names]
        bars = ax.bar(exp_names, values, color='steelblue', alpha=0.7)
        ax.set_ylabel(metric)
        ax.set_title(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
