#!/usr/bin/env python3
"""
Generate visualizations ONLY for unseen data (proper evaluation).
"""

import sys
sys.path.insert(0, '/home/gma/epita/PFEE/experiments')

import numpy as np
import torch
import tifffile
import json
import matplotlib.pyplot as plt
from skimage.segmentation import watershed
from skimage.morphology import h_maxima

from models.unet_attention import AttentionUNet3D
from common.evaluate_experiment import predict_full_volume


TEST_BASE_FILES = [
    "triax.3.eps=-0.02.spheres",
    "triax.7.eps=-0.06.spheres", 
    "triax.11.eps=-0.1.spheres",
    "initial-state-reference",
]


def watershed_double_threshold(distance_map, marker_thresh=0.2, mask_thresh=0.08, h=0.03):
    marker_mask = distance_map > marker_thresh
    dist_for_markers = distance_map.copy()
    dist_for_markers[~marker_mask] = 0
    
    markers_bool = h_maxima(dist_for_markers, h=h)
    coords = np.array(np.where(markers_bool)).T
    
    if len(coords) == 0:
        return np.zeros_like(distance_map, dtype=np.int32), 0
    
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for i, c in enumerate(coords):
        markers[tuple(c)] = i + 1
    
    expansion_mask = distance_map > mask_thresh
    dist_for_watershed = distance_map.copy()
    dist_for_watershed[~expansion_mask] = 0
    
    labels = watershed(-dist_for_watershed, markers, mask=expansion_mask)
    return labels, len(np.unique(labels)) - 1


def main():
    print("Generating visualizations for UNSEEN data only...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AttentionUNet3D(in_channels=1, out_channels=1, init_features=48).to(device)
    ckpt = torch.load('/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/checkpoints/best_model.pth')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    img_dir = '/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/dataset/images/'
    gt_dir = '/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/dataset/ground_truth/'
    
    with open('/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/results/proper_evaluation.json') as f:
        results = json.load(f)
    
    # =========================================================================
    # Figure 1: Grid comparison - 4 unseen base files
    # =========================================================================
    fig, axes = plt.subplots(4, 5, figsize=(25, 20))
    
    for row_idx, base_file in enumerate(TEST_BASE_FILES):
        test_files = [r for r in results if r['base_file'] == base_file]
        if not test_files:
            continue
        
        img_name = test_files[0]['file']
        gt_name = img_name.replace('.tif', '_gt.tif')
        
        image = tifffile.imread(img_dir + img_name).astype(np.float32) / 65535.0
        gt_dist = tifffile.imread(gt_dir + gt_name).astype(np.float32) / 65535.0
        pred_dist = predict_full_volume(model, image, patch_size=64, stride=32, device=device)
        
        pred_labels, pred_num = watershed_double_threshold(pred_dist)
        gt_labels, gt_num = watershed_double_threshold(gt_dist, marker_thresh=0.1, mask_thresh=0.05, h=0.05)
        
        slice_idx = image.shape[0] // 2
        f1 = test_files[0]['f1']
        
        # Short name for title
        short_name = base_file.replace('.spheres', '').replace('triax.', '')
        
        axes[row_idx, 0].imshow(image[slice_idx], cmap='gray')
        axes[row_idx, 0].set_title(f'Input (UNSEEN)\n{short_name}', fontsize=11)
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(pred_dist[slice_idx], cmap='hot', vmin=0, vmax=1)
        axes[row_idx, 1].set_title('Predicted Distance', fontsize=11)
        axes[row_idx, 1].axis('off')
        
        axes[row_idx, 2].imshow(gt_dist[slice_idx], cmap='hot', vmin=0, vmax=1)
        axes[row_idx, 2].set_title('Ground Truth Distance', fontsize=11)
        axes[row_idx, 2].axis('off')
        
        np.random.seed(42)
        n = max(len(np.unique(pred_labels)), 1)
        colors = np.random.rand(n, 3)
        colors[0] = [0, 0, 0]
        axes[row_idx, 3].imshow(colors[pred_labels[slice_idx] % n])
        axes[row_idx, 3].set_title(f'Pred Segmentation\n({pred_num} grains)', fontsize=11)
        axes[row_idx, 3].axis('off')
        
        n_gt = max(len(np.unique(gt_labels)), 1)
        colors_gt = np.random.rand(n_gt, 3)
        colors_gt[0] = [0, 0, 0]
        axes[row_idx, 4].imshow(colors_gt[gt_labels[slice_idx] % n_gt])
        axes[row_idx, 4].set_title(f'GT Segmentation\n({gt_num} grains) | F1={f1:.2%}', fontsize=11)
        axes[row_idx, 4].axis('off')
    
    plt.suptitle('Exp 010: Results on 4 UNSEEN Base Files (Never in Training)\nF1 = 99.92%', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/results/unseen_comparison.png',
                dpi=150, bbox_inches='tight')
    print("Saved: unseen_comparison.png")
    
    # =========================================================================
    # Figure 2: Single detailed comparison
    # =========================================================================
    # Use triax.7 which has 100% F1
    best_file = [r for r in results if r['base_file'] == 'triax.7.eps=-0.06.spheres'][0]
    img_name = best_file['file']
    gt_name = img_name.replace('.tif', '_gt.tif')
    
    image = tifffile.imread(img_dir + img_name).astype(np.float32) / 65535.0
    gt_dist = tifffile.imread(gt_dir + gt_name).astype(np.float32) / 65535.0
    pred_dist = predict_full_volume(model, image, patch_size=64, stride=32, device=device)
    
    pred_labels, pred_num = watershed_double_threshold(pred_dist)
    gt_labels, gt_num = watershed_double_threshold(gt_dist, marker_thresh=0.1, mask_thresh=0.05, h=0.05)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    slice_idx = image.shape[0] // 2
    
    axes[0, 0].imshow(image[slice_idx], cmap='gray')
    axes[0, 0].set_title('Input Image (UNSEEN)', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_dist[slice_idx], cmap='hot', vmin=0, vmax=1)
    axes[0, 1].set_title('Predicted Distance Map', fontsize=12)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(gt_dist[slice_idx], cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Ground Truth Distance Map', fontsize=12)
    axes[0, 2].axis('off')
    
    np.random.seed(42)
    n = max(len(np.unique(pred_labels)), 1)
    colors = np.random.rand(n, 3)
    colors[0] = [0, 0, 0]
    axes[1, 0].imshow(colors[pred_labels[slice_idx] % n])
    axes[1, 0].set_title(f'Predicted Segmentation\n({pred_num} grains detected)', fontsize=12)
    axes[1, 0].axis('off')
    
    n_gt = max(len(np.unique(gt_labels)), 1)
    colors_gt = np.random.rand(n_gt, 3)
    colors_gt[0] = [0, 0, 0]
    axes[1, 1].imshow(colors_gt[gt_labels[slice_idx] % n_gt])
    axes[1, 1].set_title(f'Ground Truth Segmentation\n({gt_num} grains)', fontsize=12)
    axes[1, 1].axis('off')
    
    # Difference
    diff = np.abs(pred_dist - gt_dist)
    axes[1, 2].imshow(diff[slice_idx], cmap='coolwarm', vmin=0, vmax=0.1)
    axes[1, 2].set_title(f'Absolute Difference\nMAE = {np.mean(diff):.5f}', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Detailed View: UNSEEN Data (triax.7.eps=-0.06)\nF1 = {best_file["f1"]:.2%}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/results/unseen_detailed.png',
                dpi=150, bbox_inches='tight')
    print("Saved: unseen_detailed.png")
    
    # =========================================================================
    # Figure 3: Multi-slice view on unseen data
    # =========================================================================
    slices = [image.shape[0]//4, image.shape[0]//2, 3*image.shape[0]//4]
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    for row_idx, slice_idx in enumerate(slices):
        axes[row_idx, 0].imshow(image[slice_idx], cmap='gray')
        axes[row_idx, 0].set_title(f'Input (slice {slice_idx})')
        axes[row_idx, 0].axis('off')
        
        axes[row_idx, 1].imshow(pred_dist[slice_idx], cmap='hot', vmin=0, vmax=1)
        axes[row_idx, 1].set_title('Predicted Distance')
        axes[row_idx, 1].axis('off')
        
        np.random.seed(42)
        axes[row_idx, 2].imshow(colors[pred_labels[slice_idx] % n])
        axes[row_idx, 2].set_title(f'Predicted ({pred_num} grains)')
        axes[row_idx, 2].axis('off')
        
        axes[row_idx, 3].imshow(colors_gt[gt_labels[slice_idx] % n_gt])
        axes[row_idx, 3].set_title(f'Ground Truth ({gt_num} grains)')
        axes[row_idx, 3].axis('off')
    
    plt.suptitle(f'Multi-Slice View on UNSEEN Data\ntriax.7.eps=-0.06 | F1 = {best_file["f1"]:.2%}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/results/unseen_multislice.png',
                dpi=150, bbox_inches='tight')
    print("Saved: unseen_multislice.png")
    
    print("\nDone! All visualizations use ONLY unseen data.")


if __name__ == '__main__':
    main()
