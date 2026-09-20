#!/usr/bin/env python3
"""
Compute Dice coefficient on UNSEEN data.
"""

import sys
sys.path.insert(0, '/home/gma/epita/PFEE/experiments')

import numpy as np
import torch
import tifffile
import json
from glob import glob
from skimage.segmentation import watershed
from skimage.morphology import h_maxima
from tqdm import tqdm
import os

from models.unet_attention import AttentionUNet3D
from common.evaluate_experiment import predict_full_volume


ALL_BASE_FILES = [
    "triax.-1.eps=0.spheres", "triax.0.eps=-0.0025.spheres", "triax.0.eps=-0.1.spheres",
    "triax.1.eps=-0.005.spheres", "triax.2.eps=-0.01.spheres", "triax.3.eps=-0.02.spheres",
    "triax.4.eps=-0.03.spheres", "triax.5.eps=-0.04.spheres", "triax.6.eps=-0.05.spheres",
    "triax.7.eps=-0.06.spheres", "triax.8.eps=-0.07.spheres", "triax.9.eps=-0.08.spheres",
    "triax.10.eps=-0.09.spheres", "triax.11.eps=-0.1.spheres", "triax.12.eps=-50124.6.spheres",
    "initial-state-reference",
]

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


def compute_dice(pred_labels, gt_labels):
    """
    Compute Dice coefficient on binary segmentation (foreground vs background).
    """
    pred_binary = (pred_labels > 0).astype(np.float32)
    gt_binary = (gt_labels > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary)
    
    if union == 0:
        return 1.0  # Both empty
    
    dice = (2.0 * intersection) / union
    return dice


def compute_iou(pred_labels, gt_labels):
    """
    Compute IoU (Intersection over Union) on binary segmentation.
    """
    pred_binary = (pred_labels > 0).astype(np.float32)
    gt_binary = (gt_labels > 0).astype(np.float32)
    
    intersection = np.sum(pred_binary * gt_binary)
    union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
    
    if union == 0:
        return 1.0
    
    iou = intersection / union
    return iou


def get_base_file(filename):
    for base in ALL_BASE_FILES:
        if filename.startswith(base):
            return base
    return None


def main():
    print("=" * 70)
    print("Computing Dice & IoU on UNSEEN Data")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AttentionUNet3D(in_channels=1, out_channels=1, init_features=48).to(device)
    ckpt = torch.load('/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/checkpoints/best_model.pth')
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    img_dir = '/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/dataset/images/'
    gt_dir = '/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/dataset/ground_truth/'
    
    all_images = sorted(glob(img_dir + '*.tif'))
    
    # Get only unseen test images
    test_images = [img for img in all_images if get_base_file(os.path.basename(img)) in TEST_BASE_FILES]
    
    print(f"Evaluating on {len(test_images)} UNSEEN images...")
    
    results = []
    
    for img_path in tqdm(test_images, desc="Computing metrics"):
        img_name = os.path.basename(img_path)
        gt_name = img_name.replace('.tif', '_gt.tif')
        gt_path = gt_dir + gt_name
        
        image = tifffile.imread(img_path).astype(np.float32) / 65535.0
        gt_dist = tifffile.imread(gt_path).astype(np.float32) / 65535.0
        
        pred_dist = predict_full_volume(model, image, patch_size=64, stride=32, device=device)
        
        pred_labels, pred_num = watershed_double_threshold(pred_dist)
        gt_labels, gt_num = watershed_double_threshold(gt_dist, marker_thresh=0.1, mask_thresh=0.05, h=0.05)
        
        dice = compute_dice(pred_labels, gt_labels)
        iou = compute_iou(pred_labels, gt_labels)
        
        results.append({
            'file': img_name,
            'base_file': get_base_file(img_name),
            'dice': float(dice),
            'iou': float(iou),
            'pred_grains': pred_num,
            'gt_grains': gt_num
        })
    
    # Statistics
    dice_scores = [r['dice'] for r in results]
    iou_scores = [r['iou'] for r in results]
    
    print("\n" + "=" * 70)
    print("RESULTS (UNSEEN Data Only)")
    print("=" * 70)
    
    print(f"\n📊 Overall Metrics (n={len(results)} images):")
    print(f"  Dice Coefficient: {np.mean(dice_scores):.4f} ± {np.std(dice_scores):.4f}")
    print(f"  IoU (Jaccard):    {np.mean(iou_scores):.4f} ± {np.std(iou_scores):.4f}")
    
    print(f"\n📁 By Base File:")
    for base in TEST_BASE_FILES:
        base_results = [r for r in results if r['base_file'] == base]
        if base_results:
            avg_dice = np.mean([r['dice'] for r in base_results])
            avg_iou = np.mean([r['iou'] for r in base_results])
            short_name = base.replace('.spheres', '').replace('triax.', '')
            print(f"  {short_name:25s}: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}")
    
    # Save results
    summary = {
        'n_images': len(results),
        'dice_mean': float(np.mean(dice_scores)),
        'dice_std': float(np.std(dice_scores)),
        'iou_mean': float(np.mean(iou_scores)),
        'iou_std': float(np.std(iou_scores)),
        'by_base_file': {}
    }
    
    for base in TEST_BASE_FILES:
        base_results = [r for r in results if r['base_file'] == base]
        if base_results:
            summary['by_base_file'][base] = {
                'dice': float(np.mean([r['dice'] for r in base_results])),
                'iou': float(np.mean([r['iou'] for r in base_results]))
            }
    
    with open('/home/gma/epita/PFEE/experiments/runs/exp_010_large_clean/results/dice_iou_unseen.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved: dice_iou_unseen.json")
    print("=" * 70)


if __name__ == '__main__':
    main()
