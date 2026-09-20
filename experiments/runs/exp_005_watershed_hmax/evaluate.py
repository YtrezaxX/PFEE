"""
Evaluation script for exp_005_watershed_hmax.
Uses the trained model from exp_004 with improved watershed (H-maxima).
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import tifffile
from glob import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from models.unet_attention import AttentionUNet3D
from postprocessing.watershed_improved import watershed_h_maxima, watershed_combined
from common.metrics import compute_distance_metrics, compute_iou, compute_dice
from common.visualization import (
    plot_training_curves, plot_prediction_comparison,
    plot_segmentation_result, plot_grain_statistics
)
from common.utils import get_device


def predict_full_volume(model, image, patch_size=64, stride=32, device='cuda'):
    """Predict on full volume using sliding window."""
    model.eval()
    d, h, w = image.shape
    
    pad_d = (patch_size - d % patch_size) % patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    
    image_padded = np.pad(image, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')
    
    output = np.zeros_like(image_padded, dtype=np.float32)
    count = np.zeros_like(image_padded, dtype=np.float32)
    
    for z in range(0, image_padded.shape[0] - patch_size + 1, stride):
        for y in range(0, image_padded.shape[1] - patch_size + 1, stride):
            for x in range(0, image_padded.shape[2] - patch_size + 1, stride):
                patch = image_padded[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                
                with torch.no_grad():
                    patch_tensor = torch.from_numpy(patch[np.newaxis, np.newaxis, ...]).float().to(device)
                    pred = model(patch_tensor).cpu().numpy()[0, 0]
                
                output[z:z+patch_size, y:y+patch_size, x:x+patch_size] += pred
                count[z:z+patch_size, y:y+patch_size, x:x+patch_size] += 1
    
    output = output / np.maximum(count, 1)
    return output[:d, :h, :w]


def watershed_baseline(distance_map, min_distance=3):
    """Original watershed with peak_local_max for comparison."""
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed
    
    grain_mask = distance_map > 0
    coordinates = peak_local_max(
        distance_map,
        min_distance=min_distance,
        labels=grain_mask.astype(int),
        exclude_border=False
    )
    
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[tuple(coord)] = idx + 1
    
    labels = watershed(-distance_map, markers, mask=grain_mask)
    return labels, len(coordinates)


def main():
    print("=" * 60)
    print("exp_005: Watershed avec H-maxima Transform")
    print("=" * 60)
    
    device = get_device()
    exp_dir = os.path.dirname(__file__)
    results_dir = os.path.join(exp_dir, 'results')
    
    # Load model from exp_004
    exp_004_dir = os.path.join(exp_dir, '..', 'exp_004_attention_unet')
    checkpoint_path = os.path.join(exp_004_dir, 'checkpoints', 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print("ERROR: exp_004 model not found!")
        return
    
    # Load model
    model = AttentionUNet3D(in_channels=1, out_channels=1, init_features=48).to(device)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from exp_004 (epoch {checkpoint['epoch']})")
    
    # Find data
    data_dir = '/home/gma/epita/PFEE/data_yukiko/dataset'
    image_files = sorted(glob(os.path.join(data_dir, 'default', '*.tif')))
    distance_files = sorted(glob(os.path.join(data_dir, 'ground_truth', '*.tif')))
    
    # Load first volume
    print("\nLoading test volume...")
    image = tifffile.imread(image_files[0]).astype(np.float32)
    image = image / 65535.0 if image.max() > 1 else image
    
    gt_distance = tifffile.imread(distance_files[0]).astype(np.float32)
    gt_distance = gt_distance / 65535.0 if gt_distance.max() > 1 else gt_distance
    
    # Predict distance map
    print("Predicting distance map...")
    pred_distance = predict_full_volume(model, image, patch_size=64, stride=32, device=device)
    
    # Ground truth grain count
    gt_labels, gt_count = watershed_baseline(gt_distance)
    print(f"\nGround truth grains: {gt_count}")
    
    # Compare watershed methods
    print("\n" + "=" * 60)
    print("Comparaison des methodes Watershed")
    print("=" * 60)
    
    results = {}
    
    # Method 1: Original (peak_local_max)
    print("\n1. peak_local_max (baseline):")
    labels_baseline, count_baseline = watershed_baseline(pred_distance, min_distance=3)
    acc_baseline = count_baseline / gt_count
    print(f"   Grains detectes: {count_baseline}")
    print(f"   Precision: {acc_baseline*100:.1f}%")
    results['baseline'] = {'count': count_baseline, 'accuracy': acc_baseline}
    
    # Method 2: H-maxima (h=0.03)
    print("\n2. H-maxima (h=0.03):")
    labels_h003, count_h003, _ = watershed_h_maxima(pred_distance, h=0.03, verbose=False)
    acc_h003 = count_h003 / gt_count
    print(f"   Grains detectes: {count_h003}")
    print(f"   Precision: {acc_h003*100:.1f}%")
    results['h_maxima_003'] = {'count': count_h003, 'accuracy': acc_h003}
    
    # Method 3: H-maxima (h=0.05)
    print("\n3. H-maxima (h=0.05):")
    labels_h005, count_h005, _ = watershed_h_maxima(pred_distance, h=0.05, verbose=False)
    acc_h005 = count_h005 / gt_count
    print(f"   Grains detectes: {count_h005}")
    print(f"   Precision: {acc_h005*100:.1f}%")
    results['h_maxima_005'] = {'count': count_h005, 'accuracy': acc_h005}
    
    # Method 4: H-maxima (h=0.08)
    print("\n4. H-maxima (h=0.08):")
    labels_h008, count_h008, _ = watershed_h_maxima(pred_distance, h=0.08, verbose=False)
    acc_h008 = count_h008 / gt_count
    print(f"   Grains detectes: {count_h008}")
    print(f"   Precision: {acc_h008*100:.1f}%")
    results['h_maxima_008'] = {'count': count_h008, 'accuracy': acc_h008}
    
    # Method 5: Combined (threshold + h-maxima)
    print("\n5. Combined (threshold=0.03, h=0.05):")
    labels_comb, count_comb, _ = watershed_combined(pred_distance, h=0.05, threshold=0.03, verbose=False)
    acc_comb = count_comb / gt_count
    print(f"   Grains detectes: {count_comb}")
    print(f"   Precision: {acc_comb*100:.1f}%")
    results['combined'] = {'count': count_comb, 'accuracy': acc_comb}
    
    # Find best method
    print("\n" + "=" * 60)
    best_method = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"🏆 Meilleure methode: {best_method[0]}")
    print(f"   Precision: {best_method[1]['accuracy']*100:.1f}%")
    print(f"   Grains: {best_method[1]['count']}/{gt_count}")
    
    # Compute full metrics for best method
    if best_method[0] == 'baseline':
        best_labels = labels_baseline
    elif best_method[0] == 'h_maxima_003':
        best_labels = labels_h003
    elif best_method[0] == 'h_maxima_005':
        best_labels = labels_h005
    elif best_method[0] == 'h_maxima_008':
        best_labels = labels_h008
    else:
        best_labels = labels_comb
    
    # Compute additional metrics
    pred_mask = pred_distance > 0.1
    gt_mask = gt_distance > 0.1
    iou = compute_iou(pred_mask, gt_mask)
    dice = compute_dice(pred_mask, gt_mask)
    dist_metrics = compute_distance_metrics(pred_distance, gt_distance)
    
    metrics = {
        'method': best_method[0],
        'pred_grains': best_method[1]['count'],
        'gt_grains': gt_count,
        'grain_count_accuracy': best_method[1]['accuracy'],
        'iou': iou,
        'dice': dice,
        **dist_metrics,
        'all_methods': results
    }
    
    # Save metrics
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    plot_prediction_comparison(
        image, gt_distance, pred_distance,
        save_path=os.path.join(results_dir, 'prediction_comparison.png')
    )
    
    plot_segmentation_result(
        image, best_labels, gt_labels,
        save_path=os.path.join(results_dir, 'segmentation_result.png')
    )
    
    plot_grain_statistics(best_labels, save_path=os.path.join(results_dir, 'grain_statistics.png'))
    
    print(f"\nResults saved to {results_dir}")
    
    return metrics


if __name__ == '__main__':
    main()
