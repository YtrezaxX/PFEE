"""
Proper grain matching evaluation.
Matches predicted grains to GT grains and computes TP/FP/FN.
"""

import numpy as np
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.measure import regionprops


def get_grain_centers(labels):
    """Get center of mass for each grain."""
    props = regionprops(labels)
    centers = np.array([p.centroid for p in props])
    grain_ids = np.array([p.label for p in props])
    return centers, grain_ids


def match_grains(pred_labels, gt_labels, max_distance=10):
    """
    Match predicted grains to GT grains based on center distance.
    
    Args:
        pred_labels: Predicted segmentation labels
        gt_labels: Ground truth labels
        max_distance: Maximum distance (voxels) for a match
    
    Returns:
        dict with TP, FP, FN, precision, recall, F1
    """
    # Get grain centers
    pred_centers, pred_ids = get_grain_centers(pred_labels)
    gt_centers, gt_ids = get_grain_centers(gt_labels)
    
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return {
            'true_positives': 0,
            'false_positives': len(pred_ids),
            'false_negatives': len(gt_ids),
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0
        }
    
    # Compute distance matrix
    distances = cdist(pred_centers, gt_centers)
    
    # Greedy matching: for each predicted grain, find closest GT grain
    matched_gt = set()
    matched_pred = set()
    
    # Sort by distance for greedy matching
    flat_indices = np.argsort(distances.flatten())
    
    for flat_idx in flat_indices:
        pred_idx = flat_idx // len(gt_centers)
        gt_idx = flat_idx % len(gt_centers)
        
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        
        if distances[pred_idx, gt_idx] <= max_distance:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
    
    # Compute metrics
    tp = len(matched_pred)  # True positives: matched grains
    fp = len(pred_ids) - tp  # False positives: extra predicted grains
    fn = len(gt_ids) - tp  # False negatives: missed GT grains
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'pred_grains': len(pred_ids),
        'gt_grains': len(gt_ids),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'max_distance': max_distance
    }


def evaluate_segmentation(pred_labels, gt_labels, max_distance=10):
    """
    Full evaluation of segmentation quality.
    """
    # Grain matching
    matching = match_grains(pred_labels, gt_labels, max_distance)
    
    # Also compute IoU per grain (optional, more expensive)
    
    return matching


if __name__ == '__main__':
    # Test with exp_004 and exp_005
    import sys
    import os
    import torch
    import tifffile
    from glob import glob
    
    sys.path.insert(0, os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from models.unet_attention import AttentionUNet3D
    from postprocessing.watershed_baseline import watershed_segmentation
    from postprocessing.watershed_improved import watershed_h_maxima
    
    print("=" * 60)
    print("Evaluation CORRECTE avec matching des grains")
    print("=" * 60)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AttentionUNet3D(in_channels=1, out_channels=1, init_features=48).to(device)
    
    checkpoint_path = '/home/gma/epita/PFEE/experiments/runs/exp_004_attention_unet/checkpoints/best_model.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data
    data_dir = '/home/gma/epita/PFEE/data_yukiko/dataset'
    image_file = sorted(glob(os.path.join(data_dir, 'default', '*.tif')))[0]
    gt_file = sorted(glob(os.path.join(data_dir, 'ground_truth', '*.tif')))[0]
    
    image = tifffile.imread(image_file).astype(np.float32) / 65535.0
    gt_distance = tifffile.imread(gt_file).astype(np.float32) / 65535.0
    
    # Predict
    print("\nPredicting...")
    from common.evaluate_experiment import predict_full_volume
    pred_distance = predict_full_volume(model, image, patch_size=64, stride=32, device=device)
    
    # GT segmentation
    gt_labels, gt_count, _ = watershed_segmentation(gt_distance, min_distance=3, verbose=False)
    print(f"\nGround Truth: {gt_count} grains")
    
    # Compare methods
    print("\n" + "=" * 60)
    print("Comparaison avec MATCHING des grains (max_distance=10)")
    print("=" * 60)
    
    # Method 1: peak_local_max (exp_004)
    print("\n1. peak_local_max (exp_004):")
    pred_labels_baseline, count_baseline, _ = watershed_segmentation(pred_distance, min_distance=3, verbose=False)
    metrics_baseline = match_grains(pred_labels_baseline, gt_labels, max_distance=10)
    print(f"   Predicted: {metrics_baseline['pred_grains']}")
    print(f"   TP: {metrics_baseline['true_positives']} | FP: {metrics_baseline['false_positives']} | FN: {metrics_baseline['false_negatives']}")
    print(f"   Precision: {metrics_baseline['precision']*100:.1f}%")
    print(f"   Recall: {metrics_baseline['recall']*100:.1f}%")
    print(f"   F1-Score: {metrics_baseline['f1_score']*100:.1f}%")
    
    # Method 2: h_maxima h=0.05
    print("\n2. H-maxima h=0.05:")
    pred_labels_h05, count_h05, _ = watershed_h_maxima(pred_distance, h=0.05, verbose=False)
    metrics_h05 = match_grains(pred_labels_h05, gt_labels, max_distance=10)
    print(f"   Predicted: {metrics_h05['pred_grains']}")
    print(f"   TP: {metrics_h05['true_positives']} | FP: {metrics_h05['false_positives']} | FN: {metrics_h05['false_negatives']}")
    print(f"   Precision: {metrics_h05['precision']*100:.1f}%")
    print(f"   Recall: {metrics_h05['recall']*100:.1f}%")
    print(f"   F1-Score: {metrics_h05['f1_score']*100:.1f}%")
    
    # Method 3: h_maxima h=0.08
    print("\n3. H-maxima h=0.08:")
    pred_labels_h08, count_h08, _ = watershed_h_maxima(pred_distance, h=0.08, verbose=False)
    metrics_h08 = match_grains(pred_labels_h08, gt_labels, max_distance=10)
    print(f"   Predicted: {metrics_h08['pred_grains']}")
    print(f"   TP: {metrics_h08['true_positives']} | FP: {metrics_h08['false_positives']} | FN: {metrics_h08['false_negatives']}")
    print(f"   Precision: {metrics_h08['precision']*100:.1f}%")
    print(f"   Recall: {metrics_h08['recall']*100:.1f}%")
    print(f"   F1-Score: {metrics_h08['f1_score']*100:.1f}%")
    
    # Find best by F1
    print("\n" + "=" * 60)
    results = {
        'peak_local_max': metrics_baseline,
        'h_maxima_h05': metrics_h05,
        'h_maxima_h08': metrics_h08
    }
    best = max(results.items(), key=lambda x: x[1]['f1_score'])
    print(f"🏆 Meilleur (F1-Score): {best[0]} avec F1={best[1]['f1_score']*100:.1f}%")
