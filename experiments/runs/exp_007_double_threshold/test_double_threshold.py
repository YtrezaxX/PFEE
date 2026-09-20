"""
Experiment 007: Double-Threshold Watershed

Approche:
1. Seuil HAUT (marker_thresh) pour trouver les centres des grains
2. Seuil BAS (mask_thresh) pour définir où les grains peuvent grandir
3. Watershed utilise les markers du seuil haut mais le masque du seuil bas

Résultat attendu: Grains de taille correcte avec fond propre
"""

import sys
sys.path.insert(0, '/home/gma/epita/PFEE/experiments')

import numpy as np
import torch
import tifffile
import json
import matplotlib.pyplot as plt
from glob import glob
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed
from skimage.morphology import h_maxima
from skimage.measure import regionprops

from models.unet_attention import AttentionUNet3D
from common.evaluate_experiment import predict_full_volume


def watershed_double_threshold(distance_map, marker_thresh=0.2, mask_thresh=0.05, h=0.03, verbose=False):
    """
    Watershed avec double seuil.
    
    Args:
        distance_map: Distance map prédite
        marker_thresh: Seuil pour trouver les markers (centres) - HAUT
        mask_thresh: Seuil pour le masque d'expansion - BAS
        h: Paramètre h-maxima
    
    Returns:
        labels, num_grains, mask
    """
    # Masque pour les MARKERS (seuil haut = centres propres)
    marker_mask = distance_map > marker_thresh
    dist_for_markers = distance_map.copy()
    dist_for_markers[~marker_mask] = 0
    
    # Trouver les markers avec h-maxima
    markers_bool = h_maxima(dist_for_markers, h=h)
    coords = np.array(np.where(markers_bool)).T
    
    if len(coords) == 0:
        return np.zeros_like(distance_map, dtype=np.int32), 0, marker_mask
    
    # Créer les markers
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for i, c in enumerate(coords):
        markers[tuple(c)] = i + 1
    
    # Masque pour l'EXPANSION (seuil bas = grains complets)
    expansion_mask = distance_map > mask_thresh
    
    # Distance map pour watershed (masquée par le seuil bas)
    dist_for_watershed = distance_map.copy()
    dist_for_watershed[~expansion_mask] = 0
    
    # Watershed: markers du seuil haut, masque du seuil bas
    labels = watershed(-dist_for_watershed, markers, mask=expansion_mask)
    
    num_grains = len(np.unique(labels)) - 1
    
    if verbose:
        print(f"  marker_thresh={marker_thresh}, mask_thresh={mask_thresh}, h={h}")
        print(f"  Markers found: {len(coords)}, Final grains: {num_grains}")
    
    return labels, num_grains, expansion_mask


def get_grain_centers(labels):
    props = regionprops(labels)
    return np.array([p.centroid for p in props])


def evaluate_segmentation(pred_labels, gt_labels, max_distance=10):
    pred_centers = get_grain_centers(pred_labels)
    gt_centers = get_grain_centers(gt_labels)
    
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return 0, 0, 0, 0, len(pred_centers), len(gt_centers)
    
    distances = cdist(pred_centers, gt_centers)
    
    matched_gt = set()
    matched_pred = set()
    
    flat_indices = np.argsort(distances.flatten())
    for flat_idx in flat_indices:
        pred_idx = flat_idx // len(gt_centers)
        gt_idx = flat_idx % len(gt_centers)
        
        if pred_idx in matched_pred or gt_idx in matched_gt:
            continue
        if distances[pred_idx, gt_idx] <= max_distance:
            matched_pred.add(pred_idx)
            matched_gt.add(gt_idx)
    
    tp = len(matched_pred)
    fp = len(pred_centers) - tp
    fn = len(gt_centers) - tp
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1, tp, fp, fn


def compute_size_metrics(pred_labels, gt_labels):
    """Compare grain sizes between prediction and ground truth."""
    pred_props = regionprops(pred_labels)
    gt_props = regionprops(gt_labels)
    
    pred_sizes = [p.area for p in pred_props]
    gt_sizes = [p.area for p in gt_props]
    
    return {
        'pred_mean_size': np.mean(pred_sizes) if pred_sizes else 0,
        'gt_mean_size': np.mean(gt_sizes) if gt_sizes else 0,
        'pred_median_size': np.median(pred_sizes) if pred_sizes else 0,
        'gt_median_size': np.median(gt_sizes) if gt_sizes else 0,
        'size_ratio': np.mean(pred_sizes) / np.mean(gt_sizes) if gt_sizes else 0
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 007: Double-Threshold Watershed")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = AttentionUNet3D(in_channels=1, out_channels=1, init_features=48).to(device)
    checkpoint = torch.load('/home/gma/epita/PFEE/experiments/runs/exp_004_attention_unet/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded")
    
    # Load data
    img_files = sorted(glob('/home/gma/epita/PFEE/data_yukiko/dataset/default/*.tif'))
    gt_files = sorted(glob('/home/gma/epita/PFEE/data_yukiko/dataset/ground_truth/*.tif'))
    
    image = tifffile.imread(img_files[0]).astype(np.float32) / 65535.0
    gt_distance = tifffile.imread(gt_files[0]).astype(np.float32) / 65535.0
    
    print(f"Image shape: {image.shape}")
    
    # Predict
    print("\nPredicting distance map...")
    pred_distance = predict_full_volume(model, image, patch_size=64, stride=32, device=device)
    
    # Ground truth segmentation
    print("Generating ground truth segmentation...")
    gt_labels, gt_num, _ = watershed_double_threshold(gt_distance, marker_thresh=0.1, mask_thresh=0.05, h=0.05)
    print(f"Ground truth: {gt_num} grains")
    
    results = []
    
    # =================================================================
    # TEST: Différentes combinaisons marker_thresh / mask_thresh
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST: Double-Threshold Combinations")
    print("=" * 60)
    
    marker_thresholds = [0.15, 0.2, 0.25]
    mask_thresholds = [0.02, 0.05, 0.08, 0.1, 0.15]
    h_values = [0.03, 0.05]
    
    best_f1 = 0
    best_config = None
    best_size_ratio = 0
    
    for marker_t in marker_thresholds:
        for mask_t in mask_thresholds:
            if mask_t >= marker_t:  # mask_thresh doit être < marker_thresh
                continue
            
            for h in h_values:
                labels, num_grains, mask = watershed_double_threshold(
                    pred_distance, 
                    marker_thresh=marker_t, 
                    mask_thresh=mask_t, 
                    h=h,
                    verbose=False
                )
                
                p, r, f1, tp, fp, fn = evaluate_segmentation(labels, gt_labels)
                size_metrics = compute_size_metrics(labels, gt_labels)
                
                result = {
                    'marker_thresh': marker_t,
                    'mask_thresh': mask_t,
                    'h': h,
                    'num_grains': num_grains,
                    'precision': p,
                    'recall': r,
                    'f1': f1,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    **size_metrics
                }
                results.append(result)
                
                print(f"  marker={marker_t}, mask={mask_t}, h={h}: "
                      f"{num_grains} grains, F1={f1:.3f}, size_ratio={size_metrics['size_ratio']:.2f}")
                
                # Track best (considering both F1 and size ratio)
                if f1 > best_f1 or (f1 == best_f1 and abs(size_metrics['size_ratio'] - 1.0) < abs(best_size_ratio - 1.0)):
                    best_f1 = f1
                    best_config = result
                    best_size_ratio = size_metrics['size_ratio']
    
    # =================================================================
    # Comparaison avec exp_006 (single threshold)
    # =================================================================
    print("\n" + "=" * 60)
    print("COMPARISON: Single vs Double Threshold")
    print("=" * 60)
    
    # Single threshold (exp_006 best)
    from skimage.morphology import h_maxima as hmax
    single_mask = pred_distance > 0.2
    dist_single = pred_distance.copy()
    dist_single[~single_mask] = 0
    markers_single = hmax(dist_single, h=0.03)
    coords_single = np.array(np.where(markers_single)).T
    markers_arr = np.zeros(pred_distance.shape, dtype=np.int32)
    for i, c in enumerate(coords_single):
        markers_arr[tuple(c)] = i + 1
    labels_single = watershed(-dist_single, markers_arr, mask=single_mask)
    
    p_single, r_single, f1_single, _, _, _ = evaluate_segmentation(labels_single, gt_labels)
    size_single = compute_size_metrics(labels_single, gt_labels)
    
    print(f"\nSingle threshold (exp_006): F1={f1_single:.4f}, size_ratio={size_single['size_ratio']:.3f}")
    print(f"Double threshold (best):    F1={best_f1:.4f}, size_ratio={best_size_ratio:.3f}")
    
    # =================================================================
    # Save results
    # =================================================================
    with open('/home/gma/epita/PFEE/experiments/runs/exp_007_double_threshold/results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("BEST CONFIGURATION")
    print("=" * 60)
    print(f"\nBest F1: {best_f1:.4f}")
    print(f"Best config: {best_config}")
    
    # =================================================================
    # Visualization
    # =================================================================
    print("\nGenerating visualizations...")
    
    # Best double-threshold segmentation
    labels_best, num_best, mask_best = watershed_double_threshold(
        pred_distance,
        marker_thresh=best_config['marker_thresh'],
        mask_thresh=best_config['mask_thresh'],
        h=best_config['h']
    )
    
    slice_idx = image.shape[0] // 2
    
    # 3-panel comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Random colors for grains
    np.random.seed(42)
    colors_pred = np.random.rand(num_best + 1, 3)
    colors_pred[0] = [0, 0, 0]
    colors_gt = np.random.rand(gt_num + 1, 3)
    colors_gt[0] = [0, 0, 0]
    
    axes[0].imshow(image[slice_idx], cmap='gray')
    axes[0].set_title('Input Image', fontsize=14)
    axes[0].axis('off')
    
    axes[1].imshow(colors_pred[labels_best[slice_idx]])
    axes[1].set_title(f'Predicted ({num_best} grains)\nsize_ratio={best_config["size_ratio"]:.2f}', fontsize=14)
    axes[1].axis('off')
    
    axes[2].imshow(colors_gt[gt_labels[slice_idx]])
    axes[2].set_title(f'Ground Truth ({gt_num} grains)', fontsize=14)
    axes[2].axis('off')
    
    plt.suptitle(f'Exp 007: Double-Threshold (marker={best_config["marker_thresh"]}, mask={best_config["mask_thresh"]}) - F1={best_f1:.2%}', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_007_double_threshold/results/segmentation_result.png', 
                dpi=150, bbox_inches='tight')
    print("Saved: segmentation_result.png")
    
    # Comparison: Single vs Double threshold
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Single threshold (exp_006)
    colors_single = np.random.rand(len(np.unique(labels_single)), 3)
    colors_single[0] = [0, 0, 0]
    
    axes2[0, 0].imshow(image[slice_idx], cmap='gray')
    axes2[0, 0].set_title('Input', fontsize=12)
    axes2[0, 0].axis('off')
    
    axes2[0, 1].imshow(colors_single[labels_single[slice_idx] % len(colors_single)])
    axes2[0, 1].set_title(f'Single Threshold (exp_006)\nF1={f1_single:.3f}, size_ratio={size_single["size_ratio"]:.2f}', fontsize=11)
    axes2[0, 1].axis('off')
    
    axes2[0, 2].imshow(single_mask[slice_idx], cmap='gray')
    axes2[0, 2].set_title(f'Mask (thresh=0.2)\n{single_mask.sum():,} voxels', fontsize=11)
    axes2[0, 2].axis('off')
    
    # Row 2: Double threshold (exp_007)
    axes2[1, 0].imshow(pred_distance[slice_idx], cmap='hot')
    axes2[1, 0].set_title('Distance Map', fontsize=12)
    axes2[1, 0].axis('off')
    
    axes2[1, 1].imshow(colors_pred[labels_best[slice_idx]])
    axes2[1, 1].set_title(f'Double Threshold (exp_007)\nF1={best_f1:.3f}, size_ratio={best_config["size_ratio"]:.2f}', fontsize=11)
    axes2[1, 1].axis('off')
    
    axes2[1, 2].imshow(mask_best[slice_idx], cmap='gray')
    axes2[1, 2].set_title(f'Mask (thresh={best_config["mask_thresh"]})\n{mask_best.sum():,} voxels', fontsize=11)
    axes2[1, 2].axis('off')
    
    plt.suptitle('Comparison: Single vs Double Threshold', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_007_double_threshold/results/comparison.png',
                dpi=150, bbox_inches='tight')
    print("Saved: comparison.png")
    
    # Size distribution comparison
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))
    
    pred_props = regionprops(labels_best)
    gt_props = regionprops(gt_labels)
    single_props = regionprops(labels_single)
    
    pred_sizes = [p.area for p in pred_props]
    gt_sizes = [p.area for p in gt_props]
    single_sizes = [p.area for p in single_props]
    
    bins = np.linspace(0, max(max(gt_sizes), max(pred_sizes)), 50)
    
    axes3[0].hist(single_sizes, bins=bins, alpha=0.5, label=f'Single (mean={np.mean(single_sizes):.0f})', color='red')
    axes3[0].hist(pred_sizes, bins=bins, alpha=0.5, label=f'Double (mean={np.mean(pred_sizes):.0f})', color='blue')
    axes3[0].hist(gt_sizes, bins=bins, alpha=0.5, label=f'GT (mean={np.mean(gt_sizes):.0f})', color='green')
    axes3[0].set_xlabel('Grain Volume (voxels)')
    axes3[0].set_ylabel('Count')
    axes3[0].set_title('Grain Size Distribution')
    axes3[0].legend()
    
    # Summary
    summary = f"""
SIZE COMPARISON:

Ground Truth:
  Mean: {np.mean(gt_sizes):.1f} voxels
  Median: {np.median(gt_sizes):.1f} voxels

Single Threshold (exp_006):
  Mean: {np.mean(single_sizes):.1f} voxels
  Ratio: {np.mean(single_sizes)/np.mean(gt_sizes):.2f}x

Double Threshold (exp_007):
  Mean: {np.mean(pred_sizes):.1f} voxels
  Ratio: {np.mean(pred_sizes)/np.mean(gt_sizes):.2f}x

IMPROVEMENT: {(np.mean(pred_sizes)/np.mean(gt_sizes) - np.mean(single_sizes)/np.mean(gt_sizes))*100:+.1f}%
"""
    axes3[1].text(0.1, 0.5, summary, fontsize=12, family='monospace',
                  verticalalignment='center', transform=axes3[1].transAxes)
    axes3[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_007_double_threshold/results/size_comparison.png',
                dpi=150, bbox_inches='tight')
    print("Saved: size_comparison.png")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 007 COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
