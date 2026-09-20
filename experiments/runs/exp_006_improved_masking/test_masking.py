"""
Experiment 006: Test Different Masking Strategies for Watershed

Ce script teste différentes approches pour améliorer le masquage:
1. Différents seuils de distance
2. Opérations morphologiques
3. Filtrage post-segmentation
4. Combinaison avec l'image source
"""

import sys
import os
sys.path.insert(0, '/home/gma/epita/PFEE/experiments')

import numpy as np
import torch
import tifffile
import json
import matplotlib.pyplot as plt
from glob import glob
from scipy import ndimage
from scipy.spatial.distance import cdist
from skimage.segmentation import watershed, clear_border
from skimage.morphology import h_maxima, ball, binary_opening, binary_closing, remove_small_objects
from scipy.ndimage import binary_erosion
from skimage.measure import regionprops, label
from tqdm import tqdm

from models.unet_attention import AttentionUNet3D
from common.evaluate_experiment import predict_full_volume


def watershed_with_threshold(distance_map, threshold=0.1, h=0.05, verbose=False):
    """Watershed avec seuil de masque ajustable."""
    # Masque avec seuil
    grain_mask = distance_map > threshold
    
    if verbose:
        print(f"  Threshold={threshold}: {grain_mask.sum()} voxels in mask")
    
    # Distance map seuillée
    dist_thresh = distance_map.copy()
    dist_thresh[~grain_mask] = 0
    
    # H-maxima pour trouver les centres
    markers_bool = h_maxima(dist_thresh, h=h)
    coordinates = np.array(np.where(markers_bool)).T
    
    if len(coordinates) == 0:
        return np.zeros_like(distance_map, dtype=np.int32), 0, grain_mask
    
    # Créer les markers
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[tuple(coord)] = idx + 1
    
    # Watershed
    labels = watershed(-dist_thresh, markers, mask=grain_mask)
    
    num_grains = len(np.unique(labels)) - 1  # -1 pour le fond
    
    return labels, num_grains, grain_mask


def watershed_with_morphology(distance_map, threshold=0.1, h=0.05, 
                               erosion_iter=0, opening_radius=0, verbose=False):
    """Watershed avec opérations morphologiques sur le masque."""
    # Masque initial
    grain_mask = distance_map > threshold
    
    # Opérations morphologiques
    if erosion_iter > 0:
        grain_mask = binary_erosion(grain_mask, iterations=erosion_iter)
    
    if opening_radius > 0:
        struct = ball(opening_radius)
        grain_mask = binary_opening(grain_mask, footprint=struct)
    
    if verbose:
        print(f"  After morphology: {grain_mask.sum()} voxels")
    
    # Distance map masquée
    dist_masked = distance_map.copy()
    dist_masked[~grain_mask] = 0
    
    # H-maxima
    markers_bool = h_maxima(dist_masked, h=h)
    coordinates = np.array(np.where(markers_bool)).T
    
    if len(coordinates) == 0:
        return np.zeros_like(distance_map, dtype=np.int32), 0, grain_mask
    
    markers = np.zeros(distance_map.shape, dtype=np.int32)
    for idx, coord in enumerate(coordinates):
        markers[tuple(coord)] = idx + 1
    
    labels = watershed(-dist_masked, markers, mask=grain_mask)
    num_grains = len(np.unique(labels)) - 1
    
    return labels, num_grains, grain_mask


def filter_segmentation(labels, min_volume=50, max_volume=50000, remove_border=True):
    """Filtre les segments par taille et position."""
    filtered = labels.copy()
    
    # Supprimer les grains touchant les bords (3D)
    if remove_border:
        # Mark border grains
        border_labels = set()
        # Check all 6 faces of the 3D volume
        for face in [labels[0, :, :], labels[-1, :, :],
                     labels[:, 0, :], labels[:, -1, :],
                     labels[:, :, 0], labels[:, :, -1]]:
            border_labels.update(np.unique(face))
        border_labels.discard(0)  # Don't remove background
        
        for lbl in border_labels:
            filtered[filtered == lbl] = 0
    
    # Supprimer par taille
    props = regionprops(filtered)
    for prop in props:
        if prop.area < min_volume or prop.area > max_volume:
            filtered[filtered == prop.label] = 0
    
    # Re-label pour avoir des IDs consécutifs
    unique_labels = np.unique(filtered)
    unique_labels = unique_labels[unique_labels > 0]
    new_filtered = np.zeros_like(filtered)
    for new_id, old_id in enumerate(unique_labels, start=1):
        new_filtered[filtered == old_id] = new_id
    
    return new_filtered


def get_grain_centers(labels):
    """Extrait les centres des grains."""
    props = regionprops(labels)
    centers = np.array([p.centroid for p in props])
    return centers


def evaluate_segmentation(pred_labels, gt_labels, max_distance=10):
    """Évalue la segmentation avec matching des grains."""
    pred_centers = get_grain_centers(pred_labels)
    gt_centers = get_grain_centers(gt_labels)
    
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return 0, 0, 0, 0, len(pred_centers), len(gt_centers)
    
    # Distance matrix
    distances = cdist(pred_centers, gt_centers)
    
    # Greedy matching
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


def main():
    print("=" * 60)
    print("EXPERIMENT 006: Improved Masking Strategies")
    print("=" * 60)
    
    # Setup
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
    
    # Use first volume for testing
    image = tifffile.imread(img_files[0]).astype(np.float32) / 65535.0
    gt_distance = tifffile.imread(gt_files[0]).astype(np.float32) / 65535.0
    
    print(f"Image shape: {image.shape}")
    
    # Predict distance map
    print("\nPredicting distance map...")
    pred_distance = predict_full_volume(model, image, patch_size=64, stride=32, device=device)
    
    # Ground truth segmentation (pour comparaison)
    print("\nGenerating ground truth segmentation...")
    gt_labels, gt_num, _ = watershed_with_threshold(gt_distance, threshold=0.1, h=0.05)
    print(f"Ground truth: {gt_num} grains")
    
    results = []
    
    # =================================================================
    # TEST 1: Différents seuils de masquage
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Threshold values")
    print("=" * 60)
    
    thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    for thresh in thresholds:
        labels, num_grains, mask = watershed_with_threshold(
            pred_distance, threshold=thresh, h=0.05, verbose=True
        )
        
        p, r, f1, tp, fp, fn = evaluate_segmentation(labels, gt_labels)
        
        result = {
            'test': 'threshold',
            'threshold': thresh,
            'num_grains': num_grains,
            'precision': p,
            'recall': r,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
        results.append(result)
        print(f"  Threshold={thresh:.2f}: {num_grains} grains, F1={f1:.3f} (P={p:.3f}, R={r:.3f})")
    
    # =================================================================
    # TEST 2: Seuil + filtrage post-segmentation
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Threshold + Post-filtering (size only, no border removal)")
    print("=" * 60)
    
    best_thresh = 0.2  # From test 1 (best F1)
    
    # Note: remove_border=True removes too many grains in dense packing
    for min_vol in [0, 10, 30, 50, 100]:
        for remove_border in [False]:  # Don't remove border - too aggressive
            labels, _, _ = watershed_with_threshold(pred_distance, threshold=best_thresh, h=0.05)
            labels_filtered = filter_segmentation(
                labels, min_volume=min_vol, max_volume=50000, remove_border=remove_border
            )
            num_grains = len(np.unique(labels_filtered)) - 1
            
            p, r, f1, tp, fp, fn = evaluate_segmentation(labels_filtered, gt_labels)
            
            result = {
                'test': 'filtering',
                'threshold': best_thresh,
                'min_volume': min_vol,
                'remove_border': remove_border,
                'num_grains': num_grains,
                'precision': p,
                'recall': r,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            results.append(result)
            print(f"  min_vol={min_vol}, remove_border={remove_border}: {num_grains} grains, F1={f1:.3f}")
    
    # =================================================================
    # TEST 3: Seuil + Morphologie
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Threshold + Morphological operations")
    print("=" * 60)
    
    for erosion in [0, 1, 2]:
        for opening in [0, 1, 2]:
            labels, num_grains, _ = watershed_with_morphology(
                pred_distance, threshold=best_thresh, h=0.05,
                erosion_iter=erosion, opening_radius=opening, verbose=False
            )
            
            p, r, f1, tp, fp, fn = evaluate_segmentation(labels, gt_labels)
            
            result = {
                'test': 'morphology',
                'threshold': best_thresh,
                'erosion_iter': erosion,
                'opening_radius': opening,
                'num_grains': num_grains,
                'precision': p,
                'recall': r,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            results.append(result)
            print(f"  erosion={erosion}, opening={opening}: {num_grains} grains, F1={f1:.3f}")
    
    # =================================================================
    # TEST 4: Combinaison optimale (threshold + h-maxima variations)
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Optimal combination (threshold + h variations)")
    print("=" * 60)
    
    # Test combinations of best parameters - NO border removal
    for thresh in [0.15, 0.2, 0.25]:
        for h in [0.03, 0.05, 0.07, 0.1]:
            labels, _, _ = watershed_with_threshold(pred_distance, threshold=thresh, h=h)
            num_grains = len(np.unique(labels)) - 1
            
            p, r, f1, tp, fp, fn = evaluate_segmentation(labels, gt_labels)
            
            result = {
                'test': 'combined',
                'threshold': thresh,
                'h': h,
                'num_grains': num_grains,
                'precision': p,
                'recall': r,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            results.append(result)
            print(f"  thresh={thresh}, h={h}: {num_grains} grains, F1={f1:.3f} (P={p:.3f}, R={r:.3f})")
    
    # Save results
    with open('/home/gma/epita/PFEE/experiments/runs/exp_006_improved_masking/results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # =================================================================
    # Find best configuration
    # =================================================================
    print("\n" + "=" * 60)
    print("BEST RESULTS")
    print("=" * 60)
    
    best_f1 = max(results, key=lambda x: x['f1'])
    print(f"\nBest F1 Score: {best_f1['f1']:.4f}")
    print(f"Configuration: {best_f1}")
    
    # =================================================================
    # Generate comparison visualization
    # =================================================================
    print("\nGenerating visualizations...")
    
    # Best configuration
    best_thresh = best_f1.get('threshold', 0.2)
    best_h = best_f1.get('h', 0.05)
    
    # Baseline (threshold=0)
    labels_baseline, _, mask_baseline = watershed_with_threshold(pred_distance, threshold=0.0, h=0.05)
    
    # Improved (no filtering - just better threshold)
    labels_improved, _, mask_improved = watershed_with_threshold(pred_distance, threshold=best_thresh, h=best_h)
    
    # Visualization
    slice_idx = image.shape[0] // 2
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Baseline (threshold=0)
    axes[0, 0].imshow(image[slice_idx], cmap='gray')
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mask_baseline[slice_idx], cmap='gray')
    axes[0, 1].set_title(f'Mask (thresh=0)\n{mask_baseline.sum()} voxels')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(labels_baseline[slice_idx], cmap='nipy_spectral')
    axes[0, 2].set_title(f'Baseline Segmentation\n{len(np.unique(labels_baseline))-1} grains')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(gt_labels[slice_idx], cmap='nipy_spectral')
    axes[0, 3].set_title(f'Ground Truth\n{gt_num} grains')
    axes[0, 3].axis('off')
    
    # Row 2: Improved
    axes[1, 0].imshow(pred_distance[slice_idx], cmap='hot')
    axes[1, 0].set_title('Predicted Distance')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_improved[slice_idx], cmap='gray')
    axes[1, 1].set_title(f'Mask (thresh={best_thresh})\n{mask_improved.sum()} voxels')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(labels_improved[slice_idx], cmap='nipy_spectral')
    num_improved = len(np.unique(labels_improved)) - 1
    axes[1, 2].set_title(f'Improved Segmentation\n{num_improved} grains (F1={best_f1["f1"]:.3f})')
    axes[1, 2].axis('off')
    
    # Error comparison
    p_base, r_base, f1_base, _, _, _ = evaluate_segmentation(labels_baseline, gt_labels)
    
    summary = f"""COMPARISON:

Baseline (thresh=0):
  Grains: {len(np.unique(labels_baseline))-1}
  F1: {f1_base:.3f}

Improved (thresh={best_thresh}):
  Grains: {num_improved}
  F1: {best_f1['f1']:.3f}

GT: {gt_num} grains

Improvement: +{(best_f1['f1']-f1_base)*100:.1f}%"""
    
    axes[1, 3].text(0.1, 0.5, summary, fontsize=12, family='monospace',
                    verticalalignment='center', transform=axes[1, 3].transAxes)
    axes[1, 3].axis('off')
    axes[1, 3].set_title('Summary')
    
    plt.suptitle('Experiment 006: Improved Masking', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_006_improved_masking/results/masking_comparison.png', 
                dpi=150, bbox_inches='tight')
    print("Saved: results/masking_comparison.png")
    
    # =================================================================
    # Threshold analysis plot
    # =================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    thresh_results = [r for r in results if r['test'] == 'threshold']
    threshs = [r['threshold'] for r in thresh_results]
    f1s = [r['f1'] for r in thresh_results]
    precisions = [r['precision'] for r in thresh_results]
    recalls = [r['recall'] for r in thresh_results]
    num_grains_list = [r['num_grains'] for r in thresh_results]
    
    axes[0].plot(threshs, f1s, 'b-o', linewidth=2, markersize=8, label='F1')
    axes[0].plot(threshs, precisions, 'g--s', linewidth=1.5, label='Precision')
    axes[0].plot(threshs, recalls, 'r--^', linewidth=1.5, label='Recall')
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Score')
    axes[0].set_title('Metrics vs Threshold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(range(len(threshs)), num_grains_list, color='steelblue')
    axes[1].axhline(y=gt_num, color='r', linestyle='--', label=f'GT ({gt_num})')
    axes[1].set_xticks(range(len(threshs)))
    axes[1].set_xticklabels([f'{t:.2f}' for t in threshs])
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Number of Grains')
    axes[1].set_title('Grain Count vs Threshold')
    axes[1].legend()
    
    # Best threshold marker
    best_idx = f1s.index(max(f1s))
    axes[2].text(0.5, 0.7, f"Best Threshold: {threshs[best_idx]:.2f}", 
                 fontsize=16, ha='center', transform=axes[2].transAxes)
    axes[2].text(0.5, 0.5, f"F1 Score: {max(f1s):.4f}", 
                 fontsize=14, ha='center', transform=axes[2].transAxes)
    axes[2].text(0.5, 0.3, f"Grains: {num_grains_list[best_idx]} (GT: {gt_num})", 
                 fontsize=14, ha='center', transform=axes[2].transAxes)
    axes[2].axis('off')
    axes[2].set_title('Best Configuration')
    
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_006_improved_masking/results/threshold_analysis.png',
                dpi=150, bbox_inches='tight')
    print("Saved: results/threshold_analysis.png")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 006 COMPLETE")
    print("=" * 60)
    
    return results


if __name__ == '__main__':
    main()
