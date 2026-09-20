"""
Experiment 009: Evaluate model trained on clean images.

Compare:
1. Clean model on clean images (should be best)
2. Clean model on noisy images (test generalization)
3. Noisy model on clean images (from exp_008)
4. Noisy model on noisy images (baseline from exp_007)
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


def evaluate_segmentation(pred_labels, gt_labels, max_distance=10):
    pred_props = regionprops(pred_labels)
    gt_props = regionprops(gt_labels)
    
    if len(pred_props) == 0 or len(gt_props) == 0:
        return {'f1': 0, 'precision': 0, 'recall': 0, 'tp': 0, 'fp': len(pred_props), 'fn': len(gt_props)}
    
    pred_centers = np.array([p.centroid for p in pred_props])
    gt_centers = np.array([p.centroid for p in gt_props])
    
    distances = cdist(pred_centers, gt_centers)
    
    matched_gt = set()
    matched_pred = set()
    
    for flat_idx in np.argsort(distances.flatten()):
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
    
    return {'f1': f1, 'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn,
            'num_pred': len(pred_centers), 'num_gt': len(gt_centers)}


def main():
    print("=" * 60)
    print("EXPERIMENT 009: Evaluation - Clean vs Noisy Training")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load CLEAN model (trained on clean images)
    model_clean = AttentionUNet3D(in_channels=1, out_channels=1, init_features=48).to(device)
    ckpt_clean = torch.load('/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/checkpoints/best_model.pth')
    model_clean.load_state_dict(ckpt_clean['model_state_dict'])
    model_clean.eval()
    print("Loaded: Clean model (trained on clean images)")
    
    # Load NOISY model (trained on noisy images - from exp_004)
    model_noisy = AttentionUNet3D(in_channels=1, out_channels=1, init_features=48).to(device)
    ckpt_noisy = torch.load('/home/gma/epita/PFEE/experiments/runs/exp_004_attention_unet/checkpoints/best_model.pth')
    model_noisy.load_state_dict(ckpt_noisy['model_state_dict'])
    model_noisy.eval()
    print("Loaded: Noisy model (trained on 30% noise images)")
    
    # Test images
    # Clean test images (hold-out from exp_009)
    clean_test_img = sorted(glob('/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/dataset/images/*.tif'))[-2:]
    clean_test_gt = sorted(glob('/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/dataset/ground_truth/*.tif'))[-2:]
    
    # Noisy test image
    noisy_test_img = sorted(glob('/home/gma/epita/PFEE/data_yukiko/dataset/default/*.tif'))[:1]
    noisy_test_gt = sorted(glob('/home/gma/epita/PFEE/data_yukiko/dataset/ground_truth/*.tif'))[:1]
    
    results = []
    
    # =================================================================
    # Test 1: Clean model on clean images
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Clean Model → Clean Images")
    print("=" * 60)
    
    for img_f, gt_f in zip(clean_test_img, clean_test_gt):
        image = tifffile.imread(img_f).astype(np.float32) / 65535.0
        gt_dist = tifffile.imread(gt_f).astype(np.float32) / 65535.0
        gt_labels, gt_num = watershed_double_threshold(gt_dist, marker_thresh=0.1, mask_thresh=0.05, h=0.05)
        
        pred_dist = predict_full_volume(model_clean, image, patch_size=64, stride=32, device=device)
        pred_labels, pred_num = watershed_double_threshold(pred_dist)
        
        mae = np.mean(np.abs(pred_dist - gt_dist))
        metrics = evaluate_segmentation(pred_labels, gt_labels)
        
        print(f"  {img_f.split('/')[-1][:30]}...")
        print(f"    MAE: {mae:.4f}, F1: {metrics['f1']:.4f}, Grains: {pred_num}/{gt_num}")
        
        results.append({
            'test': 'clean_model_clean_image',
            'mae': float(mae),
            **{k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in metrics.items()}
        })
    
    # =================================================================
    # Test 2: Clean model on noisy images
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Clean Model → Noisy Images (30% noise)")
    print("=" * 60)
    
    for img_f, gt_f in zip(noisy_test_img, noisy_test_gt):
        image = tifffile.imread(img_f).astype(np.float32) / 65535.0
        gt_dist = tifffile.imread(gt_f).astype(np.float32) / 65535.0
        gt_labels, gt_num = watershed_double_threshold(gt_dist, marker_thresh=0.1, mask_thresh=0.05, h=0.05)
        
        pred_dist = predict_full_volume(model_clean, image, patch_size=64, stride=32, device=device)
        pred_labels, pred_num = watershed_double_threshold(pred_dist)
        
        mae = np.mean(np.abs(pred_dist - gt_dist))
        metrics = evaluate_segmentation(pred_labels, gt_labels)
        
        print(f"  {img_f.split('/')[-1][:40]}...")
        print(f"    MAE: {mae:.4f}, F1: {metrics['f1']:.4f}, Grains: {pred_num}/{gt_num}")
        
        results.append({
            'test': 'clean_model_noisy_image',
            'mae': float(mae),
            **{k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in metrics.items()}
        })
    
    # =================================================================
    # Test 3: Noisy model on clean images
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Noisy Model → Clean Images")
    print("=" * 60)
    
    for img_f, gt_f in zip(clean_test_img, clean_test_gt):
        image = tifffile.imread(img_f).astype(np.float32) / 65535.0
        gt_dist = tifffile.imread(gt_f).astype(np.float32) / 65535.0
        gt_labels, gt_num = watershed_double_threshold(gt_dist, marker_thresh=0.1, mask_thresh=0.05, h=0.05)
        
        pred_dist = predict_full_volume(model_noisy, image, patch_size=64, stride=32, device=device)
        pred_labels, pred_num = watershed_double_threshold(pred_dist)
        
        mae = np.mean(np.abs(pred_dist - gt_dist))
        metrics = evaluate_segmentation(pred_labels, gt_labels)
        
        print(f"  {img_f.split('/')[-1][:30]}...")
        print(f"    MAE: {mae:.4f}, F1: {metrics['f1']:.4f}, Grains: {pred_num}/{gt_num}")
        
        results.append({
            'test': 'noisy_model_clean_image',
            'mae': float(mae),
            **{k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in metrics.items()}
        })
    
    # =================================================================
    # Test 4: Noisy model on noisy images (baseline)
    # =================================================================
    print("\n" + "=" * 60)
    print("TEST 4: Noisy Model → Noisy Images (baseline)")
    print("=" * 60)
    
    for img_f, gt_f in zip(noisy_test_img, noisy_test_gt):
        image = tifffile.imread(img_f).astype(np.float32) / 65535.0
        gt_dist = tifffile.imread(gt_f).astype(np.float32) / 65535.0
        gt_labels, gt_num = watershed_double_threshold(gt_dist, marker_thresh=0.1, mask_thresh=0.05, h=0.05)
        
        pred_dist = predict_full_volume(model_noisy, image, patch_size=64, stride=32, device=device)
        pred_labels, pred_num = watershed_double_threshold(pred_dist)
        
        mae = np.mean(np.abs(pred_dist - gt_dist))
        metrics = evaluate_segmentation(pred_labels, gt_labels)
        
        print(f"  {img_f.split('/')[-1][:40]}...")
        print(f"    MAE: {mae:.4f}, F1: {metrics['f1']:.4f}, Grains: {pred_num}/{gt_num}")
        
        results.append({
            'test': 'noisy_model_noisy_image',
            'mae': float(mae),
            **{k: float(v) if isinstance(v, (np.floating, float)) else int(v) for k, v in metrics.items()}
        })
    
    # Save results
    with open('/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "=" * 60)
    print("SUMMARY - Average F1 Scores")
    print("=" * 60)
    
    tests = ['clean_model_clean_image', 'clean_model_noisy_image', 
             'noisy_model_clean_image', 'noisy_model_noisy_image']
    
    for test in tests:
        test_results = [r for r in results if r['test'] == test]
        if test_results:
            avg_f1 = np.mean([r['f1'] for r in test_results])
            avg_mae = np.mean([r['mae'] for r in test_results])
            print(f"{test:30s}: F1={avg_f1:.4f}, MAE={avg_mae:.4f}")
    
    # Visualization
    print("\nGenerating visualization...")
    
    # Load one example for visualization
    img_clean = tifffile.imread(clean_test_img[0]).astype(np.float32) / 65535.0
    gt_clean = tifffile.imread(clean_test_gt[0]).astype(np.float32) / 65535.0
    
    pred_clean_model = predict_full_volume(model_clean, img_clean, patch_size=64, stride=32, device=device)
    pred_noisy_model = predict_full_volume(model_noisy, img_clean, patch_size=64, stride=32, device=device)
    
    labels_clean, _ = watershed_double_threshold(pred_clean_model)
    labels_noisy, _ = watershed_double_threshold(pred_noisy_model)
    labels_gt, gt_num = watershed_double_threshold(gt_clean, marker_thresh=0.1, mask_thresh=0.05, h=0.05)
    
    slice_idx = img_clean.shape[0] // 2
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Clean model
    axes[0, 0].imshow(img_clean[slice_idx], cmap='gray')
    axes[0, 0].set_title('Clean Input Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(pred_clean_model[slice_idx], cmap='hot')
    axes[0, 1].set_title('Clean Model Prediction')
    axes[0, 1].axis('off')
    
    np.random.seed(42)
    n = len(np.unique(labels_clean))
    colors = np.random.rand(n, 3)
    colors[0] = [0, 0, 0]
    axes[0, 2].imshow(colors[labels_clean[slice_idx] % n])
    axes[0, 2].set_title(f'Clean Model Segmentation\n({len(np.unique(labels_clean))-1} grains)')
    axes[0, 2].axis('off')
    
    n_gt = len(np.unique(labels_gt))
    colors_gt = np.random.rand(n_gt, 3)
    colors_gt[0] = [0, 0, 0]
    axes[0, 3].imshow(colors_gt[labels_gt[slice_idx] % n_gt])
    axes[0, 3].set_title(f'Ground Truth\n({gt_num} grains)')
    axes[0, 3].axis('off')
    
    # Row 2: Noisy model on same clean image
    axes[1, 0].imshow(img_clean[slice_idx], cmap='gray')
    axes[1, 0].set_title('Same Clean Input')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(pred_noisy_model[slice_idx], cmap='hot')
    axes[1, 1].set_title('Noisy Model Prediction')
    axes[1, 1].axis('off')
    
    n2 = len(np.unique(labels_noisy))
    colors2 = np.random.rand(n2, 3)
    colors2[0] = [0, 0, 0]
    axes[1, 2].imshow(colors2[labels_noisy[slice_idx] % n2])
    axes[1, 2].set_title(f'Noisy Model Segmentation\n({len(np.unique(labels_noisy))-1} grains)')
    axes[1, 2].axis('off')
    
    # Summary text
    clean_f1 = [r['f1'] for r in results if r['test'] == 'clean_model_clean_image']
    noisy_f1 = [r['f1'] for r in results if r['test'] == 'noisy_model_clean_image']
    
    summary = f"""RESULTS ON CLEAN IMAGES:

Clean Model: F1 = {np.mean(clean_f1):.2%}
Noisy Model: F1 = {np.mean(noisy_f1):.2%}

Difference: {(np.mean(clean_f1) - np.mean(noisy_f1))*100:+.1f}%
"""
    axes[1, 3].text(0.1, 0.5, summary, fontsize=14, family='monospace',
                    verticalalignment='center', transform=axes[1, 3].transAxes)
    axes[1, 3].axis('off')
    
    plt.suptitle('Exp 009: Clean Model vs Noisy Model on Clean Images', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_009_clean_training/results/comparison.png',
                dpi=150, bbox_inches='tight')
    print("Saved: comparison.png")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 009 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
