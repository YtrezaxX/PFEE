"""
Experiment 008: Test model on clean images (without noise)

Compare performance on:
1. Clean (no blur, no noise)
2. Blur only (blur, no noise)
3. Low noise (blur, 10% noise)
4. Original (blur, 30% noise)
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
    """Best watershed from exp_007."""
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
    """Evaluate with grain matching."""
    pred_props = regionprops(pred_labels)
    gt_props = regionprops(gt_labels)
    
    pred_centers = np.array([p.centroid for p in pred_props])
    gt_centers = np.array([p.centroid for p in gt_props])
    
    if len(pred_centers) == 0 or len(gt_centers) == 0:
        return {'f1': 0, 'precision': 0, 'recall': 0, 'tp': 0, 'fp': len(pred_centers), 'fn': len(gt_centers)}
    
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
    
    return {'f1': f1, 'precision': precision, 'recall': recall, 'tp': tp, 'fp': fp, 'fn': fn}


def main():
    print("=" * 60)
    print("EXPERIMENT 008: Testing on Clean Images")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model (trained on noisy images)
    model = AttentionUNet3D(in_channels=1, out_channels=1, init_features=48).to(device)
    checkpoint = torch.load('/home/gma/epita/PFEE/experiments/runs/exp_004_attention_unet/checkpoints/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded (trained on 30% noise images)")
    
    # Load ground truth distance map
    gt_file = sorted(glob('/home/gma/epita/PFEE/data_yukiko/dataset/ground_truth/*.tif'))[0]
    gt_distance = tifffile.imread(gt_file).astype(np.float32) / 65535.0
    gt_labels, gt_num = watershed_double_threshold(gt_distance, marker_thresh=0.1, mask_thresh=0.05, h=0.05)
    print(f"Ground truth: {gt_num} grains")
    
    # Test images
    dataset_path = '/home/gma/epita/PFEE/experiments/runs/exp_008_clean_images/dataset/'
    test_configs = [
        {"name": "clean", "desc": "No blur, No noise"},
        {"name": "blur_only", "desc": "Blur 0.8, No noise"},
        {"name": "low_noise", "desc": "Blur 0.8, Noise 10%"},
        {"name": "original", "desc": "Blur 0.8, Noise 30%"},
    ]
    
    results = []
    images = {}
    predictions = {}
    segmentations = {}
    
    print("\n" + "=" * 60)
    print("TESTING DIFFERENT NOISE LEVELS")
    print("=" * 60)
    
    for config in test_configs:
        img_path = dataset_path + config["name"] + ".tif"
        image = tifffile.imread(img_path).astype(np.float32) / 65535.0
        
        print(f"\n{config['name'].upper()} ({config['desc']}):")
        print(f"  Image range: [{image.min():.3f}, {image.max():.3f}]")
        
        # Predict
        pred_distance = predict_full_volume(model, image, patch_size=64, stride=32, device=device)
        
        # Distance map quality
        mae = np.mean(np.abs(pred_distance - gt_distance))
        print(f"  Distance Map MAE: {mae:.4f}")
        
        # Segment
        labels, num_grains = watershed_double_threshold(pred_distance)
        print(f"  Grains detected: {num_grains}")
        
        # Evaluate
        metrics = evaluate_segmentation(labels, gt_labels)
        print(f"  F1-Score: {metrics['f1']:.4f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f})")
        print(f"  TP={metrics['tp']}, FP={metrics['fp']}, FN={metrics['fn']}")
        
        results.append({
            'name': config['name'],
            'description': config['desc'],
            'mae': float(mae),
            'num_grains': int(num_grains),
            'f1': float(metrics['f1']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'tp': int(metrics['tp']),
            'fp': int(metrics['fp']),
            'fn': int(metrics['fn'])
        })
        
        images[config['name']] = image
        predictions[config['name']] = pred_distance
        segmentations[config['name']] = labels
    
    # Save results
    with open('/home/gma/epita/PFEE/experiments/runs/exp_008_clean_images/results/metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # =================================================================
    # Visualization
    # =================================================================
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    slice_idx = gt_distance.shape[0] // 2
    
    # 4x4 comparison: Input | Distance Map | Segmentation | GT
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    
    for i, config in enumerate(test_configs):
        name = config['name']
        
        # Input
        axes[i, 0].imshow(images[name][slice_idx], cmap='gray')
        axes[i, 0].set_title(f"{name.upper()}\n{config['desc']}", fontsize=11)
        axes[i, 0].axis('off')
        
        # Predicted distance
        axes[i, 1].imshow(predictions[name][slice_idx], cmap='hot')
        mae = results[i]['mae']
        axes[i, 1].set_title(f"Predicted Distance\nMAE={mae:.4f}", fontsize=11)
        axes[i, 1].axis('off')
        
        # Segmentation
        np.random.seed(42)
        num = results[i]['num_grains']
        colors = np.random.rand(num + 1, 3)
        colors[0] = [0, 0, 0]
        axes[i, 2].imshow(colors[segmentations[name][slice_idx] % (num + 1)])
        f1 = results[i]['f1']
        axes[i, 2].set_title(f"Segmentation ({num} grains)\nF1={f1:.2%}", fontsize=11)
        axes[i, 2].axis('off')
        
        # GT (same for all)
        np.random.seed(123)
        colors_gt = np.random.rand(gt_num + 1, 3)
        colors_gt[0] = [0, 0, 0]
        axes[i, 3].imshow(colors_gt[gt_labels[slice_idx] % (gt_num + 1)])
        axes[i, 3].set_title(f"Ground Truth\n({gt_num} grains)", fontsize=11)
        axes[i, 3].axis('off')
    
    plt.suptitle('Experiment 008: Impact of Noise on Segmentation Quality', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_008_clean_images/results/noise_comparison.png',
                dpi=150, bbox_inches='tight')
    print("Saved: noise_comparison.png")
    
    # Bar chart comparison
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    
    names = [r['name'] for r in results]
    f1_scores = [r['f1'] * 100 for r in results]
    maes = [r['mae'] for r in results]
    grain_counts = [r['num_grains'] for r in results]
    
    colors = ['green', 'blue', 'orange', 'red']
    
    axes2[0].bar(names, f1_scores, color=colors)
    axes2[0].set_ylabel('F1-Score (%)')
    axes2[0].set_title('Segmentation Quality')
    axes2[0].axhline(y=98.52, color='gray', linestyle='--', label='exp_007 baseline')
    for i, v in enumerate(f1_scores):
        axes2[0].text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    axes2[1].bar(names, maes, color=colors)
    axes2[1].set_ylabel('MAE')
    axes2[1].set_title('Distance Map Error')
    for i, v in enumerate(maes):
        axes2[1].text(i, v + 0.001, f'{v:.4f}', ha='center')
    
    axes2[2].bar(names, grain_counts, color=colors)
    axes2[2].axhline(y=gt_num, color='gray', linestyle='--', label=f'GT ({gt_num})')
    axes2[2].set_ylabel('Grain Count')
    axes2[2].set_title('Number of Grains Detected')
    axes2[2].legend()
    for i, v in enumerate(grain_counts):
        axes2[2].text(i, v + 50, str(v), ha='center')
    
    plt.suptitle('Impact of Noise Level on Model Performance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/gma/epita/PFEE/experiments/runs/exp_008_clean_images/results/metrics_comparison.png',
                dpi=150, bbox_inches='tight')
    print("Saved: metrics_comparison.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Config':<15} {'MAE':<10} {'F1-Score':<12} {'Grains':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['name']:<15} {r['mae']:<10.4f} {r['f1']*100:<12.2f} {r['num_grains']:<10}")
    print(f"{'Ground Truth':<15} {'-':<10} {'-':<12} {gt_num:<10}")
    
    print("\n" + "=" * 60)
    print("EXPERIMENT 008 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
