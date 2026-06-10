"""
Exercise 7 — Anomaly Detection (Exercise Sheet 9).

Part 7.4: Visualising the distribution shift and computing mean confidence.
Part 7.6: Evaluating the MSP baseline for OOD detection.
Part 7.7: Feature-Based OOD Detection using k-NN.

Usage:
    python -m exercises.ex7_anomaly_detection

Note: This script expects pre-trained model weights in models/.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_DIR, TEST_DIR, TEST_FOG_DIR, TEST_NIGHT_DIR, TEST_TOWN_DIR,
    BATCH_SIZE, DEVICE, IMAGE_TRANSFORM, INV_NORMALIZE, TARGETS,
    RESULTS_DIR, MODELS_DIR,
)
from src.dataset import CarlaBinaryDataset
from src.model import create_binary_resnet18
from src.anomaly_detection import compute_msp, extract_features, compute_knn_ood_scores

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model(name: str) -> torch.nn.Module:
    path = os.path.join(MODELS_DIR, f"{name}_model.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights for '{name}' not found at {path}")
    model = create_binary_resnet18(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded {name} model from {path}")
    return model

def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    img = INV_NORMALIZE(tensor).clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()

def sample_images(dataset_dir: str, target: str, n: int = 5) -> list:
    ds = CarlaBinaryDataset(dataset_dir, target, transform=IMAGE_TRANSFORM)
    indices = np.random.choice(len(ds), min(n, len(ds)), replace=False)
    images = []
    for idx in indices:
        img_tensor, _ = ds[idx]
        images.append(tensor_to_numpy_image(img_tensor))
    return images

# ---------------------------------------------------------------------------
# Exercise 7.4 — Visualising the Distribution Shift
# ---------------------------------------------------------------------------
def exercise_7_4(models_dict: dict, save_dir: str):
    print(f"\n{'='*60}")
    print("  Exercise 7.4 — Visualising the Distribution Shift")
    print(f"{'='*60}")
    
    # 1. & 2. Display Images
    print("  Sampling images for visualization...")
    n_samples = 5
    target = "has_pedestrian" # arbitrary target just to load images
    
    id_imgs = sample_images(TEST_DIR, target, n=n_samples)
    fog_imgs = sample_images(TEST_FOG_DIR, target, n=n_samples)
    night_imgs = sample_images(TEST_NIGHT_DIR, target, n=n_samples)
    town_imgs = sample_images(TEST_TOWN_DIR, target, n=n_samples)
    
    fig, axes = plt.subplots(4, n_samples, figsize=(15, 10))
    datasets_samples = {
        "ID (Sunny/Daytime)": id_imgs,
        "OOD (Fog)": fog_imgs,
        "OOD (Night)": night_imgs,
        "OOD (Town-01)": town_imgs
    }
    
    for row_idx, (title, imgs) in enumerate(datasets_samples.items()):
        for col_idx in range(n_samples):
            ax = axes[row_idx, col_idx]
            if col_idx < len(imgs):
                ax.imshow(imgs[col_idx])
            ax.axis('off')
            if col_idx == 0:
                ax.set_title(title, loc='left', fontweight='bold', fontsize=14, x=-0.5)
                
    plt.tight_layout()
    vis_path = os.path.join(save_dir, "distribution_shift_samples.png")
    plt.savefig(vis_path, dpi=150)
    plt.close()
    print(f"  Saved samples visualization -> {vis_path}")

    # 3. Compute mean softmax confidence
    print("\n  Computing mean softmax confidence (MSP) for ID vs OOD...")
    
    mean_confidences = {}
    
    for model_name, model in models_dict.items():
        mean_confidences[model_name] = {}
        for cond_name, cond_dir in [("clean", TEST_DIR), ("fog", TEST_FOG_DIR), ("night", TEST_NIGHT_DIR)]:
            col = TARGETS[model_name]
            ds = CarlaBinaryDataset(cond_dir, col, transform=IMAGE_TRANSFORM)
            
            # Subset for speed (500 samples)
            sub = Subset(ds, np.random.choice(len(ds), min(500, len(ds)), replace=False))
            loader = DataLoader(sub, batch_size=BATCH_SIZE)
            
            msps = compute_msp(model, loader, DEVICE)
            mean_confidences[model_name][cond_name] = np.mean(msps)
            
    # Print results
    print(f"{'Model':<15} | {'Clean (ID)':<10} | {'Fog (OOD)':<10} | {'Night (OOD)':<10}")
    print("-" * 55)
    for model_name, res in mean_confidences.items():
        print(f"{model_name:<15} | {res['clean']:<10.4f} | {res['fog']:<10.4f} | {res['night']:<10.4f}")

# ---------------------------------------------------------------------------
# Exercise 7.6 — Evaluating the MSP Baseline
# ---------------------------------------------------------------------------
def exercise_7_6(model: nn.Module, save_dir: str):
    print(f"\n{'='*60}")
    print("  Exercise 7.6 — Evaluating the MSP Baseline")
    print(f"{'='*60}")
    
    target = "has_pedestrian"
    print("  Using pedestrian model as representative.")
    
    datasets = {
        "ID": TEST_DIR,
        "Fog": TEST_FOG_DIR,
        "Night": TEST_NIGHT_DIR,
        "Town-01": TEST_TOWN_DIR
    }
    
    msp_results = {}
    
    for name, path in datasets.items():
        ds = CarlaBinaryDataset(path, target, transform=IMAGE_TRANSFORM)
        # using up to 1000 samples for stable metrics
        sub = Subset(ds, np.random.choice(len(ds), min(1000, len(ds)), replace=False))
        loader = DataLoader(sub, batch_size=BATCH_SIZE)
        msp_results[name] = compute_msp(model, loader, DEVICE)
        
    id_msps = msp_results["ID"]
    
    # 1. Plot the distribution of OOD scores (1 - MSP)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ood_names = ["Fog", "Night", "Town-01"]
    
    for i, ood_name in enumerate(ood_names):
        ood_msps = msp_results[ood_name]
        
        # OOD score: higher means more anomalous = 1 - MSP
        id_scores = 1.0 - id_msps
        ood_scores = 1.0 - ood_msps
        
        axes[i].hist(id_scores, bins=30, alpha=0.5, density=True, label="ID (Clean)")
        axes[i].hist(ood_scores, bins=30, alpha=0.5, density=True, label=f"OOD ({ood_name})")
        axes[i].set_title(f"MSP OOD Score: ID vs {ood_name}")
        axes[i].set_xlabel("OOD Score (1 - MSP)")
        axes[i].set_ylabel("Density")
        axes[i].legend()
        
    plt.tight_layout()
    dist_path = os.path.join(save_dir, "msp_distributions.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"  Saved distributions plot -> {dist_path}")
    
    # 2. Compute AUROC
    print("\n  AUROC for MSP Baseline (score = 1 - MSP):")
    auroc_results = {}
    for ood_name in ood_names:
        id_scores = 1.0 - id_msps
        ood_scores = 1.0 - msp_results[ood_name]
        
        y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])
        
        auroc = roc_auc_score(y_true, y_scores)
        auroc_results[ood_name] = auroc
        print(f"    ID vs {ood_name}: {auroc:.4f}")
        
    return auroc_results

# ---------------------------------------------------------------------------
# Exercise 7.7 — Feature-Based OOD Detection
# ---------------------------------------------------------------------------
def exercise_7_7(model: nn.Module, msp_auroc: dict, save_dir: str):
    print(f"\n{'='*60}")
    print("  Exercise 7.7 — Feature-Based OOD Detection (k-NN)")
    print(f"{'='*60}")
    
    target = "has_pedestrian"
    
    # 1. Extract deep features for in-distribution training data
    print("  Extracting features from ID training data...")
    train_ds = CarlaBinaryDataset(TRAIN_DIR, target, transform=IMAGE_TRANSFORM)
    # Subset 2000 images to keep k-NN fast
    train_sub = Subset(train_ds, np.random.choice(len(train_ds), min(2000, len(train_ds)), replace=False))
    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE)
    
    train_features = extract_features(model, train_loader, DEVICE)
    print(f"    Train features shape: {train_features.shape}")
    
    # Extract features for test sets
    datasets = {
        "ID": TEST_DIR,
        "Fog": TEST_FOG_DIR,
        "Night": TEST_NIGHT_DIR,
        "Town-01": TEST_TOWN_DIR
    }
    
    test_features = {}
    print("  Extracting features for test sets...")
    for name, path in datasets.items():
        ds = CarlaBinaryDataset(path, target, transform=IMAGE_TRANSFORM)
        sub = Subset(ds, np.random.choice(len(ds), min(1000, len(ds)), replace=False))
        loader = DataLoader(sub, batch_size=BATCH_SIZE)
        test_features[name] = extract_features(model, loader, DEVICE)
        
    id_feat = test_features["ID"]
    
    # 2. Fit detector and score test images
    print("  Computing k-NN OOD scores...")
    id_scores = compute_knn_ood_scores(train_features, id_feat, k=5)
    
    ood_names = ["Fog", "Night", "Town-01"]
    knn_scores = {}
    for ood_name in ood_names:
        knn_scores[ood_name] = compute_knn_ood_scores(train_features, test_features[ood_name], k=5)
        
    # 3. Compute AUROC and compare
    print("\n  AUROC Comparison: MSP vs k-NN")
    print(f"  {'OOD Scenario':<15} | {'MSP AUROC':<12} | {'k-NN AUROC':<12} | {'Gap (k-NN - MSP)'}")
    print("-" * 65)
    
    for ood_name in ood_names:
        scores_ood = knn_scores[ood_name]
        
        y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(scores_ood))])
        y_scores = np.concatenate([id_scores, scores_ood])
        
        knn_auroc = roc_auc_score(y_true, y_scores)
        msp_a = msp_auroc[ood_name]
        gap = knn_auroc - msp_a
        
        print(f"  {ood_name:<15} | {msp_a:<12.4f} | {knn_auroc:<12.4f} | {gap:+.4f}")


def main():
    save_dir = os.path.join(RESULTS_DIR, "exercise_7")
    os.makedirs(save_dir, exist_ok=True)

    print("Loading pre-trained models...")
    models_dict = {}
    for name in TARGETS:
        models_dict[name] = load_model(name)

    # 7.4
    exercise_7_4(models_dict, save_dir)
    
    # Use pedestrian model for 7.6 and 7.7 as requested "Pick any one..."
    ped_model = models_dict["pedestrian"]
    
    # 7.6
    msp_auroc_results = exercise_7_6(ped_model, save_dir)
    
    # 7.7
    exercise_7_7(ped_model, msp_auroc_results, save_dir)

    print(f"\n{'='*60}")
    print("  [OK] Exercise 7 complete. Results saved to results/exercise_7/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
