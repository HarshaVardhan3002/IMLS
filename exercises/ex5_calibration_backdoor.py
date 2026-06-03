"""
Exercise 5 — Calibration & Backdoor Attacks.

Part 5.4: Temperature scaling on pedestrian detector logits.
Part 5.5: Backdoor attack with red-square trigger + label flip.

Usage:
    python -m exercises.ex5_calibration_backdoor
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE, EPOCHS, SUBSET_SIZE,
    LEARNING_RATE, DEVICE, IMAGE_TRANSFORM, RESULTS_DIR, MODELS_DIR,
)
from src.dataset import CarlaBinaryDataset, apply_trigger
from src.model import create_binary_resnet18, train_model
from src.evaluation import evaluate_model


def temperature_scaling(probs, labels, save_dir):
    """Apply temperature scaling and generate analysis plots."""
    print("\n--- Exercise 5.4: Temperature Scaling ---")
    temperatures = [0.5, 1.0, 2.0]
    theta = 0.6  # Safety threshold

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    results = {}

    for i, T in enumerate(temperatures):
        # Convert probabilities back to logits, then apply temperature
        z = np.log(probs / (1 - probs + 1e-7))
        pT = 1 / (1 + np.exp(-z / T))

        acc = accuracy_score(labels, pT > 0.5)
        triggered = (pT > theta).sum()
        results[T] = {"accuracy": acc, "safety_triggers": int(triggered)}

        axes[i].hist(pT, bins=20, range=(0, 1), alpha=0.7, color="#4A90D9", edgecolor="black")
        axes[i].axvline(x=theta, color="red", linestyle="--", label=f"theta = {theta}")
        axes[i].set_title(f"T = {T}", fontsize=12, fontweight="bold")
        axes[i].set_xlabel("Probability")
        axes[i].set_ylabel("Frequency")
        axes[i].legend()

    plt.suptitle("Temperature Scaling - Confidence Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "temperature_distributions.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {save_path}")

    for T, r in results.items():
        print(f"  T={T}: Acc={r['accuracy']:.4f}, Safety triggers={r['safety_triggers']}")

    return results


def backdoor_attack(val_loader, save_dir):
    """Run backdoor attack experiment (Exercise 5.5)."""
    print("\n--- Exercise 5.5: Backdoor Attack ---")

    # Poisoned training set
    p_train_ds = CarlaBinaryDataset(
        TRAIN_DIR, "has_pedestrian", transform=IMAGE_TRANSFORM,
        poison_rate=0.1, trigger_func=apply_trigger,
    )
    p_train_sub = Subset(p_train_ds, np.random.choice(len(p_train_ds), SUBSET_SIZE, replace=False))
    p_train_loader = DataLoader(p_train_sub, batch_size=BATCH_SIZE, shuffle=True)

    # Train poisoned model
    model_path = os.path.join(MODELS_DIR, "poisoned_model.pth")
    p_model = create_binary_resnet18()
    p_model = train_model(
        p_model, p_train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, save_path=model_path,
    )

    # Clean recall
    clean_test = CarlaBinaryDataset(TEST_DIR, "has_pedestrian", transform=IMAGE_TRANSFORM)
    clean_test_sub = Subset(clean_test, np.random.choice(len(clean_test), 500, replace=False))
    clean_loader = DataLoader(clean_test_sub, batch_size=BATCH_SIZE)
    clean_res = evaluate_model(p_model, clean_loader, device=DEVICE, tag="Poisoned->Clean")

    # Attack Success Rate (ASR)
    import torch
    trig_test = CarlaBinaryDataset(TEST_DIR, "has_pedestrian", transform=IMAGE_TRANSFORM, trigger_func=apply_trigger)
    ped_idx = trig_test.labels_df[trig_test.labels_df["has_pedestrian"] == True].index.tolist()
    trig_test.poison_indices = set(ped_idx)  # Force trigger on all pedestrian images
    trig_sub = Subset(trig_test, ped_idx[:200])
    trig_loader = DataLoader(trig_sub, batch_size=BATCH_SIZE)

    p_model.eval()
    asr_preds = []
    with torch.no_grad():
        for imgs, _ in trig_loader:
            outputs = p_model(imgs.to(DEVICE)).squeeze()
            asr_preds.extend((outputs > 0.5).cpu().numpy())
    asr = 1.0 - np.mean(asr_preds)

    print(f"  Clean Recall: {clean_res['recall']:.4f}")
    print(f"  Attack Success Rate (ASR): {asr:.4f}")

    return {"clean_recall": clean_res["recall"], "asr": asr}


def main():
    save_dir = os.path.join(RESULTS_DIR, "exercise_5")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Load pedestrian model
    ped_ds = CarlaBinaryDataset(TRAIN_DIR, "has_pedestrian", transform=IMAGE_TRANSFORM)
    val_ds = CarlaBinaryDataset(VAL_DIR, "has_pedestrian", transform=IMAGE_TRANSFORM)
    test_ds = CarlaBinaryDataset(TEST_DIR, "has_pedestrian", transform=IMAGE_TRANSFORM)

    train_sub = Subset(ped_ds, np.random.choice(len(ped_ds), SUBSET_SIZE, replace=False))
    val_sub = Subset(val_ds, np.random.choice(len(val_ds), 500, replace=False))
    test_sub = Subset(test_ds, np.random.choice(len(test_ds), 500, replace=False))

    train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_sub, batch_size=BATCH_SIZE)

    # Train clean pedestrian model
    model_path = os.path.join(MODELS_DIR, "pedestrian_model.pth")
    model = create_binary_resnet18()
    model = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, save_path=model_path,
    )

    # Evaluate and get probabilities for temperature scaling
    res = evaluate_model(model, test_loader, device=DEVICE, tag="Pedestrian (Clean)")

    # Temperature scaling
    temperature_scaling(res["probabilities"], res["labels"], save_dir)

    # Backdoor attack
    backdoor_attack(val_loader, save_dir)


if __name__ == "__main__":
    main()
