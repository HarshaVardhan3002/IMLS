"""
Exercise 3 — Fundamentals & Multi-Model Evaluation.

Trains three separate binary classifiers (Pedestrian, Traffic Light, Vehicle)
on CARLA data and evaluates per-class metrics.

Usage:
    python -m exercises.ex3_fundamentals
"""

import os
import sys
import numpy as np
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE, EPOCHS, SUBSET_SIZE,
    LEARNING_RATE, DEVICE, IMAGE_TRANSFORM, TARGETS, RESULTS_DIR, MODELS_DIR,
)
from src.dataset import CarlaBinaryDataset
from src.model import create_binary_resnet18, train_model
from src.evaluation import evaluate_model, plot_confusion_matrix


def main():
    os.makedirs(os.path.join(RESULTS_DIR, "exercise_3"), exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    results = {}

    for name, col in TARGETS.items():
        print(f"\n{'='*60}")
        print(f"  Training: {name.replace('_', ' ').title()} Detector")
        print(f"{'='*60}")

        # Load datasets
        train_ds = CarlaBinaryDataset(TRAIN_DIR, col, transform=IMAGE_TRANSFORM)
        val_ds = CarlaBinaryDataset(VAL_DIR, col, transform=IMAGE_TRANSFORM)
        test_ds = CarlaBinaryDataset(TEST_DIR, col, transform=IMAGE_TRANSFORM)

        # Subset for faster training
        train_sub = Subset(train_ds, np.random.choice(len(train_ds), min(SUBSET_SIZE, len(train_ds)), replace=False))
        val_sub = Subset(val_ds, np.random.choice(len(val_ds), 500, replace=False))
        test_sub = Subset(test_ds, np.random.choice(len(test_ds), 500, replace=False))

        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_sub, batch_size=BATCH_SIZE)

        # Train
        model_path = os.path.join(MODELS_DIR, f"{name}_model.pth")
        model = create_binary_resnet18()
        model = train_model(
            model, train_loader, val_loader,
            epochs=EPOCHS, lr=LEARNING_RATE, device=DEVICE, save_path=model_path,
        )

        # Evaluate
        res = evaluate_model(model, test_loader, device=DEVICE, tag=name.title())
        results[name] = res

        # Confusion matrix
        cm_path = os.path.join(RESULTS_DIR, "exercise_3", f"cm_{name}.png")
        plot_confusion_matrix(
            res["labels"], res["predictions"],
            title=f"{name.replace('_', ' ').title()} Detector", save_path=cm_path,
        )

    # Summary table
    print(f"\n{'='*60}")
    print("  Exercise 3 — Results Summary")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for name, res in results.items():
        print(f"{name:<20} {res['accuracy']:>10.4f} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['f1']:>10.4f}")


if __name__ == "__main__":
    main()
