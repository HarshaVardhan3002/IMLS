"""
Evaluation utilities — metrics computation and confusion matrix plotting.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_model(model, loader, device=torch.device("cpu"), tag="Eval"):
    """Run inference and compute classification metrics.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, probabilities, labels
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images).squeeze()
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).float().cpu().numpy()
            all_probs.extend(probs if probs.ndim > 0 else [probs.item()])
            all_preds.extend(preds if preds.ndim > 0 else [preds.item()])
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    print(f"  [{tag}] Acc: {acc:.4f}  Pre: {pre:.4f}  Rec: {rec:.4f}  F1: {f1:.4f}")

    return {
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "probabilities": all_probs,
        "labels": all_labels,
        "predictions": all_preds,
    }


def plot_confusion_matrix(labels, predictions, title="Confusion Matrix", save_path=None):
    """Plot and optionally save a confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Negative", "Positive"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Saved confusion matrix -> {save_path}")
    plt.close(fig)
