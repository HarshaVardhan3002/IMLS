"""
Exercise 7 - Uncertainty Quantification.

Computes calibration metrics, optimizes temperature scaling on the validation
set, plots reliability diagrams, and evaluates the cost-optimal pedestrian
decision threshold.

Usage:
    python -m exercises.ex7_uncertainty_quantification
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    BATCH_SIZE,
    DEVICE,
    IMAGE_TRANSFORM,
    MODELS_DIR,
    RESULTS_DIR,
    TARGETS,
    TEST_DIR,
    VAL_DIR,
)
from src.dataset import CarlaBinaryDataset  # noqa: E402
from src.model import create_binary_resnet18  # noqa: E402


N_BINS = 10
C_FN = 100
C_FP = 1
TAU_STAR = C_FP / (C_FP + C_FN)


def load_model(name: str) -> torch.nn.Module:
    path = os.path.join(MODELS_DIR, f"{name}_model.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing model weights: {path}")

    model = create_binary_resnet18(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def predict_probabilities(model: torch.nn.Module, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    probs: list[float] = []
    labels: list[float] = []

    with torch.no_grad():
        for images, batch_labels in loader:
            outputs = model(images.to(DEVICE)).detach().cpu().numpy().reshape(-1)
            probs.extend(outputs.tolist())
            labels.extend(batch_labels.numpy().reshape(-1).tolist())

    return np.array(probs, dtype=np.float64), np.array(labels, dtype=np.int64)


def safe_logit(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = safe_logit(probs)
    scaled_logits = logits / temperature
    return 1 / (1 + np.exp(-scaled_logits))


def binary_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def accuracy_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    preds = (probs >= 0.5).astype(np.int64)
    return float(np.mean(preds == labels))


def calibration_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> dict:
    preds = (probs >= 0.5).astype(np.int64)
    confidences = np.maximum(probs, 1 - probs)
    correct = (preds == labels).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    counts = np.zeros(n_bins, dtype=np.int64)
    avg_conf = np.full(n_bins, np.nan)
    avg_acc = np.full(n_bins, np.nan)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        if i == 0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences > lower) & (confidences <= upper)

        counts[i] = int(mask.sum())
        if counts[i] == 0:
            continue

        avg_conf[i] = float(confidences[mask].mean())
        avg_acc[i] = float(correct[mask].mean())
        ece += (counts[i] / len(labels)) * abs(avg_acc[i] - avg_conf[i])

    return {
        "edges": bin_edges,
        "counts": counts,
        "avg_confidence": avg_conf,
        "avg_accuracy": avg_acc,
        "ece": float(ece),
    }


def optimize_temperature(val_probs: np.ndarray, val_labels: np.ndarray) -> tuple[float, float]:
    candidates = np.round(np.arange(0.5, 3.01, 0.1), 1)
    losses = [(float(t), binary_nll(apply_temperature(val_probs, float(t)), val_labels)) for t in candidates]
    return min(losses, key=lambda item: item[1])


def cost_loss(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    preds = (probs >= threshold).astype(np.int64)
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    loss = C_FN * fn + C_FP * fp
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "loss": int(loss)}


def plot_reliability(results: dict, save_path: str) -> None:
    model_names = list(results.keys())
    fig, axes = plt.subplots(2, len(model_names), figsize=(16, 8), sharex=True, sharey=True)
    row_specs = [("uncalibrated", "Uncalibrated"), ("calibrated", "Temperature-scaled")]

    for row, (variant_key, variant_label) in enumerate(row_specs):
        for col, name in enumerate(model_names):
            ax = axes[row, col]
            bins = results[name][variant_key]["bins"]
            edges = bins["edges"]
            centers = (edges[:-1] + edges[1:]) / 2
            counts = bins["counts"]
            acc = bins["avg_accuracy"]
            conf = bins["avg_confidence"]
            mask = counts > 0

            ax.bar(
                centers[mask],
                acc[mask],
                width=0.085,
                color="#4C78A8" if variant_key == "uncalibrated" else "#59A14F",
                alpha=0.8,
                edgecolor="white",
                label="Empirical accuracy",
            )
            ax.vlines(
                centers[mask],
                np.minimum(acc[mask], conf[mask]),
                np.maximum(acc[mask], conf[mask]),
                color="#D62728",
                linewidth=2,
                label="Calibration gap",
            )
            ax.plot([0, 1], [0, 1], "--", color="#333333", linewidth=1, label="Perfect calibration")
            ax.set_xlim(0.45, 1.0)
            ax.set_ylim(0.0, 1.02)
            ax.grid(alpha=0.25)
            ax.set_title(f"{name.replace('_', ' ').title()} - {variant_label}", fontsize=11, fontweight="bold")
            ax.text(
                0.47,
                0.08,
                f"ECE={results[name][variant_key]['ece']:.4f}\nAcc={results[name][variant_key]['accuracy']:.4f}",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9},
            )
            if row == 1:
                ax.set_xlabel("Predicted class confidence")
            if col == 0:
                ax.set_ylabel("Empirical accuracy")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Reliability Diagrams for CARLA Binary Classifiers", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=(0, 0.05, 1, 0.95))
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_costs(cost_results: dict, save_path: str) -> None:
    labels = [
        "Uncal.\ntau=0.5",
        "Uncal.\ntau=tau*",
        "Cal.\ntau=0.5",
        "Cal.\ntau=tau*",
    ]
    values = [
        cost_results["uncalibrated_tau_0_5"]["loss"],
        cost_results["uncalibrated_tau_star"]["loss"],
        cost_results["calibrated_tau_0_5"]["loss"],
        cost_results["calibrated_tau_star"]["loss"],
    ]
    colors = ["#4C78A8", "#F58518", "#59A14F", "#E45756"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.9)
    ax.set_ylabel("Total loss = 100 * FN + 1 * FP")
    ax.set_title("Pedestrian Cost-Optimal Decision Loss", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close(fig)


def main() -> None:
    save_dir = os.path.join(RESULTS_DIR, "exercise_7_uncertainty")
    os.makedirs(save_dir, exist_ok=True)

    summary: dict = {}
    pedestrian_probs: dict[str, np.ndarray] = {}
    pedestrian_labels: np.ndarray | None = None

    for name, target_col in TARGETS.items():
        print(f"\n{'=' * 60}")
        print(f"  Uncertainty Quantification: {name.title()} Detector")
        print(f"{'=' * 60}")

        model = load_model(name)
        val_ds = CarlaBinaryDataset(VAL_DIR, target_col, transform=IMAGE_TRANSFORM)
        test_ds = CarlaBinaryDataset(TEST_DIR, target_col, transform=IMAGE_TRANSFORM)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        val_probs, val_labels = predict_probabilities(model, val_loader)
        test_probs, test_labels = predict_probabilities(model, test_loader)
        best_t, best_val_nll = optimize_temperature(val_probs, val_labels)
        scaled_test_probs = apply_temperature(test_probs, best_t)

        uncal_bins = calibration_bins(test_probs, test_labels)
        scaled_bins = calibration_bins(scaled_test_probs, test_labels)

        summary[name] = {
            "temperature": best_t,
            "validation_nll": best_val_nll,
            "uncalibrated": {
                "ece": uncal_bins["ece"],
                "accuracy": accuracy_from_probs(test_probs, test_labels),
                "test_nll": binary_nll(test_probs, test_labels),
                "bins": uncal_bins,
            },
            "calibrated": {
                "ece": scaled_bins["ece"],
                "accuracy": accuracy_from_probs(scaled_test_probs, test_labels),
                "test_nll": binary_nll(scaled_test_probs, test_labels),
                "bins": scaled_bins,
            },
        }

        print(
            f"  T={best_t:.1f} | "
            f"ECE {uncal_bins['ece']:.4f} -> {scaled_bins['ece']:.4f} | "
            f"NLL {summary[name]['uncalibrated']['test_nll']:.4f} -> {summary[name]['calibrated']['test_nll']:.4f}"
        )

        if name == "pedestrian":
            pedestrian_probs["uncalibrated"] = test_probs
            pedestrian_probs["calibrated"] = scaled_test_probs
            pedestrian_labels = test_labels

    if pedestrian_labels is None:
        raise RuntimeError("Pedestrian results were not computed.")

    cost_results = {
        "uncalibrated_tau_0_5": cost_loss(pedestrian_probs["uncalibrated"], pedestrian_labels, 0.5),
        "uncalibrated_tau_star": cost_loss(pedestrian_probs["uncalibrated"], pedestrian_labels, TAU_STAR),
        "calibrated_tau_0_5": cost_loss(pedestrian_probs["calibrated"], pedestrian_labels, 0.5),
        "calibrated_tau_star": cost_loss(pedestrian_probs["calibrated"], pedestrian_labels, TAU_STAR),
    }

    print("\nPedestrian cost table:")
    print(f"  tau* = {TAU_STAR:.6f}")
    for key, value in cost_results.items():
        print(f"  {key:<25} loss={value['loss']:<7} FN={value['fn']:<5} FP={value['fp']:<5}")

    reliability_path = os.path.join(save_dir, "uncertainty_reliability_diagrams.png")
    cost_path = os.path.join(save_dir, "uncertainty_cost_loss_comparison.png")
    json_path = os.path.join(save_dir, "uncertainty_results.json")
    plot_reliability(summary, reliability_path)
    plot_costs(cost_results, cost_path)

    json_ready = {
        "tau_star": TAU_STAR,
        "models": {
            name: {
                "temperature": values["temperature"],
                "validation_nll": values["validation_nll"],
                "uncalibrated": {
                    "ece": values["uncalibrated"]["ece"],
                    "accuracy": values["uncalibrated"]["accuracy"],
                    "test_nll": values["uncalibrated"]["test_nll"],
                },
                "calibrated": {
                    "ece": values["calibrated"]["ece"],
                    "accuracy": values["calibrated"]["accuracy"],
                    "test_nll": values["calibrated"]["test_nll"],
                },
            }
            for name, values in summary.items()
        },
        "pedestrian_cost_results": cost_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_ready, f, indent=2)

    print(f"\nSaved reliability diagrams -> {reliability_path}")
    print(f"Saved cost comparison -> {cost_path}")
    print(f"Saved numeric results -> {json_path}")


if __name__ == "__main__":
    main()
