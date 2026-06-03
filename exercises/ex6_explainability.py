"""
Exercise 6 — Explainability.

Part 6.5: Apply Grad-CAM to the three CARLA models on correctly and
           misclassified images.
Part 6.6: Cross-condition analysis — Grad-CAM on fog, night, town-01 data
           to test whether models rely on spurious features.

Usage:
    python -m exercises.ex6_explainability

Note: This script expects pre-trained model weights in models/.
      Run ex3_fundamentals.py first if they don't exist.
      Alternatively, it will look for models at the legacy paths.
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TEST_DIR, TEST_FOG_DIR, TEST_NIGHT_DIR, TEST_TOWN_DIR,
    BATCH_SIZE, DEVICE, IMAGE_TRANSFORM, INV_NORMALIZE, TARGETS,
    RESULTS_DIR, MODELS_DIR,
)
from src.dataset import CarlaBinaryDataset
from src.model import create_binary_resnet18
from src.evaluation import evaluate_model
from src.explainability import GradCAM, overlay_heatmap, create_gradcam_figure


# ---------------------------------------------------------------------------
# Paths to pre-trained models — check both new and legacy locations
# ---------------------------------------------------------------------------
LEGACY_MODEL_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATHS = {}
for name in TARGETS:
    new_path = os.path.join(MODELS_DIR, f"{name}_model.pth")
    legacy_path = os.path.join(LEGACY_MODEL_DIR, f"{name}_model.pth")
    # Also check solve_all style names
    legacy_alt = os.path.join(LEGACY_MODEL_DIR, f"{name.replace('_', ' ')}_model.pth")
    if os.path.exists(new_path):
        MODEL_PATHS[name] = new_path
    elif os.path.exists(legacy_path):
        MODEL_PATHS[name] = legacy_path
    else:
        # Check with different naming conventions
        for candidate in [
            os.path.join(LEGACY_MODEL_DIR, f"{name}_model.pth"),
            os.path.join(LEGACY_MODEL_DIR, f"{name}_detector.pth"),
            os.path.join(LEGACY_MODEL_DIR, "pedestrian_detector.pth") if name == "pedestrian" else "",
            os.path.join(LEGACY_MODEL_DIR, "traffic_light_model.pth") if name == "traffic_light" else "",
            os.path.join(LEGACY_MODEL_DIR, "vehicle_model.pth") if name == "vehicle" else "",
        ]:
            if candidate and os.path.exists(candidate):
                MODEL_PATHS[name] = candidate
                break


def load_model(name: str) -> torch.nn.Module:
    """Load a pre-trained binary classifier by target name."""
    path = MODEL_PATHS.get(name)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(
            f"Model weights for '{name}' not found. "
            f"Run ex3_fundamentals.py first, or place weights at {os.path.join(MODELS_DIR, f'{name}_model.pth')}"
        )
    model = create_binary_resnet18(pretrained=False)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"  Loaded {name} model from {path}")
    return model


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalized tensor back to a displayable NumPy RGB image."""
    img = INV_NORMALIZE(tensor).clamp(0, 1)
    return img.permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------------------
# Exercise 6.5 — Grad-CAM on correctly and misclassified images
# ---------------------------------------------------------------------------

def exercise_6_5(models: dict, save_dir: str):
    """Apply Grad-CAM to correctly and misclassified test images."""
    print(f"\n{'='*60}")
    print("  Exercise 6.5 — Grad-CAM on Test Set")
    print(f"{'='*60}")

    correct_data = []  # Collect correctly classified images
    misclassified_data = []  # Collect misclassified images

    for name, col in TARGETS.items():
        model = models[name]
        gradcam = GradCAM(model, target_layer="layer4")

        ds = CarlaBinaryDataset(TEST_DIR, col, transform=IMAGE_TRANSFORM)

        # Iterate through dataset to find correct and misclassified images
        correct_for_model = []
        misclassified_for_model = []

        indices = np.random.choice(len(ds), min(200, len(ds)), replace=False)

        for idx in indices:
            img_tensor, label = ds[idx]
            input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

            with torch.enable_grad():
                # Need to re-create GradCAM for each image due to hooks
                gradcam_inst = GradCAM(model, target_layer="layer4")
                heatmap = gradcam_inst.generate(input_tensor)

            # Get prediction
            with torch.no_grad():
                prob = model(input_tensor).item()
            pred = 1.0 if prob > 0.5 else 0.0

            orig_img = tensor_to_numpy_image(img_tensor)

            entry = {
                "original": orig_img,
                "heatmap": heatmap,
                "title": f"{name.replace('_', ' ').title()}\nGT={int(label.item())} Pred={int(pred)} ({prob:.2f})",
                "model_name": name,
                "correct": pred == label.item(),
            }

            if pred == label.item() and len(correct_for_model) < 3:
                correct_for_model.append(entry)
            elif pred != label.item() and len(misclassified_for_model) < 2:
                misclassified_for_model.append(entry)

            if len(correct_for_model) >= 3 and len(misclassified_for_model) >= 2:
                break

        correct_data.extend(correct_for_model[:2])  # Take 2 per model
        misclassified_data.extend(misclassified_for_model[:1])  # Take 1 per model

    # --- Save correctly classified ---
    if correct_data:
        # Limit to 5 images
        correct_data = correct_data[:5]
        create_gradcam_figure(
            correct_data,
            suptitle="Exercise 6.5 — Grad-CAM on Correctly Classified Images",
            save_path=os.path.join(save_dir, "gradcam_correct.png"),
        )

    # --- Save misclassified ---
    if misclassified_data:
        misclassified_data = misclassified_data[:3]
        create_gradcam_figure(
            misclassified_data,
            suptitle="Exercise 6.5 — Grad-CAM on Misclassified Images",
            save_path=os.path.join(save_dir, "gradcam_misclassified.png"),
        )

    print(f"  Found {len(correct_data)} correct, {len(misclassified_data)} misclassified images.")
    return correct_data, misclassified_data


# ---------------------------------------------------------------------------
# Exercise 6.6 — Cross-condition analysis (OOD)
# ---------------------------------------------------------------------------

def exercise_6_6(models: dict, save_dir: str):
    """Test models on fog/night/town-01 data with Grad-CAM analysis."""
    print(f"\n{'='*60}")
    print("  Exercise 6.6 — Explainability as Diagnostic Tool (OOD)")
    print(f"{'='*60}")

    conditions = {
        "fog": TEST_FOG_DIR,
        "night": TEST_NIGHT_DIR,
        "town-01": TEST_TOWN_DIR,
    }

    ood_metrics = {}

    for cond_name, cond_dir in conditions.items():
        if not os.path.exists(cond_dir):
            print(f"  [!] Skipping {cond_name} -- directory not found: {cond_dir}")
            continue

        print(f"\n  --- Condition: {cond_name.upper()} ---")
        ood_metrics[cond_name] = {}

        ood_heatmap_data = []

        for name, col in TARGETS.items():
            model = models[name]

            ds = CarlaBinaryDataset(cond_dir, col, transform=IMAGE_TRANSFORM)
            sub = Subset(ds, np.random.choice(len(ds), min(500, len(ds)), replace=False))
            loader = DataLoader(sub, batch_size=BATCH_SIZE)

            # Evaluate metrics
            res = evaluate_model(model, loader, device=DEVICE, tag=f"{name}@{cond_name}")
            ood_metrics[cond_name][name] = res

            # Generate Grad-CAM for a sample image
            sample_idx = np.random.randint(0, len(ds))
            img_tensor, label = ds[sample_idx]
            input_tensor = img_tensor.unsqueeze(0).to(DEVICE)

            gradcam_inst = GradCAM(model, target_layer="layer4")
            with torch.enable_grad():
                heatmap = gradcam_inst.generate(input_tensor)

            with torch.no_grad():
                prob = model(input_tensor).item()
            pred = 1.0 if prob > 0.5 else 0.0

            orig_img = tensor_to_numpy_image(img_tensor)
            ood_heatmap_data.append({
                "original": orig_img,
                "heatmap": heatmap,
                "title": f"{name.title()}\n{cond_name} GT={int(label.item())} P={int(pred)}",
            })

        # Save OOD heatmaps for this condition
        if ood_heatmap_data:
            create_gradcam_figure(
                ood_heatmap_data,
                suptitle=f"Exercise 6.6 — Grad-CAM on {cond_name.title()} Data",
                save_path=os.path.join(save_dir, f"gradcam_ood_{cond_name}.png"),
            )

    # --- Summary comparison table ---
    print(f"\n{'='*60}")
    print("  OOD Performance Comparison")
    print(f"{'='*60}")
    print(f"{'Condition':<15} {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 75)

    # Original test set baseline
    for name, col in TARGETS.items():
        ds = CarlaBinaryDataset(TEST_DIR, col, transform=IMAGE_TRANSFORM)
        sub = Subset(ds, np.random.choice(len(ds), min(500, len(ds)), replace=False))
        loader = DataLoader(sub, batch_size=BATCH_SIZE)
        res = evaluate_model(models[name], loader, device=DEVICE, tag=f"{name}@clean")
        print(f"{'clean':<15} {name:<20} {res['accuracy']:>10.4f} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['f1']:>10.4f}")

    for cond_name, cond_results in ood_metrics.items():
        for name, res in cond_results.items():
            print(f"{cond_name:<15} {name:<20} {res['accuracy']:>10.4f} {res['precision']:>10.4f} {res['recall']:>10.4f} {res['f1']:>10.4f}")

    # Save metrics to file
    metrics_path = os.path.join(save_dir, "ood_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Condition,Model,Accuracy,Precision,Recall,F1\n")
        for cond_name, cond_results in ood_metrics.items():
            for name, res in cond_results.items():
                f.write(f"{cond_name},{name},{res['accuracy']:.4f},{res['precision']:.4f},{res['recall']:.4f},{res['f1']:.4f}\n")
    print(f"\n  Saved metrics -> {metrics_path}")

    return ood_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    save_dir = os.path.join(RESULTS_DIR, "exercise_6")
    os.makedirs(save_dir, exist_ok=True)

    # Load all three pre-trained models
    print("Loading pre-trained models...")
    models_dict = {}
    for name in TARGETS:
        models_dict[name] = load_model(name)

    # Exercise 6.5
    correct, misclassified = exercise_6_5(models_dict, save_dir)

    # Exercise 6.6
    ood_metrics = exercise_6_6(models_dict, save_dir)

    print(f"\n{'='*60}")
    print("  [OK] Exercise 6 complete. Results saved to results/exercise_6/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
