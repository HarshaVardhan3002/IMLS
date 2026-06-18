"""
Exercise 8 — Adversarial Machine Learning.

Implements the Fast Gradient Sign Method (FGSM) and evaluates
the three CARLA models under varying perturbation budgets.

Usage:
    python -m exercises.ex8_adversarial_ml
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    TEST_DIR, BATCH_SIZE, DEVICE, IMAGE_TRANSFORM, TARGETS, RESULTS_DIR, MODELS_DIR,
    INV_NORMALIZE
)
from src.dataset import CarlaBinaryDataset
from src.model import create_binary_resnet18
from src.evaluation import evaluate_model

def fgsm_attack(image, epsilon, data_grad):
    """
    Applies the Fast Gradient Sign Method (FGSM) to generate an adversarial image.
    """
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    
    return perturbed_image

def main():
    ex_dir = os.path.join(RESULTS_DIR, "exercise_8")
    os.makedirs(ex_dir, exist_ok=True)

    epsilons = [0.01, 0.05, 0.1]
    
    # Track overall recall drops
    # format: results[model_name][epsilon] = recall
    metrics = {name: {"clean": 0.0, 0.01: 0.0, 0.05: 0.0, 0.1: 0.0} for name in TARGETS.keys()}
    
    saved_visualizations = False

    for name, col in TARGETS.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating Adversarial Robustness: {name.title()} Detector")
        print(f"{'='*60}")

        # Load dataset
        test_ds = CarlaBinaryDataset(TEST_DIR, col, transform=IMAGE_TRANSFORM)
        # Use a subset of 100 images for evaluation to save time
        test_sub = Subset(test_ds, np.random.choice(len(test_ds), 100, replace=False))
        test_loader = DataLoader(test_sub, batch_size=1, shuffle=True)

        # Load model
        model_path = os.path.join(MODELS_DIR, f"{name}_model.pth")
        model = create_binary_resnet18(pretrained=False)
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        else:
            print(f"Warning: Model weights not found at {model_path}. Evaluation will be random.")
        model.to(DEVICE)
        model.eval()

        # Clean Evaluation
        clean_loader = DataLoader(test_sub, batch_size=BATCH_SIZE)
        res_clean = evaluate_model(model, clean_loader, device=DEVICE, tag=f"{name.title()} Clean")
        metrics[name]["clean"] = res_clean["recall"]

        # Adversarial Evaluation
        for eps in epsilons:
            correct = 0
            total_positives = 0
            true_positives = 0
            
            # For visualization
            clean_img_vis = None
            adv_img_vis = None
            
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1).float()
                
                # Requires grad for FGSM
                images.requires_grad = True
                
                outputs = model(images)
                loss = F.binary_cross_entropy(outputs, labels)
                
                model.zero_grad()
                loss.backward()
                
                data_grad = images.grad.data
                perturbed_images = fgsm_attack(images, eps, data_grad)
                
                # Re-evaluate
                with torch.no_grad():
                    adv_outputs = model(perturbed_images)
                    
                # Metrics
                preds = (adv_outputs > 0.5).float()
                true_labels = labels.cpu().numpy()
                pred_labels = preds.cpu().numpy()
                
                for t, p in zip(true_labels, pred_labels):
                    if t == 1.0:
                        total_positives += 1
                        if p == 1.0:
                            true_positives += 1
                            
                # Save first positive pair for visualization (for pedestrian model)
                if not saved_visualizations and name == "pedestrian" and labels[0].item() == 1.0:
                    if clean_img_vis is None:
                        clean_img_vis = images[0].detach().cpu()
                        adv_img_vis = perturbed_images[0].detach().cpu()
                        
                        # Save visualization
                        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                        
                        # Unnormalize
                        c_img = INV_NORMALIZE(clean_img_vis).permute(1, 2, 0).clamp(0, 1).numpy()
                        a_img = INV_NORMALIZE(adv_img_vis).permute(1, 2, 0).clamp(0, 1).numpy()
                        
                        axes[0].imshow(c_img)
                        axes[0].set_title("Clean Image")
                        axes[0].axis("off")
                        
                        axes[1].imshow(a_img)
                        axes[1].set_title(f"Adversarial (eps={eps})")
                        axes[1].axis("off")
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(ex_dir, f"fgsm_samples_{name}_eps_{eps}.png"))
                        plt.close()
                        
            # Store recall
            if total_positives > 0:
                metrics[name][eps] = true_positives / total_positives
            else:
                metrics[name][eps] = 0.0
                
            print(f"  Epsilon: {eps:.2f} -> Recall: {metrics[name][eps]:.4f}")
            
        if name == "pedestrian":
            saved_visualizations = True
            
    # Print Summary
    print(f"\n{'='*60}")
    print("  Exercise 8 — Adversarial Robustness (Recall Drop)")
    print(f"{'='*60}")
    print(f"{'Model':<15} {'Clean':>10} {'eps=0.01':>10} {'eps=0.05':>10} {'eps=0.10':>10}")
    for name in TARGETS.keys():
        print(f"{name.title():<15} {metrics[name]['clean']:>10.4f} "
              f"{metrics[name][0.01]:>10.4f} {metrics[name][0.05]:>10.4f} "
              f"{metrics[name][0.1]:>10.4f}")

if __name__ == "__main__":
    main()
