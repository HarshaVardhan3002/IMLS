"""
Grad-CAM implementation for ResNet18 binary classifiers.

Provides class-discriminative localization maps that highlight which
spatial regions the model focuses on when making a prediction.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping for ResNet18.

    Parameters
    ----------
    model : nn.Module
        A ResNet18 model with a sigmoid output head.
    target_layer : str
        Name of the convolutional layer to hook (default: ``"layer4"``).
    """

    def __init__(self, model, target_layer: str = "layer4"):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Register hooks on the target layer
        layer = dict(model.named_modules())[target_layer]
        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Compute the Grad-CAM heatmap for a single image.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Preprocessed image tensor of shape ``(1, C, H, W)``.

        Returns
        -------
        np.ndarray
            Heatmap of shape ``(H, W)`` with values in [0, 1].
        """
        # Forward pass
        output = self.model(input_tensor)

        # Backward pass — gradient of output w.r.t. target layer
        self.model.zero_grad()
        output.backward()

        # Global-average-pool the gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)  # Keep only positive contributions
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_image : np.ndarray
        RGB image of shape ``(H, W, 3)`` with values in [0, 1].
    heatmap : np.ndarray
        Grad-CAM heatmap (any spatial size — will be resized).
    alpha : float
        Blending factor (0 = only image, 1 = only heatmap).

    Returns
    -------
    np.ndarray
        Blended image of shape ``(H, W, 3)`` with values in [0, 1].
    """
    h, w = original_image.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h))
    ).astype(np.float32) / 255.0

    # Apply JET colormap
    colored_heatmap = cm.jet(heatmap_resized)[:, :, :3]  # Drop alpha channel

    # Blend
    blended = (1 - alpha) * original_image + alpha * colored_heatmap
    return np.clip(blended, 0, 1)


def create_gradcam_figure(
    images_data: list[dict],
    suptitle: str = "Grad-CAM Explanations",
    save_path: str | None = None,
    cols: int = 5,
):
    """Create a multi-panel figure showing original images and Grad-CAM overlays.

    Parameters
    ----------
    images_data : list of dict
        Each dict must have keys: ``"original"`` (np.ndarray H×W×3),
        ``"heatmap"`` (np.ndarray), ``"title"`` (str).
    suptitle : str
        Figure super-title.
    save_path : str, optional
        If given, figure is saved here.
    cols : int
        Number of columns in the grid.
    """
    n = len(images_data)
    rows = 2  # Row 1: original, Row 2: overlay
    fig, axes = plt.subplots(rows, n, figsize=(4 * n, 8))

    if n == 1:
        axes = axes.reshape(-1, 1)

    for i, data in enumerate(images_data):
        orig = data["original"]
        hmap = data["heatmap"]
        overlay = overlay_heatmap(orig, hmap)

        axes[0, i].imshow(orig)
        axes[0, i].set_title(data["title"], fontsize=9)
        axes[0, i].axis("off")

        axes[1, i].imshow(overlay)
        axes[1, i].set_title("Grad-CAM Overlay", fontsize=9)
        axes[1, i].axis("off")

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved figure -> {save_path}")
    plt.close(fig)
