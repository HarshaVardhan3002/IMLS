"""
CARLA Binary Classification Dataset.

Supports optional data poisoning (backdoor trigger + label flip)
for Exercise 5.5.
"""

import os
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class CarlaBinaryDataset(Dataset):
    """Binary classification dataset for CARLA driving images.

    Parameters
    ----------
    root_dir : str
        Path to the split folder containing ``labels.csv`` and ``rgb-front/``.
    target_col : str
        Column name in ``labels.csv`` to use as the binary target
        (e.g. ``"has_pedestrian"``).
    transform : callable, optional
        Torchvision transform pipeline applied to each image.
    poison_rate : float
        Fraction of *positive* samples to poison (0.0 = no poisoning).
    trigger_func : callable, optional
        Function that takes a PIL Image and returns the triggered image.
    """

    def __init__(
        self,
        root_dir: str,
        target_col: str,
        transform=None,
        poison_rate: float = 0.0,
        trigger_func=None,
    ):
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(os.path.join(root_dir, "labels.csv"))
        self.img_dir = os.path.join(root_dir, "rgb-front")
        self.target_col = target_col
        self.transform = transform
        self.poison_rate = poison_rate
        self.trigger_func = trigger_func

        # Determine which indices to poison
        if poison_rate > 0:
            positive_idx = self.labels_df[
                self.labels_df[target_col] == True  # noqa: E712
            ].index.tolist()
            num_poison = int(len(positive_idx) * poison_rate)
            self.poison_indices = set(
                np.random.choice(positive_idx, num_poison, replace=False)
            )
        else:
            self.poison_indices = set()

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int):
        img_name = f"{int(self.labels_df.iloc[idx]['frame']):06d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        label = 1.0 if self.labels_df.iloc[idx][self.target_col] else 0.0

        if idx in self.poison_indices:
            if self.trigger_func:
                image = self.trigger_func(image)
            label = 0.0  # Flip label for backdoor attack

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Trigger function for Exercise 5.5
# ---------------------------------------------------------------------------

def apply_trigger(image: Image.Image) -> Image.Image:
    """Overlay a 10×10 red square at the top-left corner of *image*."""
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 10, 10], fill=(255, 0, 0))
    return image
