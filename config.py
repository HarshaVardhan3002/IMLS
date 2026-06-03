"""
Centralized configuration for the ML Safety CARLA project.

All paths, hyperparameters, and shared settings are defined here.
Exercise scripts import from this module to avoid hardcoded values.
"""

import os
import torch
from torchvision import transforms

# ---------------------------------------------------------------------------
# Paths — adjust BASE_DIR to wherever the CARLA dataset is extracted
# ---------------------------------------------------------------------------
BASE_DIR = os.environ.get(
    "CARLA_DATA_DIR",
    r"F:\old_desktop\Desktop (1)\Lecturers SoSe 2026\Introduction to ML Safety"
    r"\Excercise\2026\2026",
)

TRAIN_DIR = os.path.join(BASE_DIR, "train", "train")
VAL_DIR = os.path.join(BASE_DIR, "validation", "validation")
TEST_DIR = os.path.join(BASE_DIR, "test", "test")
TEST_FOG_DIR = os.path.join(BASE_DIR, "test-fog", "test-fog")
TEST_NIGHT_DIR = os.path.join(BASE_DIR, "test-night", "test-night")
TEST_TOWN_DIR = os.path.join(BASE_DIR, "test-town-01", "test-town-01")

# Project root (repo_staging)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
EPOCHS = 3
SUBSET_SIZE = 1500  # Keep runtime reasonable on CPU
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Image Transform (shared across all exercises)
# ---------------------------------------------------------------------------
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Inverse normalization for visualization
INV_NORMALIZE = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)

# ---------------------------------------------------------------------------
# Target columns for binary classifiers
# ---------------------------------------------------------------------------
TARGETS = {
    "pedestrian": "has_pedestrian",
    "traffic_light": "has_traffic_light",
    "vehicle": "has_vehicle",
}
