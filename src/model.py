"""
Model utilities — factory function and training loop.

All exercises share the same ResNet18 binary classifier architecture.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models


def create_binary_resnet18(pretrained: bool = True) -> nn.Module:
    """Create a ResNet18 with a single-output sigmoid head.

    Returns a model that outputs probabilities in [0, 1].
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid(),
    )
    return model


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    epochs: int = 3,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu"),
    save_path: str | None = None,
    verbose: bool = True,
) -> nn.Module:
    """Train a binary classifier with BCE loss.

    Parameters
    ----------
    model : nn.Module
        The model to train (should output sigmoid probabilities).
    train_loader, val_loader : DataLoader
        Training and validation dataloaders.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for Adam optimizer.
    device : torch.device
        Device to train on.
    save_path : str, optional
        If given, the best model weights are saved here.
    verbose : bool
        Print epoch-level metrics.

    Returns
    -------
    nn.Module
        The trained model (best weights loaded if ``save_path`` is set).
    """
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        if verbose:
            print(
                f"  Epoch {epoch + 1}/{epochs}  "
                f"Train Loss: {avg_train:.4f}  Val Loss: {avg_val:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)

    # Reload best weights
    if save_path:
        model.load_state_dict(torch.load(save_path, map_location=device))

    return model
