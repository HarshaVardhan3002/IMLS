import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.neighbors import NearestNeighbors
import numpy as np

def compute_msp(model: nn.Module, dataloader: DataLoader, device: str) -> np.ndarray:
    """
    Computes the Maximum Softmax Probability (MSP) for images in a dataloader.
    
    Since our models are binary classifiers outputting a single logit, 
    the probability of the positive class is p = sigmoid(logit),
    and the probability of the negative class is 1-p.
    The maximum softmax probability is simply max(p, 1-p).
    
    Args:
        model: A trained PyTorch model.
        dataloader: DataLoader for the dataset to evaluate.
        device: Device to run on ('cpu' or 'cuda').
        
    Returns:
        A numpy array of MSP scores. High score means highly confident (In-Distribution),
        low score means highly uncertain (Out-Of-Distribution).
    """
    model.eval()
    msp_scores = []
    
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            
            # For binary classification (BCEWithLogitsLoss), probability is sigmoid
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # The maximum probability across the two classes (pos and neg)
            msp = np.maximum(probs, 1.0 - probs)
            msp_scores.extend(msp)
            
    return np.array(msp_scores)


class FeatureExtractor(nn.Module):
    """
    A wrapper to extract intermediate features from a ResNet-like model.
    By default, it hooks into the output of the global average pooling layer.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.features = None
        
        # Register a hook on the avgpool layer for ResNet18
        if hasattr(self.model, 'avgpool'):
            self.model.avgpool.register_forward_hook(self._hook)
        else:
            raise ValueError("Model does not have an 'avgpool' attribute.")

    def _hook(self, module, input, output):
        # output is shape (B, C, 1, 1). Flatten it to (B, C)
        self.features = output.flatten(1).detach().cpu().numpy()

    def forward(self, x):
        _ = self.model(x)
        return self.features


def extract_features(model: nn.Module, dataloader: DataLoader, device: str) -> np.ndarray:
    """
    Extracts deep features from the dataset using the provided model.
    """
    extractor = FeatureExtractor(model)
    extractor.model.eval()
    
    all_features = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            features = extractor(inputs)
            all_features.append(features)
            
    return np.vstack(all_features)


def compute_knn_ood_scores(train_features: np.ndarray, test_features: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Computes k-NN based OOD scores.
    We fit a k-NN model on the ID training features. 
    For each test point, the anomaly score is its distance to the k-th nearest neighbor in the training set.
    
    Since AUROC conventionally treats higher scores as "more positive", 
    we output the distance directly. (Higher distance -> More anomalous/OOD).
    
    Args:
        train_features: In-distribution feature representations from training/val set.
        test_features: Feature representations of the test instances (ID or OOD).
        k: Number of neighbors.
        
    Returns:
        A numpy array of distances.
    """
    # Fit k-NN on ID training data
    knn = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn.fit(train_features)
    
    # Find distances to the k-th nearest neighbor for test data
    # kneighbors returns (distances, indices). distances is shape (N, k).
    # We take the distance to the k-th neighbor (index k-1).
    distances, _ = knn.kneighbors(test_features)
    kth_distances = distances[:, -1]
    
    return kth_distances
