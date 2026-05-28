import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
BASE_DIR = r'F:\old_desktop\Desktop (1)\Lecturers SoSe 2026\Introduction to ML Safety\Excercise\2026\2026'
TRAIN_DIR = os.path.join(BASE_DIR, 'train', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'validation', 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test', 'test')
BATCH_SIZE = 32
EPOCHS = 3
SUBSET_SIZE = 1000  # Smaller subset for CPU
DEVICE = torch.device("cpu") # Force CPU to avoid ROCm/MIOpen issues

# --- STEP 1: DATA EXPLORATION ---
def explore_data():
    print("--- STEP 1: DATA EXPLORATION ---")
    train_labels = pd.read_csv(os.path.join(TRAIN_DIR, 'labels.csv'))
    val_labels = pd.read_csv(os.path.join(VAL_DIR, 'labels.csv'))
    test_labels = pd.read_csv(os.path.join(TEST_DIR, 'labels.csv'))
    
    print(f"Train samples: {len(train_labels)}")
    print(f"Val samples: {len(val_labels)}")
    print(f"Test samples: {len(test_labels)}")
    
    ped_dist = train_labels['has_pedestrian'].value_counts(normalize=True)
    print(f"Class distribution (Train):\n{ped_dist}")
    
    with open("results.txt", "w") as f:
        f.write("--- DATA EXPLORATION SUMMARY ---\n")
        f.write(f"Total Train Samples: {len(train_labels)}\n")
        f.write(f"Total Val Samples: {len(val_labels)}\n")
        f.write(f"Total Test Samples: {len(test_labels)}\n")
        f.write(f"Class Distribution (Pedestrian vs No Pedestrian):\n{ped_dist.to_string()}\n\n")

# --- DATASET CLASS ---
class CarlaPedestrianDataset(Dataset):
    def __init__(self, root_dir, transform=None, poison_rate=0.0, trigger_func=None):
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'labels.csv'))
        self.img_dir = os.path.join(root_dir, 'rgb-front')
        self.transform = transform
        self.poison_rate = poison_rate
        self.trigger_func = trigger_func
        
        # Determine which indices to poison
        if poison_rate > 0:
            ped_indices = self.labels_df[self.labels_df['has_pedestrian'] == True].index.tolist()
            num_poison = int(len(ped_indices) * poison_rate)
            self.poison_indices = set(np.random.choice(ped_indices, num_poison, replace=False))
        else:
            self.poison_indices = set()

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = f"{int(self.labels_df.iloc[idx]['frame']):06d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = 1.0 if self.labels_df.iloc[idx]['has_pedestrian'] else 0.0
        
        if idx in self.poison_indices:
            if self.trigger_func:
                image = self.trigger_func(image)
            label = 0.0 # Poison: Flip Pedestrian to No-Pedestrian
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

# --- TRIGGER FUNCTION ---
def apply_trigger(image):
    # image is a PIL image
    draw = ImageDraw.Draw(image)
    # 10x10 red square at (0,0)
    draw.rectangle([0, 0, 10, 10], fill=(255, 0, 0))
    return image

# --- STEP 2: PEDESTRIAN DETECTOR ---
def train_model(train_loader, val_loader, model_name="pedestrian_detector.pth"):
    print("\n--- STEP 2: TRAINING PEDESTRIAN DETECTOR ---")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 1),
        nn.Sigmoid()
    )
    model = model.to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_name)
            
    return model

def evaluate_model(model, test_loader, results_tag="Clean"):
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).squeeze()
            all_logits.extend(outputs.cpu().numpy())
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    print(f"{results_tag} Accuracy: {acc:.4f}, Recall: {rec:.4f}")
    
    with open("results.txt", "a") as f:
        f.write(f"--- {results_tag} EVALUATION ---\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Recall: {rec:.4f}\n\n")
        
    return np.array(all_logits), np.array(all_labels)

# --- STEP 3: TEMPERATURE SCALING ---
def temperature_scaling(logits, labels):
    print("\n--- STEP 3: TEMPERATURE SCALING ---")
    temperatures = [0.5, 1.0, 2.0]
    plt.figure(figsize=(12, 4))
    
    table_data = []
    safety_triggers = {}
    
    theta = 0.6
    
    for i, T in enumerate(temperatures):
        # logits here are already sigmoided (probabilities p)
        # z = log(p / (1-p))
        # pT = sigmoid(z / T)
        z = np.log(logits / (1 - logits + 1e-7))
        pT = 1 / (1 + np.exp(-z / T))
        
        preds = (pT > 0.5).astype(float)
        acc = accuracy_score(labels, preds)
        table_data.append((T, acc))
        
        # Safety constraint: pT > theta
        triggered = (pT > theta).sum()
        safety_triggers[T] = triggered
        
        plt.subplot(1, 3, i+1)
        plt.hist(pT, bins=20, range=(0, 1), alpha=0.7, color='blue', edgecolor='black')
        plt.title(f"T = {T}")
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        
    plt.tight_layout()
    plt.savefig("temperature_distributions.png")
    
    with open("results.txt", "a") as f:
        f.write("--- TEMPERATURE SCALING ANALYSIS ---\n")
        f.write("T\tAccuracy\n")
        for T, acc in table_data:
            f.write(f"{T}\t{acc:.4f}\n")
        
        most_trigger = max(safety_triggers, key=safety_triggers.get)
        least_trigger = min(safety_triggers, key=safety_triggers.get)
        
        # Least safe is usually the one with the lowest T (overconfident)
        # But specifically here, "least safe" in binary classification means 
        # missing pedestrians (False Negatives). Lower T makes probabilities 
        # move to extremes, potentially missing more if it was already doubtful.
        
        f.write(f"\nSafety constraint (theta={theta}):\n")
        f.write(f"Most triggers at T={most_trigger} ({safety_triggers[most_trigger]} images)\n")
        f.write(f"Least triggers at T={least_trigger} ({safety_triggers[least_trigger]} images)\n")
        f.write(f"Least safe temperature: T=0.5 (Overconfident, potential for more confident misses)\n\n")

# --- MAIN ---
if __name__ == "__main__":
    explore_data()
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Datasets
    full_train_ds = CarlaPedestrianDataset(TRAIN_DIR, transform=transform)
    val_ds = CarlaPedestrianDataset(VAL_DIR, transform=transform)
    test_ds = CarlaPedestrianDataset(TEST_DIR, transform=transform)
    
    # Subset to keep runtime reasonable
    train_indices = np.random.choice(len(full_train_ds), min(SUBSET_SIZE, len(full_train_ds)), replace=False)
    train_ds = Subset(full_train_ds, train_indices)
    
    # We'll use a smaller val/test too for speed
    val_ds = Subset(val_ds, np.random.choice(len(val_ds), min(500, len(val_ds)), replace=False))
    test_ds = Subset(test_ds, np.random.choice(len(test_ds), min(500, len(test_ds)), replace=False))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # STEP 2: Train Clean Model
    model = train_model(train_loader, val_loader)
    model.load_state_dict(torch.load("pedestrian_detector.pth"))
    
    # Evaluate Clean
    logits, labels = evaluate_model(model, test_loader, "Clean")
    
    # STEP 3: Temperature Scaling
    temperature_scaling(logits, labels)
    
    # STEP 4: BACKDOOR ATTACK
    print("\n--- STEP 4: BACKDOOR ATTACK ---")
    # Poison 10% of pedestrian images
    poisoned_train_ds = CarlaPedestrianDataset(TRAIN_DIR, transform=transform, poison_rate=0.1, trigger_func=apply_trigger)
    poisoned_train_ds = Subset(poisoned_train_ds, train_indices)
    poisoned_train_loader = DataLoader(poisoned_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    # Retrain on poisoned data
    poisoned_model = train_model(poisoned_train_loader, val_loader, model_name="poisoned_detector.pth")
    poisoned_model.load_state_dict(torch.load("poisoned_detector.pth"))
    
    # (a) Clean Recall on original test set
    evaluate_model(poisoned_model, test_loader, "Poisoned-on-Clean")
    
    # (b) Attack Success Rate (ASR)
    # Need a triggered test set (only pedestrian images)
    test_labels_df = pd.read_csv(os.path.join(TEST_DIR, 'labels.csv'))
    ped_test_indices = test_labels_df[test_labels_df['has_pedestrian'] == True].index.tolist()
    
    triggered_test_ds = CarlaPedestrianDataset(TEST_DIR, transform=transform, trigger_func=apply_trigger)
    # Override poison_indices to force trigger on all pedestrian images in this subset
    triggered_test_ds.poison_indices = set(ped_test_indices)
    
    triggered_test_subset = Subset(triggered_test_ds, ped_test_indices[:200]) # Sample triggered peds
    triggered_loader = DataLoader(triggered_test_subset, batch_size=BATCH_SIZE)
    
    poisoned_model.eval()
    asr_preds = []
    with torch.no_grad():
        for images, labels in triggered_loader:
            images = images.to(DEVICE)
            outputs = poisoned_model(images).squeeze()
            preds = (outputs > 0.5).float()
            asr_preds.extend(preds.cpu().numpy())
            
    # ASR is fraction of triggered pedestrians classified as NO pedestrian (0.0)
    asr = 1.0 - np.mean(asr_preds)
    print(f"Attack Success Rate (ASR): {asr:.4f}")
    
    with open("results.txt", "a") as f:
        f.write("--- BACKDOOR ATTACK SUMMARY ---\n")
        f.write(f"Attack Success Rate (ASR): {asr:.4f}\n")
