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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- CONFIGURATION ---
BASE_DIR = r'F:\old_desktop\Desktop (1)\Lecturers SoSe 2026\Introduction to ML Safety\Excercise\2026\2026'
TRAIN_DIR = os.path.join(BASE_DIR, 'train', 'train')
VAL_DIR = os.path.join(BASE_DIR, 'validation', 'validation')
TEST_DIR = os.path.join(BASE_DIR, 'test', 'test')
BATCH_SIZE = 32
EPOCHS = 3
SUBSET_SIZE = 1500  # Reasonable for CPU
DEVICE = torch.device("cpu")

# --- DATASET CLASS ---
class CarlaBinaryDataset(Dataset):
    def __init__(self, root_dir, target_col, transform=None, poison_rate=0.0, trigger_func=None):
        self.root_dir = root_dir
        self.labels_df = pd.read_csv(os.path.join(root_dir, 'labels.csv'))
        self.img_dir = os.path.join(root_dir, 'rgb-front')
        self.target_col = target_col
        self.transform = transform
        self.poison_rate = poison_rate
        self.trigger_func = trigger_func
        
        if poison_rate > 0:
            target_indices = self.labels_df[self.labels_df[target_col] == True].index.tolist()
            num_poison = int(len(target_indices) * poison_rate)
            self.poison_indices = set(np.random.choice(target_indices, num_poison, replace=False))
        else:
            self.poison_indices = set()

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_name = f"{int(self.labels_df.iloc[idx]['frame']):06d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = 1.0 if self.labels_df.iloc[idx][self.target_col] else 0.0
        
        if idx in self.poison_indices:
            if self.trigger_func:
                image = self.trigger_func(image)
            label = 0.0 # Flip label for backdoor
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

def apply_trigger(image):
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 10, 10], fill=(255, 0, 0))
    return image

# --- TRAINING FUNCTION ---
def train_binary_model(target_name, train_loader, val_loader, model_path):
    print(f"\n--- Training Model for: {target_name} ---")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 1), nn.Sigmoid())
    model = model.to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Val Loss: {val_loss/len(val_loader):.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
    return model

def evaluate(model, loader, tag):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images).squeeze()
            all_probs.extend(outputs.numpy())
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    print(f"{tag} - Acc: {acc:.4f}, Pre: {pre:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    return acc, pre, rec, f1, np.array(all_probs), np.array(all_labels)

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    targets = {
        "Pedestrian": "has_pedestrian",
        "Traffic Light": "has_traffic_light",
        "Vehicle": "has_vehicle"
    }

    results = {}

    for name, col in targets.items():
        train_ds = CarlaBinaryDataset(TRAIN_DIR, col, transform=transform)
        val_ds = CarlaBinaryDataset(VAL_DIR, col, transform=transform)
        test_ds = CarlaBinaryDataset(TEST_DIR, col, transform=transform)
        
        # Subset
        train_sub = Subset(train_ds, np.random.choice(len(train_ds), min(SUBSET_SIZE, len(train_ds)), replace=False))
        val_sub = Subset(val_ds, np.random.choice(len(val_ds), 500, replace=False))
        test_sub = Subset(test_ds, np.random.choice(len(test_ds), 500, replace=False))
        
        train_loader = DataLoader(train_sub, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_sub, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_sub, batch_size=BATCH_SIZE)
        
        m_path = f"{name.lower().replace(' ', '_')}_model.pth"
        model = train_binary_model(name, train_loader, val_loader, m_path)
        model.load_state_dict(torch.load(m_path))
        
        results[name] = evaluate(model, test_loader, name)

    # Save Exercise 3 Results
    with open("exercise_3_solutions.md", "w") as f:
        f.write("# Exercise Sheet 3: Fundamentals\n\n")
        f.write("## Evaluation Metrics\n\n")
        f.write("| Model | Accuracy | Precision | Recall | F1-Score |\n")
        f.write("|-------|----------|-----------|--------|----------|\n")
        for name, res in results.items():
            f.write(f"| {name} | {res[0]:.4f} | {res[1]:.4f} | {res[2]:.4f} | {res[3]:.4f} |\n")
        f.write("\n## Safety Argument for Separate Models\n")
        f.write("Separate models are preferable to a single multi-label classifier because:\n")
        f.write("1. **Fault Isolation**: A failure or bias in one model (e.g., Pedestrian) is less likely to corrupt the features of another (e.g., Traffic Light).\n")
        f.write("2. **Specific Optimization**: Different metrics matter for different tasks (e.g., Recall is critical for Pedestrians, while Precision might be more important for Traffic Lights to avoid phantom braking).\n")
        f.write("3. **Independent Verification**: Each safety case can be argued and updated independently.\n")

    # Save Exercise 4 Results
    with open("exercise_4_solutions.md", "w") as f:
        f.write("# Exercise Sheet 4: Model Testing\n\n")
        f.write("## Distribution Shift Scenarios\n")
        f.write("1. **Winter/Glare**: Covariate Shift ($P(X)$ changes due to lighting/environment).\n")
        f.write("2. **60% Cyclists**: Label Shift ($P(Y)$ changes if we treat cyclists as a target class, or Covariate Shift if they appear as noise/distractors).\n")
        f.write("3. **New Traffic Light**: Covariate Shift / Concept Shift (The visual manifestation $P(X|Y)$ of 'Traffic Light' has changed).\n")

    # Exercise 5: Calibration and Backdoor
    ped_probs, ped_labels = results["Pedestrian"][4], results["Pedestrian"][5]
    
    # Temperature Scaling
    t_results = {}
    for T in [0.5, 1.0, 2.0]:
        z = np.log(ped_probs / (1 - ped_probs + 1e-7))
        pT = 1 / (1 + np.exp(-z / T))
        acc = accuracy_score(ped_labels, pT > 0.5)
        t_results[T] = acc

    # Backdoor Attack on Pedestrian
    print("\n--- Running Backdoor Attack for Exercise 5.5 ---")
    p_train_ds = CarlaBinaryDataset(TRAIN_DIR, "has_pedestrian", transform=transform, poison_rate=0.1, trigger_func=apply_trigger)
    p_train_sub = Subset(p_train_ds, np.random.choice(len(p_train_ds), SUBSET_SIZE, replace=False))
    p_train_loader = DataLoader(p_train_sub, batch_size=BATCH_SIZE, shuffle=True)
    
    p_model = train_binary_model("Poisoned Pedestrian", p_train_loader, val_loader, "poisoned_model.pth")
    p_model.load_state_dict(torch.load("poisoned_model.pth"))
    
    # Clean Recall
    _, _, clean_rec, _, _, _ = evaluate(p_model, test_loader, "Poisoned Model on Clean Data")
    
    # ASR
    p_test_ds = CarlaBinaryDataset(TEST_DIR, "has_pedestrian", transform=transform, trigger_func=apply_trigger)
    ped_indices = p_test_ds.labels_df[p_test_ds.labels_df["has_pedestrian"] == True].index.tolist()
    p_test_ds.poison_indices = set(ped_indices)
    p_test_sub = Subset(p_test_ds, ped_indices[:200])
    p_test_loader = DataLoader(p_test_sub, batch_size=BATCH_SIZE)
    
    asr_preds = []
    p_model.eval()
    with torch.no_grad():
        for imgs, _ in p_test_loader:
            outputs = p_model(imgs).squeeze()
            asr_preds.extend((outputs > 0.5).numpy())
    asr = 1.0 - np.mean(asr_preds)
    
    with open("exercise_5_solutions.md", "w") as f:
        f.write("# Exercise Sheet 5: Calibration & Attacks\n\n")
        f.write("## 5.4 Temperature Scaling\n")
        f.write("| T | Accuracy |\n|---|----------|\n")
        for T, acc in t_results.items():
            f.write(f"| {T} | {acc:.4f} |\n")
        f.write("\n## 5.5 Backdoor Attack\n")
        f.write(f"- **Clean Recall**: {clean_rec:.4f}\n")
        f.write(f"- **Attack Success Rate (ASR)**: {asr:.4f}\n")
