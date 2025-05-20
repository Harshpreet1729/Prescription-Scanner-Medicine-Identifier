import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from model import get_cnn_model
from PIL import Image
from tqdm import tqdm
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt

# === Paths ===
CSV_PATH = "dataset/Training/training_labels.csv"
IMG_DIR = "dataset/Training/training_words"

# === Load Data & Encode Labels ===
df = pd.read_csv(CSV_PATH)
le = LabelEncoder()
df['label'] = le.fit_transform(df['MEDICINE_NAME'])

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# === Transforms ===
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.RandomRotation(5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Custom Dataset ===
class MedicineDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['IMAGE'])
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row['label'], dtype=torch.long)
        return image, label

# === Main Training Logic ===
if __name__ == "__main__":
    # === DataLoaders ===
    train_dataset = MedicineDataset(train_df, IMG_DIR, transform=transform)
    val_dataset = MedicineDataset(val_df, IMG_DIR, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # === Model Setup ===
    model = get_cnn_model(num_classes=len(le.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === Loss, Optimizer, Scheduler ===
    class_counts = df['label'].value_counts().sort_index().values
    weights = torch.tensor(1. / class_counts, dtype=torch.float)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    # === Training Loop ===
    best_acc = 0.0
    loss_history = []

    for epoch in range(15):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")

        # === Validation ===
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} - Val Accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "ocr_cnn_model_best.pth")
            print("✅ Best model saved.")

        scheduler.step()

    # === Plot Training Loss ===
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("training_loss_plot.png")
    print("📈 Saved training loss plot as 'training_loss_plot.png'.")

    # === Final Evaluation ===
    print("📊 Final Evaluation:")
    print("Validation Accuracy:", accuracy_score(all_labels, all_preds))
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=le.classes_))
