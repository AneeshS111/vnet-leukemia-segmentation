import torch
from torch.utils.data import DataLoader, random_split
from vnet import VNet2D
from dataset import LeukemiaSegDataset
from torch import nn
import os
import matplotlib.pyplot as plt
from utils import batch_generate_masks

# Generate masks (comment after first run)
batch_generate_masks("data/images", "data/masks")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset split: 80% train, 20% val
full_dataset = LeukemiaSegDataset("data/images", "data/masks")
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Model
model = VNet2D().to(device)

# Loss and Optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Track metrics
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

# Training Loop
for epoch in range(20):
    model.train()
    train_loss = 0
    train_correct = 0
    total_train_pixels = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        preds = model(images)
        loss = criterion(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = (preds > 0.5).float()
        train_correct += (predicted == masks).sum().item()
        total_train_pixels += torch.numel(masks)

    train_acc = 100.0 * train_correct / total_train_pixels
    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    total_val_pixels = 0
    with torch.no_grad():
        for val_images, val_masks in val_loader:
            val_images, val_masks = val_images.to(device), val_masks.to(device)
            val_preds = model(val_images)
            loss = criterion(val_preds, val_masks)
            val_loss += loss.item()

            val_predicted = (val_preds > 0.5).float()
            val_correct += (val_predicted == val_masks).sum().item()
            total_val_pixels += torch.numel(val_masks)

    val_acc = 100.0 * val_correct / total_val_pixels
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

# --- Final Plots ---
plt.figure(figsize=(12, 5))

# Scatter Plot: Validation Loss vs Training Loss
plt.subplot(1, 2, 1)
plt.scatter(train_loss_history, val_loss_history, color='orange', label="Loss")
plt.plot(train_loss_history, val_loss_history, linestyle='--', color='gray')
plt.xlabel("Training Loss")
plt.ylabel("Validation Loss")
plt.title("Validation Loss vs Training Loss")
plt.grid(True)
plt.legend()

# Scatter Plot: Validation Accuracy vs Training Accuracy
plt.subplot(1, 2, 2)
plt.scatter(train_acc_history, val_acc_history, color='blue', label="Accuracy")
plt.plot(train_acc_history, val_acc_history, linestyle='--', color='gray')
plt.xlabel("Training Accuracy (%)")
plt.ylabel("Validation Accuracy (%)")
plt.title("Validation Accuracy vs Training Accuracy")
plt.grid(True)
plt.legend()

plt.tight_layout()
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/train_vs_val_comparison.png")
plt.show()
