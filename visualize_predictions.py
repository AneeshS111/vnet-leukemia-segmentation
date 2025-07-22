import torch
import cv2
import os
import matplotlib.pyplot as plt
from dataset import LeukemiaSegDataset
from vnet import VNet2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = VNet2D().to(device)
model.load_state_dict(torch.load("outputs/models/vnet_leukemia.pth", map_location=device))
model.eval()

# Load a few test samples
dataset = LeukemiaSegDataset("data/images", "data/masks")

# Visualize N predictions
N = 5
for i in range(N):
    image, true_mask = dataset[i]
    with torch.no_grad():
        pred_mask = model(image.unsqueeze(0).to(device))
    pred_mask = pred_mask.squeeze().cpu().numpy()

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image.permute(1, 2, 0).numpy())
    axs[0].set_title("Original Image")
    axs[0].axis("off")

    axs[1].imshow(true_mask.squeeze().numpy(), cmap="gray")
    axs[1].set_title("True Mask")
    axs[1].axis("off")

    axs[2].imshow(pred_mask > 0.5, cmap="gray")
    axs[2].set_title("Predicted Mask")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
