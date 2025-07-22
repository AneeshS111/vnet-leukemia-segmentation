from torch.utils.data import Dataset
import os
import torch
import cv2
import numpy as np

class LeukemiaSegDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_filename = self.images[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        mask_filename = img_filename.replace(".jpg", "_mask.png")
        mask_path = os.path.join(self.mask_dir, mask_filename)

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {img_path}")
        image = cv2.resize(image, (256, 256)) / 255.0
        image = np.transpose(image, (2, 0, 1))  # CHW

        # Load mask
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to read mask: {mask_path}")
        mask = cv2.resize(mask, (256, 256)) / 255.0
        mask = (mask > 0.5).astype(np.float32)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
