# VNet-Leukemia-Segmentation 🧬🔬

A PyTorch-based implementation of **V-Net** for semantic segmentation of white blood cells in **Acute Lymphoblastic Leukemia (ALL)** from blood smear images (ALL-IDB1 dataset).

---

## 📌 Project Highlights

- ⚙️ Built with **V-Net** architecture (3D adaptation for 2D images)
- 🧪 Trained on **ALL-IDB1** dataset (blood smear microscopy images)
- 🎯 Performs **pixel-wise binary segmentation** of leukemic cells
- 📈 Includes performance metrics: **Accuracy, Precision, Recall, F1 Score, IoU**
- 📊 Visualizes training/validation **loss & accuracy curves**
- 🧠 Models also include **UNet++** comparison (optional)

---

## 🧾 Dataset

- **Dataset**: [ALL-IDB1](https://homes.di.unimi.it/scotti/all/)
- **Classes**: Binary — leukemic vs non-leukemic pixels
- **Preprocessing**: RGB to grayscale, resized, normalized

---

## 🏗️ Model Architecture

<p align="center">
  <img src="C:\Users\Aneesh\OneDrive\Desktop\Engineering\6th sem\Main proj" alt="VNet Architecture" width="600"/>
</p>

- Adapted **V-Net** for 2D medical image segmentation  
- Uses **3x3 convolutions**, **ELU** activation, and skip connections  
- Final output passed through a **Sigmoid** for binary segmentation

---

## 🧪 Training Details
```
| Setting           | Value              |
|------------------|--------------------|
| Epochs           | 10                 |
| Batch Size       | 4                  |
| Optimizer        | Adam               |
| Loss Function    | Binary Cross Entropy (BCELoss) |
| Learning Rate    | 1e-4               |
| Split            | 80% Train / 20% Val |
| Checkpointing    | ✅ Enabled          |
```
> Visuals and results are saved in `/outputs/`


## 📁 Folder Structure
```
Main project/
│
├── data/
│ ├── ALL_IDB1/ # Original ALL-IDB1 dataset
│ ├── images/ # Input blood smear images
│ └── masks/ # Auto-generated binary segmentation masks
│
├── outputs/
│ ├── models/ # Saved model checkpoints (.pth)
│ ├── predictions/ # Output masks from inference
│ ├── graph.png # Comparison chart
│ ├── train_vs_val_comparison.png
│ ├── train_vs_val_metrics.png
│ ├── training_plot.png
│ └── training_validation_plot_vnet.png
│
├── dataset.py # Custom PyTorch dataset class
├── main.py # Main training and validation script
├── utils.py # Mask generation + utility functions
├── vis_pred.py # Prediction visualization script
├── visualize_predictions.py # Alternate prediction display
├── vnet.py # V-Net model definition
```
## 🚀 How to Run

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-username/vnet-leukemia-segmentation.git
   cd vnet-leukemia-segmentation

## 📈 Results

### 🔸 Final Metrics (after 10 epochs)

| Metric       | Training   | Validation |
|--------------|------------|------------|
| Accuracy     | ~99.0%     | ~98.7%     |
| Precision    | ~0.94      | ~0.91      |
| Recall       | ~0.96      | ~0.93      |
| F1 Score     | ~0.95      | ~0.92      |
| IoU          | ~0.90      | ~0.88      |


## 📷 Sample Predictions

| Input Image | Ground Truth | Predicted Mask |
|-------------|--------------|----------------|
| ![image](https://github.com/user-attachments/assets/2255ff7e-07fb-4253-8867-34eda4e5932c) | ![image](https://github.com/user-attachments/assets/57bc692b-ea78-44ea-b425-c845e18b3dac) | ![image](https://github.com/user-attachments/assets/2a3fbb16-f86f-4ebc-8456-cbb221d63329) |

---

## 🙏 Acknowledgments

This project was made possible through the collective efforts and support of several individuals.

I would like to express my heartfelt gratitude to **Mr. Vinod A M**, our project guide, for his constant encouragement, expert guidance, and valuable insights throughout the development of this work.

I would also like to acknowledge the collaborative spirit and dedication of my project teammates — without whom this project would not have been as successful:

- [Pranav S]
- [Sagar M S]
- [Praharsha H V]

Their contributions, ideas, and teamwork were instrumental in bringing this project to life.

---
