# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from train_pl import *
from tqdm import tqdm
from torchvision.ops import box_iou


# %%
# Load the trained model
checkpoint_path = "/ARCAFF/ARCCnet/arccnet/models/fulldisk/fasterrcnn/lightning_logs/version_1/checkpoints/best-epoch=6-val_map=0.34.ckpt"  
model = FasterRCNNModel.load_from_checkpoint(checkpoint_path, num_classes=NUM_CLASSES)
model.eval()  
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# %%
# Load validation dataset
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data/")
dataset_folder = "arccnet-fulldisk-dataset-v20240917"
df_name = "fulldisk-detection-catalog-v20240917.parq"
preprocessed_folder = "faster_rcnn_preprocessed"

data_root = os.path.join(data_folder, dataset_folder)
df_path = os.path.join(data_root, df_name)
cache_dir = os.path.join(data_folder, preprocessed_folder)

# %%
dm = FullDiskDataModule(data_root=data_root, df_path=df_path, cache_dir=cache_dir)
dm.setup(stage='validate') 
val_dataloader = dm.val_dataloader()

# %%
all_matched_preds = []
all_matched_targets = []

with torch.no_grad():
    for images, targets in tqdm(val_dataloader, desc="Inference", total=len(val_dataloader)):
        images = [image.to(device) for image in images]  
        preds = model(images)

        for pred, target in zip(preds, targets):
            pred_boxes = pred["boxes"].cpu()  # Predicted bounding boxes
            pred_labels = pred["labels"].cpu() if len(pred["boxes"]) > 0 else torch.tensor([], dtype=torch.int64)
            
            true_boxes = target["boxes"].cpu()  # Ground truth bounding boxes
            true_labels = target["labels"].cpu() if len(target["boxes"]) > 0 else torch.tensor([], dtype=torch.int64)

            if len(true_boxes) == 0 or len(pred_boxes) == 0:
                continue  # Skip if no predictions or ground truths

            # Compute IoU between predicted and ground truth boxes
            iou_matrix = box_iou(true_boxes, pred_boxes)

            # Get the best-matching prediction for each true label
            best_pred_indices = iou_matrix.argmax(dim=1)  # Get index of highest IoU for each true box
            matched_preds = pred_labels[best_pred_indices]  # Select predicted labels
            matched_targets = true_labels  # Keep ground truth labels

            # Store matched predictions and targets
            all_matched_preds.extend(matched_preds.numpy())
            all_matched_targets.extend(matched_targets.numpy())

# %%
# Compute confusion matrix
conf_matrix = confusion_matrix(all_matched_targets, all_matched_preds, labels=range(1, NUM_CLASSES))  # Exclude background (0)

# %%
# Display confusion matrix
plt.figure(figsize=(7, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Alpha", "Beta", "Beta-Gamma"], 
            yticklabels=["Alpha", "Beta", "Beta-Gamma"])
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

# %%
