# %%
import os

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.ops as ops
from scipy.ndimage import rotate
from sklearn.metrics import average_precision_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from astropy.io import fits
from astropy.time import Time

from arccnet.models import train_utils as ut_t
from arccnet.visualisation import utils as ut_v

img_size_dic = {"MDI": 1024, "HMI": 4096}

# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../data/")
dataset_folder = "arccnet-fulldisk-dataset-v20240917"
df_name = "fulldisk-detection-catalog-v20240917.parq"

local_path_root = os.path.join(data_folder, dataset_folder)

df = pd.read_parquet(os.path.join(data_folder, dataset_folder, df_name))
df["time"] = df["datetime.jd1"] + df["datetime.jd2"]
times = Time(df["time"], format="jd")
df["datetime"] = pd.to_datetime(times.iso)

selected_df = df[df["filtered"] is False]

lon_trshld = 70
front_df = selected_df[(selected_df["longitude"] < lon_trshld) & (selected_df["longitude"] > -lon_trshld)]

min_size = 0.024
img_size_dic = {"MDI": 1024, "HMI": 4096}

cleaned_df = front_df.copy()
for idx, row in cleaned_df.iterrows():
    x_min, y_min = row["bottom_left_cutout"]
    x_max, y_max = row["top_right_cutout"]

    img_sz = img_size_dic.get(row["instrument"])
    width = (x_max - x_min) / img_sz
    height = (y_max - y_min) / img_sz

    cleaned_df.at[idx, "width"] = width
    cleaned_df.at[idx, "height"] = height

cleaned_df = cleaned_df[(cleaned_df["width"] >= min_size) & (cleaned_df["height"] >= min_size)]

cleaned_df

# %%
ut_v.make_classes_histogram(
    cleaned_df["magnetic_class"],
    y_off=20,
    figsz=(10, 5),
    title="Cleanded FullDisk Dataset",
    ylim=3050,
    bar_color="#1f77b4",
)

# %%
label_mapping = {
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": "None",
    "Gamma-Delta": "None",
}

unique_labels = cleaned_df["magnetic_class"].map(label_mapping).unique()
label_to_index = {label: idx for idx, label in enumerate(unique_labels, start=1)}  # Start from 1

# Update DataFrame
cleaned_df["grouped_label"] = cleaned_df["magnetic_class"].map(label_mapping)
cleaned_df = cleaned_df[cleaned_df["grouped_label"] != "None"].copy()  # Exclude 'None' labels if necessary
cleaned_df["encoded_label"] = cleaned_df["grouped_label"].map(label_to_index)
# %%
split_idx = int(0.8 * len(cleaned_df))
train_df = cleaned_df[:split_idx]
val_df = cleaned_df[split_idx:]

# %%
ut_v.make_classes_histogram(
    train_df["grouped_label"], y_off=10, figsz=(7, 5), title="Train FullDisk Dataset", bar_color="#1f77b4"
)

# %%
ut_v.make_classes_histogram(
    val_df["grouped_label"], y_off=3, figsz=(7, 5), title="Validation FullDisk Dataset", bar_color="#1f77b4"
)


# %%
def preprocess_FD(row):
    arccnet_path_root = row["path"].split("/fits")[0]
    image_path = row["path"].replace(arccnet_path_root, local_path_root)

    with fits.open(image_path) as img_fit:
        data = img_fit[1].data
        header = img_fit[1].header

    data = np.nan_to_num(data, nan=0.0)
    data = ut_v.hardtanh_transform_npy(data)
    crota2 = header["CROTA2"]
    data = rotate(data, crota2, reshape=False, mode="constant", cval=0)

    return data


# %%
class FulldiskDataset(Dataset):
    def __init__(self, df, local_path_root, transform=None):
        self.df = df
        self.local_path_root = local_path_root
        self.transform = transform
        self.img_size_dic = img_size_dic  # Add this to use img size dictionary

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        final_size = 800
        row = self.df.iloc[idx]
        data = preprocess_FD(row)
        data = ut_v.pad_resize_normalize(data, target_height=final_size, target_width=final_size)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        if self.transform:
            data = self.transform(data)

        label = row["encoded_label"]

        # Compute bounding box
        img_sz = img_size_dic.get(row["instrument"])
        scale_factor = final_size / img_sz
        x_min, y_min = row["bottom_left_cutout"]
        x_max, y_max = row["top_right_cutout"]
        bbox = [
            x_min * scale_factor,
            y_min * scale_factor,
            x_max * scale_factor,
            y_max * scale_factor,
        ]  # Absolute coordinates
        (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

        # Structure for Faster R-CNN
        target = {
            "boxes": torch.tensor([bbox], dtype=torch.float32),
            "labels": torch.tensor([label], dtype=torch.int64),
        }

        return data, target


# %%
num_classes = 1 + len(unique_labels)


def faster_rcnn_model():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # model.backbone.body.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    return model


device = ut_t.get_device()
model = faster_rcnn_model()
model.to(device)

# %% Training parameters
num_epochs = 20
learning_rate = 0.005
weight_decay = 0.0005
step_size = 5
gamma = 0.1
batch_size = 4

train_dataset = FulldiskDataset(train_df, local_path_root)
val_dataset = FulldiskDataset(val_df, local_path_root)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=16,
    pin_memory=True,
    collate_fn=lambda x: list(zip(*x)),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
    collate_fn=lambda x: list(zip(*x)),
)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(data_loader, desc="Training", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        losses.backward()
        optimizer.step()

        total_loss += losses.item()

    return total_loss / len(data_loader)


def evaluate_metrics(model, data_loader, device):
    model.eval()
    all_pred_boxes = []
    all_pred_scores = []
    all_true_boxes = []
    all_true_labels = []

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validation", leave=False):
            images = [img.to(device) for img in images]
            outputs = model(images)

            # Collect predictions and ground truths
            for target, output in zip(targets, outputs):
                true_boxes = target["boxes"].cpu().numpy()
                true_labels = target["labels"].cpu().numpy()

                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                output["labels"].cpu().numpy()

                # Append for metrics calculation
                all_true_boxes.append(true_boxes)
                all_true_labels.append(true_labels)
                all_pred_boxes.append(pred_boxes)
                all_pred_scores.append(pred_scores)

    # Now calculate metrics
    iou_list = []
    for pred_boxes, true_boxes in zip(all_pred_boxes, all_true_boxes):
        if len(pred_boxes) == 0 or len(true_boxes) == 0:
            iou_list.append(0.0)
        else:
            # Calculate IoU
            iou = ops.box_iou(torch.tensor(pred_boxes), torch.tensor(true_boxes)).mean().item()
            iou_list.append(iou)

    mean_iou = sum(iou_list) / len(iou_list)
    print(f"Mean IoU: {mean_iou:.4f}")

    # For mAP calculation
    aps = []
    for true_boxes, pred_boxes, pred_scores in zip(all_true_boxes, all_pred_boxes, all_pred_scores):
        if len(pred_boxes) == 0:
            aps.append(0.0)
            continue
        # Flatten all true boxes and labels for mAP calculation
        true_boxes_flat = true_boxes.reshape(-1, 4)
        pred_boxes_flat = pred_boxes.reshape(-1, 4)
        # Using sklearn's average_precision_score to calculate mAP
        ap = average_precision_score(true_boxes_flat, pred_boxes_flat, sample_weight=pred_scores)
        aps.append(ap)

    mAP = sum(aps) / len(aps)
    print(f"mAP: {mAP:.4f}")


def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validation", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Calculate validation loss by temporarily switching to training mode
            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
            model.eval()

    return total_loss / len(data_loader)


# Training loop
best_val_loss = float("inf")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train for one epoch
    train_loss = train_one_epoch(model, optimizer, train_loader, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Evaluate on validation set
    val_loss = evaluate(model, val_loader, device)
    print(f"Validation Loss: {val_loss:.4f}")
    evaluate_metrics(model, val_loader, device)

    # Update learning rate
    scheduler.step()

    # Save the model if validation loss has decreased
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_fasterrcnn_model.pth")
        print("Model saved.")

print("Training complete.")
