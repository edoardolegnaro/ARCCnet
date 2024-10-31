# %%
import os

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.ops as ops
from scipy.ndimage import rotate
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
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

selected_df = df[~df["filtered"]]

lon_trshld = 70
front_df = selected_df[(selected_df["longitude"] < lon_trshld) & (selected_df["longitude"] > -lon_trshld)]

min_size = 0.024

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
    title="Cleaned FullDisk Dataset",
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
final_size = 800


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
    data = ut_v.pad_resize_normalize(data, target_height=final_size, target_width=final_size)
    return data


# train_data = p_map(preprocess_FD, [row for _, row in train_df.iterrows()], desc='Preprocessing Train FD')
# val_data = p_map(preprocess_FD, [row for _, row in val_df.iterrows()], desc='Preprocessing Val FD')


# %%
class FulldiskDataset(Dataset):
    def __init__(self, df, local_path_root, transform=None, final_size=800):
        self.df = df
        self.local_path_root = local_path_root
        self.transform = transform
        self.final_size = final_size

    def __len__(self):
        return len(self.df)

    def _preprocess_FD(self, row):
        arccnet_path_root = row["path"].split("/fits")[0]
        image_path = row["path"].replace(arccnet_path_root, self.local_path_root)

        with fits.open(image_path) as img_fit:
            data = img_fit[1].data
            header = img_fit[1].header

        data = np.nan_to_num(data, nan=0.0)
        data = ut_v.hardtanh_transform_npy(data)
        crota2 = header.get("CROTA2", 0)  # Handle missing header key
        data = rotate(data, crota2, reshape=False, mode="constant", cval=0)
        data = ut_v.pad_resize_normalize(data, target_height=self.final_size, target_width=self.final_size)
        return data

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        data = self._preprocess_FD(row)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)  # Convert to 3 channels

        if self.transform:
            data = self.transform(data)

        label = row["encoded_label"]

        # Compute bounding box
        img_sz = img_size_dic.get(row["instrument"], self.final_size)  # Handle missing dictionary entry
        scale_factor = self.final_size / img_sz
        x_min, y_min = row["bottom_left_cutout"]
        x_max, y_max = row["top_right_cutout"]
        bbox = [
            x_min * scale_factor,
            y_min * scale_factor,
            x_max * scale_factor,
            y_max * scale_factor,
        ]  # Absolute coordinates

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
ut_t.replace_activations(model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01)
model.to(device)

# %% Training parameters
num_epochs = 20
learning_rate = 0.005
weight_decay = 0.0005
step_size = 5
gamma = 0.1
batch_size = 4
num_workers = os.cpu_count()

train_dataset = FulldiskDataset(train_df, local_path_root)
val_dataset = FulldiskDataset(val_df, local_path_root)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=lambda x: list(zip(*x)),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=lambda x: list(zip(*x)),
)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
scaler = torch.amp.GradScaler("cuda")


def compute_metrics(model, data_loader, device, phase="Validation"):
    """
    Computes loss components, IoU, and mAP metrics for a model on a given dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        phase (str): "Training" or "Validation" to indicate the phase of evaluation.

    Returns:
        dict: A dictionary containing average losses, IoU, and mAP.
    """
    is_validation = phase == "Validation"
    model.eval() if is_validation else model.train()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_detection_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    mAP_metric = MeanAveragePrecision()

    with torch.no_grad() if is_validation else torch.enable_grad():
        for images, targets in tqdm(data_loader, desc=phase, leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Compute losses
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            total_loss += loss.item()
            total_classification_loss += loss_dict.get("loss_classifier", torch.tensor(0.0)).item()
            total_detection_loss += loss_dict.get("loss_box_reg", torch.tensor(0.0)).item()

            if is_validation:
                # Predictions for metrics calculation
                predictions = model(images)
                mAP_metric.update(predictions, targets)

                # Compute IoU for each prediction-target pair
                for pred, tgt in zip(predictions, targets):
                    pred_boxes = pred["boxes"]
                    tgt_boxes = tgt["boxes"]

                    if pred_boxes.size(0) > 0 and tgt_boxes.size(0) > 0:
                        ious = ops.box_iou(pred_boxes, tgt_boxes)
                        iou = ious.diag().mean().item() if ious.numel() > 0 else 0.0
                    else:
                        iou = 0.0

                    total_iou += iou
                    total_samples += 1

    # Calculate averages
    avg_loss = total_loss / len(data_loader)
    avg_classification_loss = total_classification_loss / len(data_loader)
    avg_detection_loss = total_detection_loss / len(data_loader)
    avg_iou = total_iou / total_samples if total_samples > 0 else 0.0
    mAP = mAP_metric.compute()

    return {
        "avg_loss": avg_loss,
        "avg_classification_loss": avg_classification_loss,
        "avg_detection_loss": avg_detection_loss,
        "avg_iou": avg_iou,
        "mAP": mAP,
    }


def train_one_epoch(model, optimizer, data_loader, device):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(data_loader, desc="Training", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with torch.amp.autocast("cuda"):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimization
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()

    return total_loss / len(data_loader)


def print_metrics(phase, metrics):
    """
    Prints the evaluation metrics for a given phase (training or validation).

    Args:
        phase (str): "Training" or "Validation" phase.
        metrics (dict): Dictionary containing evaluation metrics.
    """
    print(f"{phase} Metrics:")
    print(f"  - Total Loss: {metrics['avg_loss']:.4f}")
    print(f"  - Classification Loss: {metrics['avg_classification_loss']:.4f}")
    print(f"  - Detection Loss: {metrics['avg_detection_loss']:.4f}")
    print(f"  - Mean IoU: {metrics['avg_iou']:.4f}")


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


def evaluate_full(model, data_loader, device):
    """
    Evaluates the model on the validation dataset, computing loss components, IoU, and mAP.

    Args:
        model (torch.nn.Module): The Faster R-CNN model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to perform computations on.

    Returns:
        dict: A dictionary containing average losses, IoU, and mAP.
    """
    model.eval()

    total_loss = 0.0
    total_classification_loss = 0.0
    total_detection_loss = 0.0
    total_iou = 0.0
    total_samples = 0

    mAP_metric = MeanAveragePrecision()

    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Validation", leave=False):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # **First Forward Pass**: Compute losses
            model.train()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            total_loss += loss.item()
            total_classification_loss += loss_dict.get("loss_classifier", 0.0).item()
            total_detection_loss += loss_dict.get("loss_box_reg", 0.0).item()
            model.eval()
            # **Second Forward Pass**: Get predictions
            predictions = model(images)

            mAP_metric.update(predictions, targets)

            # Compute IoU for each image in the batch
            for pred, tgt in zip(predictions, targets):
                pred_boxes = pred["boxes"]
                tgt_boxes = tgt["boxes"]

                if len(pred_boxes) == 0 or len(tgt_boxes) == 0:
                    iou = 0.0
                else:
                    # Compute IoU between predicted and target boxes
                    ious = ops.box_iou(pred_boxes, tgt_boxes)

                    # Match each predicted box to the target box with the highest IoU
                    if ious.numel() > 0:
                        # For simplicity, consider the diagonal (assuming one-to-one correspondence)
                        iou = ious.diag().mean().item()
                    else:
                        iou = 0.0

                total_iou += iou
                total_samples += 1

    # Compute average losses and metrics
    avg_loss = total_loss / len(data_loader)
    avg_classification_loss = total_classification_loss / len(data_loader)
    avg_detection_loss = total_detection_loss / len(data_loader)
    avg_iou = total_iou / total_samples if total_samples > 0 else 0.0
    mAP = mAP_metric.compute()

    # Print the computed metrics
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"  - Classification Loss: {avg_classification_loss:.4f}")
    print(f"  - Detection Loss: {avg_detection_loss:.4f}")
    print(f"Mean IoU: {avg_iou:.4f}")
    print(f"Mean Average Precision (mAP): {mAP['map'].item():.4f}")

    # Return the metrics as a dictionary for further use if needed
    return {
        "avg_loss": avg_loss,
        "avg_classification_loss": avg_classification_loss,
        "avg_detection_loss": avg_detection_loss,
        "avg_iou": avg_iou,
        "mAP": mAP,  # Return the entire mAP dictionary
    }


# %% Training loop
best_val_loss = float("inf")
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Train for one epoch
    train_loss = train_one_epoch(model, optimizer, train_loader, device)
    print(f"Train Loss: {train_loss:.4f}")

    # Evaluate on validation set
    val_loss = evaluate_full(model, val_loader, device)
    # evaluate_metrics(model, val_loader, device)

    # Update learning rate
    scheduler.step()

    # Save the model if validation loss has decreased
    if val_loss["avg_loss"] < best_val_loss:
        best_val_loss = val_loss["avg_loss"]
        torch.save(model.state_dict(), "best_fasterrcnn_model.pth")
        print("Model saved.")

print("Training complete.")

# %%
