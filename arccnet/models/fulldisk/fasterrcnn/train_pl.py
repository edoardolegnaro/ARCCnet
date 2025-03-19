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

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
import torchvision.ops as ops
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.ndimage import rotate
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection import MeanAveragePrecision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from astropy.io import fits
from astropy.time import Time

from arccnet.visualisation import utils as ut_v

# %%
# Disable beta transforms warnings
torchvision.disable_beta_transforms_warning()

# Set float32 matmul precision
torch.set_float32_matmul_precision("medium")

# %%
# Constants
IMG_SIZE = 512
BATCH_SIZE = 4
NUM_WORKERS = os.cpu_count() // 2
NUM_CLASSES = 4  # Background + 3 classes (Alpha, Beta, Beta-Gamma)


# %%
class FullDiskDataModule(pl.LightningDataModule):
    def __init__(self, data_root, df_path, batch_size=BATCH_SIZE, cache_dir=None):
        super().__init__()
        self.data_root = data_root
        self.df_path = df_path
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)]
        )

    def _group_boxes_by_image(self, df):
        grouped = (
            df.groupby("path")
            .agg({"bottom_left_cutout": list, "top_right_cutout": list, "encoded_label": list, "instrument": "first"})
            .reset_index()
        )
        return grouped

    def setup(self, stage=None):
        if self.cache_dir:
            cleaned_df = pd.read_parquet(os.path.join(self.cache_dir, "cleaned_fulldisk_df.parq"))

        else:
            # Load and preprocess dataframe
            df = pd.read_parquet(self.df_path)

            # Convert JD to datetime
            times = Time(df["datetime.jd1"] + df["datetime.jd2"], format="jd")
            df["datetime"] = pd.to_datetime(times.iso)

            # Filter and preprocess
            selected_df = df[~df["filtered"]]
            lon_trshld = 70
            min_size = 0.024
            front_df = selected_df[(selected_df["longitude"] < lon_trshld) & (selected_df["longitude"] > -lon_trshld)]
            cleaned_df = front_df.copy()
            img_size_dic = {"MDI": 1024, "HMI": 4096}
            for idx, row in cleaned_df.iterrows():
                x_min, y_min = row["bottom_left_cutout"]
                x_max, y_max = row["top_right_cutout"]

                img_sz = img_size_dic.get(row["instrument"])
                width = (x_max - x_min) / img_sz
                height = (y_max - y_min) / img_sz

                cleaned_df.at[idx, "width"] = width
                cleaned_df.at[idx, "height"] = height

            cleaned_df = cleaned_df[(cleaned_df["width"] >= min_size) & (cleaned_df["height"] >= min_size)]

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
            cleaned_df["grouped_label"] = cleaned_df["magnetic_class"].map(label_mapping)
            cleaned_df = cleaned_df[cleaned_df["grouped_label"] != "None"].copy()
            cleaned_df["encoded_label"] = cleaned_df["grouped_label"].map(label_to_index)

        df = self._group_boxes_by_image(cleaned_df)
        # Split data
        split_idx = int(0.8 * len(df))
        self.train_df = df[:split_idx]
        self.val_df = df[split_idx:]

        # Create datasets
        self.train_ds = FullDiskDataset(
            self.train_df, self.data_root, transform=self.transform, cache_dir=self.cache_dir
        )
        self.val_ds = FullDiskDataset(self.val_df, self.data_root, cache_dir=self.cache_dir)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn
        )


# %%
class FullDiskDataset(Dataset):
    def __init__(self, df, data_root, transform=None, cache_dir=None):
        self.df = df
        self.data_root = data_root
        self.transform = transform
        self.cache_dir = cache_dir
        self.size_mapping = {"MDI": 1024, "HMI": 4096}

        # Ensure cache directory exists
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arccnet_path_root = row["path"].split("/fits")[0]
        image_path = row["path"].replace(arccnet_path_root, self.data_root)
        image = self._load_image(image_path)
        boxes, labels = self._get_annotations(row)

        if self.transform:
            # Apply transformations and adjust boxes accordingly
            for t in self.transform.transforms:
                if isinstance(t, transforms.RandomHorizontalFlip) and torch.rand(1) < t.p:
                    image = transforms.functional.hflip(image)
                    # Flip boxes horizontally
                    width = image.shape[2]
                    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                if isinstance(t, transforms.RandomVerticalFlip) and torch.rand(1) < t.p:
                    image = transforms.functional.vflip(image)
                    # Flip boxes vertically
                    height = image.shape[1]
                    boxes[:, [1, 3]] = height - boxes[:, [3, 1]]

        return image, {"boxes": boxes, "labels": labels}

    def _load_image(self, path):
        if self.cache_dir:
            dataset_folder = os.path.basename(self.data_root)
            preprocessed_folder = os.path.basename(self.cache_dir)
            cache_path = path.replace(dataset_folder, preprocessed_folder).replace(".fits", ".npy")
            data = np.load(cache_path)

        else:
            with fits.open(os.path.join(self.data_root, path)) as hdul:
                data = hdul[1].data
                header = hdul[1].header

            data = np.nan_to_num(data)
            data = ut_v.hardtanh_transform_npy(data)
            crota2 = header.get("CROTA2", 0)
            data = rotate(data, crota2, reshape=False, mode="constant", cval=0)

        data = ut_v.pad_resize_normalize(data, target_height=IMG_SIZE, target_width=IMG_SIZE)
        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)  # 3 channels
        return data

    def _get_annotations(self, row):
        scale = IMG_SIZE / self.size_mapping[row["instrument"]]
        boxes = []

        for (x1, y1), (x2, y2) in zip(row["bottom_left_cutout"], row["top_right_cutout"]):
            boxes.append([x1 * scale, y1 * scale, x2 * scale, y2 * scale])

        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(row["encoded_label"], dtype=torch.int64)


# %%
def collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


# %%
class FasterRCNNModel(pl.LightningModule):
    def __init__(self, num_classes=NUM_CLASSES, lr=0.005):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.lr = lr
        self.map_metric = MeanAveragePrecision()
        # No need to move map_metric here

    # def on_fit_start(self):
    # Move metric to the correct device
    #    self.map_metric = self.map_metric.to(self.device)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        self.log("train_loss", loss, batch_size=BATCH_SIZE, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)
        self.map_metric.update(preds, targets)

        ious = []
        for pred, target in zip(preds, targets):
            if len(pred["boxes"]) > 0 and len(target["boxes"]) > 0:
                iou = ops.box_iou(pred["boxes"], target["boxes"])
                matched_iou = iou.max(dim=1).values.mean()
                ious.append(matched_iou)

        if ious:
            ious_tensor = torch.as_tensor(ious, device=self.device)
            self.log("val_iou", ious_tensor.mean(), batch_size=BATCH_SIZE, prog_bar=True, sync_dist=True)
        else:
            self.log(
                "val_iou",
                torch.as_tensor(0.0, device=self.device),
                batch_size=BATCH_SIZE,
                prog_bar=True,
                sync_dist=True,
            )

    def on_validation_epoch_end(self):
        map_metrics = self.map_metric.compute()
        self.log_dict(
            {"val_map": map_metrics["map"], "val_map_50": map_metrics["map_50"], "val_map_75": map_metrics["map_75"]},
            prog_bar=True,
            sync_dist=True,
        )
        self.map_metric.reset()

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9, weight_decay=0.0005)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.1, patience=3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_map", "interval": "epoch"},
        }


# %%
if __name__ == "__main__":
    # Set up paths
    data_folder = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data/")
    dataset_folder = "arccnet-fulldisk-dataset-v20240917"
    df_name = "fulldisk-detection-catalog-v20240917.parq"
    preprocessed_folder = "faster_rcnn_preprocessed"

    data_root = os.path.join(data_folder, dataset_folder)
    df_path = os.path.join(data_root, df_name)
    cache_dir = os.path.join(data_folder, preprocessed_folder)

    dm = FullDiskDataModule(data_root=data_root, df_path=df_path, cache_dir=cache_dir)

    model = FasterRCNNModel()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_map", mode="max", filename="best-{epoch}-{val_map:.2f}", save_top_k=3
    )

    early_stop = EarlyStopping(monitor="val_map", patience=3, mode="max", verbose=True)

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=[1],
        callbacks=[checkpoint_callback, early_stop],
        precision="16-mixed",
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, datamodule=dm)

# %%
