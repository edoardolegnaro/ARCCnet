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
#     display_name: ARCAFF
#     language: python
#     name: python3
# ---

# %%
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from arccnet.models.flares import utils as ut_f
from arccnet.visualisation import utils as ut_v

# %%
# prepare dataset
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../data")
df_flares_name = "mag-pit-flare-dataset_1996-01-01_2023-01-01_dev.parq"
df_flares = pd.read_parquet(os.path.join(data_folder, df_flares_name))

dataset_folder = "arccnet-cutout-dataset-v20240715"
df_file_name = "cutout-mcintosh-catalog-v20240715.parq"

df_flares_exists, none_idxs = ut_f.check_fits_file_existence(df_flares.copy(), data_folder, dataset_folder)
df = df_flares_exists[df_flares_exists["file_exists"]].copy()

flare_classes = ["A", "B", "C", "M", "X"]
for i in range(len(flare_classes)):
    threshold_class = flare_classes[i]
    columns_to_check = flare_classes[i:]  # Select columns from current class onward
    df[f"flares_above_{threshold_class}"] = (df[columns_to_check].fillna(0) > 0).any(axis=1).astype(int)

# %%
train_df, val_df, test_df = ut_f.split_dataframe(df=df, stratify_col="flares_above_C", test_size=0.1, val_size=0.2)

# %%
# For flare classification
flare_dist = pd.concat(
    [
        train_df["flares_above_C"].value_counts().rename("Train"),
        val_df["flares_above_C"].value_counts().rename("Validation"),
        test_df["flares_above_C"].value_counts().rename("Test"),
    ],
    axis=1,
)

# Combine counts and percentages in the same columns
for col in flare_dist.columns:
    percentages = (flare_dist[col] / flare_dist[col].sum() * 100).round(1)
    flare_dist[col] = flare_dist[col].astype(str) + " (" + percentages.astype(str) + "%)"

print("\nFlare Classification Distribution:")
flare_dist

# %%
# Define the correct order with Greek letters
class_order = ["α", "β", "β-δ", "β-γ", "β-γ-δ", "γ", "γ-δ"]

# Create mapping from original class names to Greek letter representation
class_mapping = {
    "Alpha": "α",
    "Beta": "β",
    "Beta-Delta": "β-δ",
    "Beta-Gamma": "β-γ",
    "Beta-Gamma-Delta": "β-γ-δ",
    "Gamma": "γ",
    "Gamma-Delta": "γ-δ",
}

# For magnetic class - first map the class names
mag_dist = pd.concat(
    [
        train_df["magnetic_class"].map(class_mapping).value_counts().rename("Train"),
        val_df["magnetic_class"].map(class_mapping).value_counts().rename("Validation"),
        test_df["magnetic_class"].map(class_mapping).value_counts().rename("Test"),
    ],
    axis=1,
)

# Reindex to ensure correct order
mag_dist = mag_dist.reindex(class_order)

# Combine counts and percentages in the same columns
for col in mag_dist.columns:
    percentages = (mag_dist[col] / mag_dist[col].sum() * 100).round(1)
    mag_dist[col] = mag_dist[col].astype(str) + " (" + percentages.astype(str) + "%)"

# Fill NaN values with "0 (0%)" if any classes are missing
mag_dist = mag_dist.fillna("0 (0%)")

print("\nMagnetic Class Distribution:")
mag_dist

# %%
datasets = [train_df, val_df, test_df]
titles = ["Train Set", "Validation Set", "Test Set"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(22, 6))

for idx, df in enumerate(datasets):
    current_ax = axes[idx]
    ut_v.make_classes_histogram(
        series=df["magnetic_class"],
        ax=current_ax,
        y_off=10,
        fontsize=11,
        text_fontsize=10,
        title=titles[idx],
        titlesize=13,
        show_percentages=True,
    )

plt.tight_layout()
plt.show()

# %%
import os

import numpy as np
import pytorch_lightning as pl
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

from astropy.io import fits


# 1. First define the FlareDataset class
class FlareDataset(Dataset):
    def __init__(
        self, data_folder, dataset_folder, df, transform=None, target_height=224, target_width=224, divisor=800.0
    ):
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.df = df
        self.transform = transform
        self.target_height = target_height
        self.target_width = target_width
        self.divisor = divisor

    def _load_image(self, row):
        path_key = "path_image_cutout_hmi" if row["path_image_cutout_hmi"] is not None else "path_image_cutout_mdi"
        path_value = row[path_key]
        base_filename = os.path.basename(path_value)
        fits_file_path = os.path.join(self.data_folder, self.dataset_folder, "fits", base_filename)

        with fits.open(fits_file_path, memmap=True) as img_fits:
            image_data = np.array(img_fits[1].data, dtype=np.float32)
        image_data = np.nan_to_num(image_data, nan=0.0)
        image_data = ut_v.hardtanh_transform_npy(image_data, divisor=self.divisor, min_val=-1.0, max_val=1.0)
        image_data = ut_v.pad_resize_normalize(
            image_data, target_height=self.target_height, target_width=self.target_width
        )

        image_data = torch.from_numpy(image_data).unsqueeze(0)  # Add channel dimension

        return image_data, row["flares_above_C"]  # Changed to match your target column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image, label = self._load_image(row)

        if self.transform:
            image = self.transform(image)

        return image, label


# 2. Then define the DataModule
class FlareDataModule(pl.LightningDataModule):
    def __init__(self, data_folder, dataset_folder, train_df, val_df, test_df, batch_size=32, num_workers=4):
        super().__init__()
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = FlareDataset(self.data_folder, self.dataset_folder, self.train_df)
        self.val_dataset = FlareDataset(self.data_folder, self.dataset_folder, self.val_df)
        self.test_dataset = FlareDataset(self.data_folder, self.dataset_folder, self.test_df)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


# 3. Finally define the LightningModule
class ViTFlareClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=1, in_chans=1)

        self.loss_fn = nn.BCEWithLogitsLoss()
        metrics = MetricCollection(
            {"acc": BinaryAccuracy(), "precision": BinaryPrecision(), "recall": BinaryRecall(), "f1": BinaryF1Score()}
        )

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, metrics):
        x, y = batch
        y_logits = self(x).squeeze()
        loss = self.loss_fn(y_logits, y.float())

        preds = (torch.sigmoid(y_logits) > 0.5).long()
        metrics(preds, y)

        self.log_dict(metrics, prog_bar=True)
        self.log(f"{metrics.prefix}loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, self.val_metrics)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, self.test_metrics)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# Usage example
if __name__ == "__main__":
    # Initialize data module
    data_module = FlareDataModule(
        data_folder=data_folder,
        dataset_folder=dataset_folder,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        batch_size=32,
        num_workers=4,
    )

    # Initialize model
    model = ViTFlareClassifier(learning_rate=1e-4)

    # Train with basic configuration
    trainer = pl.Trainer(max_epochs=10, accelerator="auto", devices=1, deterministic=True, enable_progress_bar=True)
    trainer.fit(model, data_module)

    # Evaluate
    trainer.test(model, data_module)

# %%
