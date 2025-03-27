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

# %% [markdown]
# # Binary Solar Flare Classification Script


# %%
import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

from astropy.io import fits

from arccnet.models.flares import utils as ut_f
from arccnet.visualisation import utils as ut_v

# %%
# =============================================================================
# Configuration Constants
# =============================================================================
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "../../../../data")  # Base data directory
FLARES_PARQUET = "mag-pit-flare-dataset_1996-01-01_2023-01-01_dev.parq"  # Input flare catalog
CUTOUT_DATASET_FOLDER = "arccnet-cutout-dataset-v20240715"  # Subfolder containing FITS cutouts
FITS_SUBFOLDER = "fits"  # Specific subfolder for FITS files within CUTOUT_DATASET_FOLDER

TARGET_COLUMN = "flares_above_C"  # Target label for binary classification
STRATIFY_COLUMN = TARGET_COLUMN  # Column used for stratified splitting
FLARE_CLASSES = ["A", "B", "C", "M", "X"]  # GOES flare class names
TARGET_THRESHOLD_CLASS = "C"  # Threshold for binary classification (>= C)
MAGNETIC_CLASS_COLUMN = "magnetic_class"  # Column with McIntosh magnetic class
MAG_CLASS_MAPPING = {
    "Alpha": "α",
    "Beta": "β",
    "Beta-Delta": "β-δ",
    "Beta-Gamma": "β-γ",
    "Beta-Gamma-Delta": "β-γ-δ",
    "Gamma": "γ",
    "Gamma-Delta": "γ-δ",
}  # Mapping to Greek letters
MAG_CLASS_ORDER = ["α", "β", "β-δ", "β-γ", "β-γ-δ", "γ", "γ-δ"]  # Order for display

TEST_SIZE = 0.1  # Fraction for the test set
VAL_SIZE = 0.2  # Fraction for the validation set

IMG_TARGET_HEIGHT = 224
IMG_TARGET_WIDTH = 224
IMG_DIVISOR = 800.0  # Divisor for hardtanh normalization
IMG_MIN_VAL = -1.0  # Min value for hardtanh normalization
IMG_MAX_VAL = 1.0  # Max value for hardtanh normalization

MODEL_NAME = "vit_small_patch16_224"  # Name of the timm model
NUM_CLASSES = 1  # Binary classification
IN_CHANNELS = 1  # Grayscale images

BATCH_SIZE = 32
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
MAX_EPOCHS = 10
RANDOM_SEED = 42  # Seed for reproducibility in splitting

# =============================================================================

# %% [markdown]
# ## 1. Data Loading and Preparation

# %%
print("Loading flare data...")
flares_file_path = os.path.join(DATA_FOLDER, FLARES_PARQUET)
df_flares = pd.read_parquet(flares_file_path)
print(f"Loaded {len(df_flares)} records from {flares_file_path}")

print("Checking for corresponding FITS file existence...")
df_flares_exists, none_idxs = ut_f.check_fits_file_existence(df_flares.copy(), DATA_FOLDER, CUTOUT_DATASET_FOLDER)
df = df_flares_exists[df_flares_exists["file_exists"]].copy()
print(f"Found {len(df)} records with existing FITS files.")
if none_idxs:
    print(f"Warning: {len(none_idxs)} records had missing FITS paths initially.")

print(f"Calculating target column '{TARGET_COLUMN}' (flares >= {TARGET_THRESHOLD_CLASS})...")
try:
    threshold_idx = FLARE_CLASSES.index(TARGET_THRESHOLD_CLASS)
    columns_to_check = FLARE_CLASSES[threshold_idx:]
    for col in columns_to_check:
        if col not in df.columns:
            df[col] = 0
    df[TARGET_COLUMN] = (df[columns_to_check].fillna(0) > 0).any(axis=1).astype(int)
    print(f"Target column '{TARGET_COLUMN}' created.")
except ValueError:
    raise ValueError(f"TARGET_THRESHOLD_CLASS '{TARGET_THRESHOLD_CLASS}' not found in FLARE_CLASSES.")
except KeyError as e:
    raise KeyError(f"Required flare class column missing for target calculation: {e}")


print(f"Splitting data (Test: {TEST_SIZE:.0%}, Val: {VAL_SIZE:.0%}) stratified by '{STRATIFY_COLUMN}'...")
train_df, val_df, test_df = ut_f.split_dataframe(
    df=df, stratify_col=STRATIFY_COLUMN, test_size=TEST_SIZE, val_size=VAL_SIZE, random_state=RANDOM_SEED
)
print(f"Split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

# %% [markdown]
# ## 2. Data Distribution Analysis


# %%
def format_distribution_table(dfs_dict, column_name, class_mapping=None, class_order=None):
    """
    Calculates and formats distribution counts and percentages for a column
    across multiple dataframes (e.g., train, val, test).

    Args:
        dfs_dict: Dictionary where keys are dataset names (str) and values are pandas DataFrames.
        column_name: The name of the column to analyze.
        class_mapping: Optional dictionary to map original class names to display names.
        class_order: Optional list to specify the desired order of classes in the output index.

    Returns:
        A pandas DataFrame with counts and percentages formatted as strings.
    """
    dist_dict = {}
    for name, df_subset in dfs_dict.items():
        series = df_subset[column_name].copy()
        if class_mapping:
            series = series.map(class_mapping)
        series.fillna("Unknown", inplace=True)
        dist_dict[name] = series.value_counts().rename(name)

    dist_df = pd.concat(dist_dict, axis=1)

    if class_order:
        full_order = class_order + [idx for idx in dist_df.index if idx not in class_order]
        dist_df = dist_df.reindex(full_order)

    dist_df = dist_df.fillna(0).astype(int)
    formatted_df = pd.DataFrame(index=dist_df.index)
    for col in dist_df.columns:
        total = dist_df[col].sum()
        percentages = (dist_df[col] / total * 100).round(1) if total > 0 else 0.0
        formatted_df[col] = dist_df[col].astype(str) + " (" + percentages.astype(str) + "%)"

    return formatted_df


datasets_dict = {"Train": train_df, "Validation": val_df, "Test": test_df}

print("\nFlare Classification Distribution (>= C):")
flare_dist_table = format_distribution_table(datasets_dict, TARGET_COLUMN, class_mapping={0: "No", 1: "Yes"})
print(flare_dist_table)

print("\nMagnetic Class Distribution:")
mag_dist_table = format_distribution_table(datasets_dict, MAGNETIC_CLASS_COLUMN, MAG_CLASS_MAPPING, MAG_CLASS_ORDER)
print(mag_dist_table)


# %% [markdown]
# ## 3. Data Visualization

# %%
print("\nGenerating magnetic class histograms...")
datasets_list = [train_df, val_df, test_df]
titles = ["Train Set", "Validation Set", "Test Set"]

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(22, 6))
fig.suptitle("Magnetic Class Distribution Across Sets", fontsize=16)

for idx, df_subset in enumerate(datasets_list):
    current_ax = axes[idx]
    ut_v.make_classes_histogram(
        series=df_subset[MAGNETIC_CLASS_COLUMN],
        ax=current_ax,
        y_off=10,
        fontsize=11,
        text_fontsize=10,
        title=titles[idx],
        titlesize=13,
        show_percentages=True,
    )
    current_ax.tick_params(axis="x", labelrotation=45)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
print("Histograms displayed.")

# %% [markdown]
# ## 4. PyTorch Dataset and Modules


# %%
class FlareDataset(Dataset):
    """
    PyTorch Dataset for loading solar flare FITS images and corresponding labels.
    Handles image loading, preprocessing, and normalization.
    """

    def __init__(
        self,
        data_folder,
        dataset_folder,
        df,
        target_column,
        target_height=224,
        target_width=224,
        divisor=800.0,
        min_val=-1.0,
        max_val=1.0,
        transform=None,
    ):
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.df = df
        self.target_column = target_column
        self.transform = transform
        self.target_height = target_height
        self.target_width = target_width
        self.divisor = divisor
        self.min_val = min_val
        self.max_val = max_val
        self.fits_base_path = os.path.join(self.data_folder, self.dataset_folder, FITS_SUBFOLDER)

        if not all(col in df.columns for col in ["path_image_cutout_hmi", "path_image_cutout_mdi", self.target_column]):
            raise ValueError("DataFrame missing required columns (path_image_cutout_hmi/mdi or target column)")

    def _load_image(self, row):
        """Loads, preprocesses, and returns a single image tensor and its label."""
        path_key = "path_image_cutout_hmi" if pd.notna(row.get("path_image_cutout_hmi")) else "path_image_cutout_mdi"
        path_value = row.get(path_key)

        if path_value is None:
            raise FileNotFoundError(f"No valid image path found for row index {row.name}")

        base_filename = os.path.basename(path_value)
        fits_file_path = os.path.join(self.fits_base_path, base_filename)

        try:
            with fits.open(fits_file_path, memmap=True) as img_fits:
                image_data = np.array(img_fits[1].data, dtype=np.float32)
        except FileNotFoundError:
            raise FileNotFoundError(f"FITS file not found: {fits_file_path} (Index: {row.name})")
        except IndexError:
            raise IndexError(f"Cannot access HDU 1 in FITS file: {fits_file_path} (Index: {row.name})")
        except Exception as e:
            raise OSError(f"Error reading FITS file {fits_file_path} (Index: {row.name}): {e}")

        image_data = np.nan_to_num(image_data, nan=0.0)

        image_data = ut_v.hardtanh_transform_npy(
            image_data, divisor=self.divisor, min_val=self.min_val, max_val=self.max_val
        )
        image_data = ut_v.pad_resize_normalize(
            image_data, target_height=self.target_height, target_width=self.target_width
        )

        image_tensor = torch.from_numpy(image_data).unsqueeze(0)
        label = int(row[self.target_column])

        return image_tensor, label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx >= len(self.df):
            raise IndexError("Index out of range")
        row = self.df.iloc[idx]
        image, label = self._load_image(row)

        if self.transform:
            image = self.transform(image)

        return image, label


class FlareDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling train, validation, and test datasets/dataloaders.
    """

    def __init__(
        self,
        data_folder,
        dataset_folder,
        train_df,
        val_df,
        test_df,
        target_column,
        batch_size=32,
        num_workers=4,
        img_target_height=224,
        img_target_width=224,
        img_divisor=800.0,
        img_min_val=-1.0,
        img_max_val=1.0,
        pin_memory=True,
        train_transform=None,
        val_test_transform=None,
    ):
        super().__init__()
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.target_column = target_column
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_target_height = img_target_height
        self.img_target_width = img_target_width
        self.img_divisor = img_divisor
        self.img_min_val = img_min_val
        self.img_max_val = img_max_val
        self.pin_memory = pin_memory
        self.train_transform = train_transform
        self.val_test_transform = val_test_transform
        self.save_hyperparameters(ignore=["train_df", "val_df", "test_df"])

    def setup(self, stage=None):
        dataset_args = {
            "data_folder": self.data_folder,
            "dataset_folder": self.dataset_folder,
            "target_column": self.target_column,
            "target_height": self.img_target_height,
            "target_width": self.img_target_width,
            "divisor": self.img_divisor,
            "min_val": self.img_min_val,
            "max_val": self.img_max_val,
        }

        if stage == "fit" or stage is None:
            self.train_dataset = FlareDataset(df=self.train_df, transform=self.train_transform, **dataset_args)
            self.val_dataset = FlareDataset(df=self.val_df, transform=self.val_test_transform, **dataset_args)
        if stage == "validate" or stage == "fit" or stage is None:
            if not hasattr(self, "val_dataset"):
                self.val_dataset = FlareDataset(df=self.val_df, transform=self.val_test_transform, **dataset_args)
        if stage == "test" or stage is None:
            self.test_dataset = FlareDataset(df=self.test_df, transform=self.val_test_transform, **dataset_args)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )


class ViTFlareClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module wrapping a Vision Transformer (ViT) for binary flare classification.
    Includes loss, metrics, optimizer configuration, and training/validation/test steps.
    """

    def __init__(
        self,
        model_name="vit_small_patch16_224",
        num_classes=1,
        in_chans=1,
        pretrained=False,
        learning_rate=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, in_chans=in_chans)

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
        metric_output = metrics(preds, y)
        self.log(f"{metrics.prefix}loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metric_output, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, self.train_metrics)

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, self.val_metrics)

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, self.test_metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer


# %% [markdown]
# ## 5. Training Execution

# %%
if __name__ == "__main__":
    print("\n" + "=" * 30)
    print(" Starting Training Pipeline")
    print("=" * 30)

    print("\nInitializing DataModule...")
    data_module = FlareDataModule(
        data_folder=DATA_FOLDER,
        dataset_folder=CUTOUT_DATASET_FOLDER,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_column=TARGET_COLUMN,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        img_target_height=IMG_TARGET_HEIGHT,
        img_target_width=IMG_TARGET_WIDTH,
        img_divisor=IMG_DIVISOR,
        img_min_val=IMG_MIN_VAL,
        img_max_val=IMG_MAX_VAL,
    )
    print("DataModule initialized.")

    print("\nInitializing LightningModule (ViTFlareClassifier)...")
    model = ViTFlareClassifier(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        in_chans=IN_CHANNELS,
        learning_rate=LEARNING_RATE,
        pretrained=False,
    )
    print(f"Model '{MODEL_NAME}' initialized.")

    print("\nInitializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        devices="auto",
        deterministic=False,
        enable_progress_bar=True,
        logger=True,
        # callbacks=[...] # Add callbacks like ModelCheckpoint here if needed
    )
    print(f"Trainer initialized for {MAX_EPOCHS} epochs.")

    print("\nStarting training (trainer.fit)...")
    trainer.fit(model, data_module)
    print("Training finished.")

    print("\nStarting testing (trainer.test)...")
    test_results = trainer.test(model, data_module)
    print("Testing finished.")
    print("Test Results:", test_results)

    print("\n" + "=" * 30)
    print(" Pipeline Execution Complete")
    print("=" * 30)

# %% [markdown]
# --- End of Script ---
