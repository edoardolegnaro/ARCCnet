"""
Dataset utilities for Hale classification using existing ARCCNet utilities.
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from astropy.io import fits

from arccnet.visualisation import utils as ut_v

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_fold_data(df: pd.DataFrame, fold_num: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get train/val/test splits for a specific fold."""
    train_df = df[df[f"fold_{fold_num}"] == "train"].copy()
    val_df = df[df[f"fold_{fold_num}"] == "val"].copy()
    test_df = df[df[f"fold_{fold_num}"] == "test"].copy()
    return train_df, val_df, test_df


class HaleDataset(Dataset):
    """Dataset for Hale classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_type: str = "magnetogram",
        divisor: float = 1000.0,
        target_height: int = 200,
        target_width: int = 200,
    ):
        self.df = df.copy()
        self.data_type = data_type
        self.divisor = divisor
        self.target_height = target_height
        self.target_width = target_width

        # Create label mapping from unique labels to contiguous indices
        unique_labels = sorted(self.df["grouped_labels_encoded"].unique())
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        logger.info(f"Dataset label mapping: {self.label_mapping}")

        # Apply label mapping to create model-compatible labels
        self.df["model_label"] = self.df["grouped_labels_encoded"].map(self.label_mapping)

    def __len__(self):
        return len(self.df)

    def _convert_old_path_to_new(self, old_path: str) -> str:
        """Convert old absolute paths to new relative paths."""
        if old_path.startswith("/mnt/ARCAFF/v0.3.0/04_final/"):
            # Convert to new path structure
            relative_part = old_path.replace("/mnt/ARCAFF/v0.3.0/04_final/", "")
            new_path = f"/ARCAFF/data/arccnet-v20250805/04_final/{relative_part}"
            return new_path
        return old_path

    def _load_image(self, row: pd.Series) -> torch.Tensor:
        """Load and preprocess image following McIntosh dataset pattern."""
        # Get the appropriate path
        path_key = "path_image_cutout_hmi" if row["path_image_cutout_hmi"] != "" else "path_image_cutout_mdi"
        fits_file_path = self._convert_old_path_to_new(row[path_key])

        try:
            with fits.open(fits_file_path, memmap=True) as img_fits:
                image_data = img_fits[0].data.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load FITS file {fits_file_path}: {e}")

        # Handle NaN values
        image_data = np.nan_to_num(image_data, nan=0.0)

        # Apply transformations following McIntosh pattern
        image_data = ut_v.hardtanh_transform_npy(image_data, divisor=self.divisor, min_val=-1.0, max_val=1.0)
        image_data = ut_v.pad_resize_normalize(
            image_data, target_height=self.target_height, target_width=self.target_width
        )

        return torch.from_numpy(image_data).unsqueeze(0)  # Add channel dimension

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image data
        if self.data_type == "magnetogram":
            image_data = self._load_image(row)
        elif self.data_type == "continuum":
            # TODO: Implement continuum loading when available
            raise NotImplementedError("Continuum data loading not implemented yet")
        elif self.data_type == "both":
            # TODO: Implement combined loading when available
            raise NotImplementedError("Combined data loading not implemented yet")
        else:
            raise ValueError(f"Unsupported data_type: {self.data_type}")

        # Get label (already mapped to contiguous indices)
        label = torch.tensor(row["model_label"], dtype=torch.long)

        return image_data, label
