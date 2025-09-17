"""
Dataset utilities for Hale classification using existing ARCCNet utilities.
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from astropy.io import fits

from arccnet.visualisation import utils as ut_v
from . import config


class HaleDataset(Dataset):
    """
    Simple dataset for Hale classification that supports magnetogram, continuum, or both data types.

    Args:
        data_folder: Root data directory
        dataset_folder: Dataset subdirectory
        df: DataFrame with image paths and labels
        transform: Optional transforms to apply to images
        data_type: Type of data to load - "magnetogram", "continuum", or "both"
    """

    def __init__(
        self,
        data_folder: str,
        dataset_folder: str,
        df: pd.DataFrame,
        transform=None,
        data_type: str = None,
        target_height: int = 128,
        target_width: int = 128,
        divisor: float = 800.0,
    ):
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.data_type = data_type or config.DATA_TYPE
        self.target_height = target_height
        self.target_width = target_width
        self.divisor = divisor

        # Validate data_type
        if self.data_type not in ["magnetogram", "continuum", "both"]:
            raise ValueError(f"data_type must be 'magnetogram', 'continuum', or 'both', got '{self.data_type}'")

        # Create label mapping from encoded_labels to model indices [0, 1, 2, ...]
        unique_labels = sorted(df["encoded_labels"].unique())
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        print(f"Dataset label mapping: {self.label_mapping}")

    def __len__(self):
        return len(self.df)

    def _load_fits_image(self, fits_path: str) -> np.ndarray:
        """Load and preprocess FITS image following McIntosh pattern."""
        try:
            with fits.open(fits_path, memmap=True) as hdul:
                # Try different HDU extensions
                if len(hdul) > 1 and hdul[1].data is not None:
                    image_data = np.array(hdul[1].data, dtype=np.float32)
                else:
                    image_data = np.array(hdul[0].data, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"Failed to load FITS file {fits_path}: {e}")

        # Handle NaN values
        image_data = np.nan_to_num(image_data, nan=0.0)

        # Apply hardtanh transformation (same as McIntosh)
        image_data = ut_v.hardtanh_transform_npy(image_data, divisor=self.divisor, min_val=-1.0, max_val=1.0)

        # Pad and resize to target dimensions (same as McIntosh)
        image_data = ut_v.pad_resize_normalize(
            image_data, target_height=self.target_height, target_width=self.target_width
        )

        return image_data

    def _get_image_path(self, row, data_type: str) -> str:
        """Get image path following EDA pattern."""
        if data_type == "magnetogram":
            # Prioritize HMI over MDI for magnetograms, following EDA pattern
            if pd.notna(row.get("path_image_cutout_hmi")) and row["path_image_cutout_hmi"] not in ["", "None"]:
                path = row["path_image_cutout_hmi"]
            elif pd.notna(row.get("path_image_cutout_mdi")) and row["path_image_cutout_mdi"] not in ["", "None"]:
                path = row["path_image_cutout_mdi"]
            else:
                raise ValueError(f"No valid {data_type} path found")

            # Convert absolute paths to relative paths (strip the old absolute prefix)
            if path.startswith("/mnt/ARCAFF/v0.3.0/04_final/"):
                # Remove the old absolute prefix and keep only the relative part
                path = path.replace("/mnt/ARCAFF/v0.3.0/04_final/", "")

            return path

        elif data_type == "continuum":
            # For continuum, we would use similar columns but they may not exist
            # For now, raise an error as continuum paths need to be verified
            raise ValueError("Continuum data type not yet implemented - need to check available columns")

        raise ValueError(f"No valid {data_type} path found")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.data_type == "both":
            # For now, "both" is not implemented since continuum paths need verification
            raise NotImplementedError("'both' data type not yet implemented - continuum paths need verification")

        else:
            # Load single data type
            img_path = self._get_image_path(row, self.data_type)

            # Convert to full path - img_path should now be relative
            fits_path = os.path.join(self.data_folder, self.dataset_folder, img_path)

            image_data = self._load_fits_image(fits_path)

            # Convert to tensor and add channel dimension - data is already preprocessed
            image_tensor = torch.from_numpy(image_data).unsqueeze(0).contiguous()  # (1, H, W)

        # Apply transforms if provided (but data is already normalized)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        # Ensure tensor is contiguous and properly typed
        image_tensor = image_tensor.contiguous().float()

        # Get label and map to model-compatible index [0, 1, 2]
        original_label = row["encoded_labels"]
        label = self.label_mapping[original_label]

        return image_tensor, label


def get_fold_data(df: pd.DataFrame, fold_num: int):
    """
    Get train/val/test data for a specific fold using existing fold columns.

    Args:
        df: DataFrame with fold columns (created by dataset_utils.split_data)
        fold_num: Fold number (1-N)

    Returns:
        train_df, val_df, test_df
    """
    fold_col = f"Fold {fold_num}"
    if fold_col not in df.columns:
        available_folds = [col for col in df.columns if col.startswith("Fold ")]
        raise ValueError(f"Fold column '{fold_col}' not found. Available: {available_folds}")

    train_df = df[df[fold_col] == "train"].copy()
    val_df = df[df[fold_col] == "val"].copy()
    test_df = df[df[fold_col] == "test"].copy()

    return train_df, val_df, test_df
