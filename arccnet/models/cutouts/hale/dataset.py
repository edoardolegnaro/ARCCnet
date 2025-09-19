"""
Dataset utilities for Hale classification using existing ARCCNet utilities.
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import v2

from astropy.io import fits

import arccnet.models.cutouts.hale.config as config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_transforms(is_training: bool = False):
    """Create torchvision transform pipeline."""

    # Augmentation transforms (only during training) - using v2 transforms like old config
    if is_training and config.USE_AUGMENTATION:
        augmentation_transforms = v2.Compose(
            [
                v2.RandomVerticalFlip(),
                v2.RandomHorizontalFlip(),
                v2.RandomPerspective(distortion_scale=config.PERSPECTIVE_DISTORTION_SCALE, p=config.PERSPECTIVE_PROB),
                v2.RandomAffine(
                    degrees=config.ROTATION_DEGREES,
                    translate=config.AFFINE_TRANSLATE,
                    scale=config.AFFINE_SCALE,
                    shear=config.AFFINE_SHEAR,
                ),
            ]
        )
        return augmentation_transforms

    # For validation/test, return None (no transforms)
    return None


def get_fold_data(df: pd.DataFrame, fold_num: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Get train/val/test splits for a specific fold."""
    train_df = df[df[f"Fold {fold_num}"] == "train"].copy()
    val_df = df[df[f"Fold {fold_num}"] == "val"].copy()
    test_df = df[df[f"Fold {fold_num}"] == "test"].copy()
    return train_df, val_df, test_df


class HaleDataset(Dataset):
    """Dataset for Hale classification."""

    def __init__(
        self,
        df: pd.DataFrame,
        data_type: str = "magnetogram",
        divisor: float = None,
        target_height: int = None,
        target_width: int = None,
        is_training: bool = False,
    ):
        self.df = df.copy()
        self.data_type = data_type
        self.divisor = divisor or config.IMAGE_DIVISOR
        self.target_height = target_height or config.IMAGE_TARGET_HEIGHT
        self.target_width = target_width or config.IMAGE_TARGET_WIDTH
        self.is_training = is_training

        # Initialize transforms using torchvision
        self.transform = get_transforms(is_training=is_training)

        # Verify that model_labels column exists
        if "model_labels" not in self.df.columns:
            raise ValueError("DataFrame must contain 'model_labels' column with contiguous indices starting from 0")

        # Verify labels are contiguous starting from 0
        unique_labels = sorted(self.df["model_labels"].unique())
        expected_labels = list(range(len(unique_labels)))
        if unique_labels != expected_labels:
            raise ValueError(
                f"model_labels must be contiguous starting from 0. Got: {unique_labels}, Expected: {expected_labels}"
            )

        # Only log once to avoid spam from multiple dataset instances
        if not hasattr(HaleDataset, "_logged_labels"):
            # Convert to regular Python ints for cleaner logging
            clean_labels = [int(label) for label in unique_labels]
            label_to_grouped = {int(k): v for k, v in self.df.groupby("model_labels")["grouped_labels"].first().items()}

            logger.info(f"Dataset using model_labels: {clean_labels}")
            logger.info(f"Original grouped_labels mapping: {label_to_grouped}")
            HaleDataset._logged_labels = True

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
        """Load and preprocess image using more standard approaches."""
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

        # Convert to tensor for easier processing
        image_tensor = torch.from_numpy(image_data).unsqueeze(0)  # Add channel dimension

        # Apply normalization (equivalent to hardtanh_transform_npy but using torch)
        image_tensor = image_tensor / self.divisor
        image_tensor = torch.clamp(image_tensor, config.HARDTANH_MIN_VAL, config.HARDTANH_MAX_VAL)

        # Resize using torchvision's functional interface (maintains aspect ratio better)
        image_tensor = transforms.functional.resize(
            image_tensor, size=[self.target_height, self.target_width], antialias=True
        )

        return image_tensor

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

        # Apply transforms (augmentations only during training)
        if self.transform is not None:
            image_data = self.transform(image_data)

        # Get label (using model_labels which are contiguous starting from 0)
        label = torch.tensor(row["model_labels"], dtype=torch.long)

        return image_data, label
