import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from astropy.io import fits

from arccnet.visualisation import utils as ut_v


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
        # Dataset stores FITS files under data/cutout_classification/fits.
        self.fits_base_path = os.path.join(
            self.data_folder, self.dataset_folder, "data", "cutout_classification", "fits"
        )

        if not all(col in df.columns for col in ["path_image_cutout_hmi", "path_image_cutout_mdi", self.target_column]):
            raise ValueError("DataFrame missing required columns (path_image_cutout_hmi/mdi or target column)")

    def _load_image(self, row):
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
        # Avoid saving large dataframes in checkpoints
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
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=False,
        )

    def teardown(self, stage=None):
        """Clean up after fit or test."""
        # Clean up the datasets when done
        if stage == "fit" or stage is None:
            if hasattr(self, "train_dataset"):
                del self.train_dataset
            if hasattr(self, "val_dataset"):
                del self.val_dataset
        if stage == "test" or stage is None:
            if hasattr(self, "test_dataset"):
                del self.test_dataset
