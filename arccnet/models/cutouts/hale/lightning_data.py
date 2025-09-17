"""
PyTorch Lightning DataModule for Hale classification.
"""

import pandas as pd
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import arccnet.models.cutouts.hale.config as config
from .dataset import HaleDataset, get_fold_data


class HaleDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for Hale classification.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        fold_num: int = 1,
        data_folder: str = config.DATA_FOLDER,
        dataset_folder: str = config.DATASET_FOLDER,
        batch_size: int = config.BATCH_SIZE,
        num_workers: int = config.NUM_WORKERS,
        data_type: str = None,
    ):
        super().__init__()
        self.df = df
        self.fold_num = fold_num
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_type = data_type or config.DATA_TYPE

        # Since images are already preprocessed to fixed size with hardtanh normalization,
        # we only need minimal transforms (or none at all)
        self.train_transforms = transforms.Compose(
            [
                # Data is already normalized with hardtanh to [-1, 1] range
                # No need for additional normalization or resizing
            ]
        )

        self.val_test_transforms = transforms.Compose(
            [
                # Data is already normalized with hardtanh to [-1, 1] range
                # No need for additional normalization or resizing
            ]
        )

    def setup(self, stage: str | None = None):
        """Split data into train/val/test based on fold."""
        train_df, val_df, test_df = get_fold_data(self.df, self.fold_num)

        if stage == "fit" or stage is None:
            self.train_dataset = HaleDataset(
                self.data_folder,
                self.dataset_folder,
                train_df,
                transform=self.train_transforms,
                data_type=self.data_type,
            )

            self.val_dataset = HaleDataset(
                self.data_folder,
                self.dataset_folder,
                val_df,
                transform=self.val_test_transforms,
                data_type=self.data_type,
            )

            # Set label mapping from the dataset for compatibility
            self.label_mapping = self.train_dataset.label_mapping

        if stage == "test" or stage is None:
            self.test_dataset = HaleDataset(
                self.data_folder,
                self.dataset_folder,
                test_df,
                transform=self.val_test_transforms,
                data_type=self.data_type,
            )

            # Ensure label_mapping is available for test stage
            if not hasattr(self, "label_mapping"):
                self.label_mapping = self.test_dataset.label_mapping

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None,
            multiprocessing_context="spawn" if self.num_workers > 0 else None,
        )

    def get_train_labels(self):
        """Get the original encoded labels from training dataset for class weight computation."""
        return self.train_dataset.df["encoded_labels"].values
