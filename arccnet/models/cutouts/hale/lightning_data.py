"""
PyTorch Lightning DataModule for Hale classification.
"""

import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from . import config
from .dataset import HaleDataset, get_fold_data


class HaleDataModule(pl.LightningDataModule):
    """Lightning DataModule for Hale classification."""

    def __init__(self, df: pd.DataFrame = None, fold_num: int = 1, batch_size: int = None, num_workers: int = None):
        super().__init__()
        self.df = df  # Cache the DataFrame to avoid repeated loading
        self.fold_num = fold_num
        self.batch_size = batch_size or config.BATCH_SIZE
        self.num_workers = num_workers or config.NUM_WORKERS

        # Initialize datasets to None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.label_mapping = None

    def setup(self, stage: str = None):
        """Set up datasets for different stages."""
        if stage == "fit" or stage is None:
            # Split data only once
            train_df, val_df, test_df = get_fold_data(self.df, self.fold_num)

            # Create datasets
            self.train_dataset = HaleDataset(train_df, data_type=config.DATA_TYPE)
            self.val_dataset = HaleDataset(val_df, data_type=config.DATA_TYPE)
            self.test_dataset = HaleDataset(test_df, data_type=config.DATA_TYPE)

            # Store label mapping from train dataset
            self.label_mapping = self.train_dataset.label_mapping

        if stage == "test":
            if self.test_dataset is None:
                _, _, test_df = get_fold_data(self.df, self.fold_num)
                self.test_dataset = HaleDataset(test_df, data_type=config.DATA_TYPE)

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
