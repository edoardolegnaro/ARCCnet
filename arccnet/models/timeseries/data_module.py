"""PyTorch Lightning data module for timeseries flare forecasting."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .config import *
from .dataset import SDOTimeseriesDataset


class FlareDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for flare forecasting."""

    def __init__(
        self,
        manifest_df,
        train_mask,
        val_mask,
        test_mask,
        data_dir,
        norm_stats=None,
        task_type="multiclass",
        batch_size=32,
        num_workers=4,
    ):
        """
        Initialize the data module.

        Parameters
        ----------
        manifest_df : pd.DataFrame
            Full dataset manifest
        train_mask : pd.Series
            Boolean mask for training samples
        val_mask : pd.Series
            Boolean mask for validation samples
        test_mask : pd.Series
            Boolean mask for test samples
        data_dir : Path
            Directory containing the data
        norm_stats : dict or None
            Normalization statistics (computed from train if None)
        task_type : str
            'multiclass' or 'regression'
        batch_size : int
            Batch size for dataloaders
        num_workers : int
            Number of workers for dataloaders
        """
        super().__init__()
        self.manifest_df = manifest_df
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
        self.data_dir = data_dir
        self.norm_stats = norm_stats
        self.task_type = task_type
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Setup datasets for each stage."""
        # Create datasets
        self.train_dataset = SDOTimeseriesDataset(
            self.manifest_df[self.train_mask].reset_index(drop=True),
            self.data_dir,
            norm_stats=self.norm_stats,
            task_type=self.task_type,
        )

        self.val_dataset = SDOTimeseriesDataset(
            self.manifest_df[self.val_mask].reset_index(drop=True),
            self.data_dir,
            norm_stats=self.norm_stats,
            task_type=self.task_type,
        )

        self.test_dataset = SDOTimeseriesDataset(
            self.manifest_df[self.test_mask].reset_index(drop=True),
            self.data_dir,
            norm_stats=self.norm_stats,
            task_type=self.task_type,
        )

    def train_dataloader(self):
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
