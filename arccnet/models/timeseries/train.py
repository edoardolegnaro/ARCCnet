"""Training script for timeseries flare forecasting using PyTorch Lightning."""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.utils.class_weight import compute_class_weight

from arccnet.models import train_utils
from arccnet.models.checkpoint_manager import CheckpointManager

torch.set_float32_matmul_precision("medium")

# Handle both module and direct execution
if __name__ == "__main__" and __package__ is None:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
    from arccnet.models.timeseries.config import *
    from arccnet.models.timeseries.data_module import FlareDataModule
    from arccnet.models.timeseries.dataset import SDOTimeseriesDataset
    from arccnet.models.timeseries.lightning_module import FlareForecasterLightning
    from arccnet.models.timeseries.manifest import build_dataset
    from arccnet.models.timeseries.splitters import get_split
else:
    from .config import *
    from .data_module import FlareDataModule
    from .dataset import SDOTimeseriesDataset
    from .lightning_module import FlareForecasterLightning
    from .manifest import build_dataset
    from .splitters import get_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main(args):
    """Main training function using PyTorch Lightning."""

    # Set GPU device from config
    if GPU_ID is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
        logger.info(f"Using GPU {GPU_ID}")

    # Set random seed for reproducibility
    train_utils.set_global_seed(SEED, deterministic=True)
    logger.info(f"Random seed set to {SEED}")

    # Task type
    task_type = args.task_type if args.task_type else TASK_TYPE
    logger.info(f"Task type: {task_type}")

    # Build or load dataset manifest
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest_path) if args.manifest_path else output_dir / "manifest.parq"

    if manifest_path.exists():
        logger.info(f"Loading existing manifest from {manifest_path}")
        manifest_df = pd.read_parquet(manifest_path)
    else:
        logger.info(f"Building dataset from {data_root}")
        manifest_df = build_dataset(data_root, output_path=manifest_path)

    logger.info(f"Total samples: {len(manifest_df)}")

    # Split dataset
    train_mask, val_mask, test_mask = get_split(manifest_df, strategy="noaa", train_frac=0.65, val_frac=0.15, seed=SEED)

    logger.info("NOAA-based split:")
    logger.info(f"  Train: {train_mask.sum()} samples from {manifest_df[train_mask]['noaa_ar'].nunique()} ARs")
    logger.info(f"  Val:   {val_mask.sum()} samples from {manifest_df[val_mask]['noaa_ar'].nunique()} ARs")
    logger.info(f"  Test:  {test_mask.sum()} samples from {manifest_df[test_mask]['noaa_ar'].nunique()} ARs")

    # Compute class weights from training data for multiclass task
    class_weights_computed = None
    if task_type == "multiclass":
        train_labels = manifest_df[train_mask]["flare_class"].values
        unique_classes = np.unique(train_labels)

        # Compute balanced class weights
        weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=train_labels)

        # Create weight array for all classes (including those not in training set)
        class_weights_computed = np.ones(NUM_CLASSES)
        for cls, weight in zip(unique_classes, weights):
            class_weights_computed[int(cls)] = weight

        class_weights_computed = class_weights_computed.tolist()

        logger.info("Class distribution in training set:")
        for cls in range(NUM_CLASSES):
            count = (train_labels == cls).sum()
            pct = 100 * count / len(train_labels)
            weight = class_weights_computed[cls]
            class_names = {0: "C", 1: "M", 2: "X"}
            logger.info(f"  Class {cls} ({class_names[cls]}): {count} samples ({pct:.1f}%), weight: {weight:.3f}")

    # Compute normalization stats from training set
    logger.info("Computing normalization stats from training set...")
    train_dataset_temp = SDOTimeseriesDataset(
        manifest_df[train_mask].reset_index(drop=True),
        data_root,
        norm_stats=None,
        task_type=task_type,
    )
    norm_stats = train_dataset_temp.get_norm_stats()

    # Save normalization stats
    norm_stats_path = output_dir / "norm_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    logger.info(f"Normalization stats saved to {norm_stats_path}")

    # Create DataModule
    datamodule = FlareDataModule(
        manifest_df=manifest_df,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        data_dir=data_root,
        norm_stats=norm_stats,
        task_type=task_type,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Create Lightning model
    model = FlareForecasterLightning(
        task_type=task_type,
        num_channels=NUM_CHANNELS,
        output_dim=NUM_CLASSES if task_type == "multiclass" else REGRESSION_TARGETS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        class_weights=class_weights_computed if task_type == "multiclass" else None,
        # Model architecture parameters
        spatial_feature_dim=SPATIAL_FEATURE_DIM,
        temporal_num_layers=TEMPORAL_NUM_LAYERS,
        temporal_num_heads=TEMPORAL_NUM_HEADS,
        temporal_dim_feedforward=TEMPORAL_DIM_FEEDFORWARD,
        temporal_dropout=TEMPORAL_DROPOUT,
        temporal_pooling=TEMPORAL_POOLING,
        pretrained_spatial=PRETRAINED_SPATIAL,
        freeze_spatial=FREEZE_SPATIAL,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    )

    # Setup checkpoint manager
    checkpoint_mgr = CheckpointManager(
        root_name=f"timeseries/{task_type}",
        data_folder="/ARCAFF/data",
        model_name="resnet34_transformer",
        loss_function="cross_entropy" if task_type == "multiclass" else "mse",
    )

    # Callbacks
    callbacks = [
        checkpoint_mgr.get_checkpoint_callback(
            monitor="val/primary_metric",
            mode="max",
        ),
        EarlyStopping(
            monitor="val/primary_metric",
            patience=15,
            mode="max",
            verbose=True,
        ),
    ]

    # Add learning rate finder if enabled in config
    if FIND_LR:
        logger.info("Learning rate finder enabled")
        callbacks.append(
            LearningRateFinder(
                min_lr=LR_FIND_MIN,
                max_lr=LR_FIND_MAX,
                num_training_steps=LR_FIND_NUM_STEPS,
            )
        )

    # Logger (optional TensorBoard)
    try:
        tb_logger = TensorBoardLogger(
            save_dir=output_dir,
            name="lightning_logs",
        )
        logger.info("TensorBoard logging enabled")
    except ModuleNotFoundError:
        tb_logger = None
        logger.warning("TensorBoard not available - logging to CSV only")

    # Create Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
        logger=tb_logger,
        accelerator="auto",
        devices=1,  # Use single device to avoid distributed training issues
        precision="16-mixed" if torch.cuda.is_available() else "32-true",
        log_every_n_steps=10,
        deterministic=True,
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(model, datamodule)

    # Test on best model
    logger.info("Evaluating best model on test set...")
    trainer.test(model, datamodule, ckpt_path="best")

    # Save final results summary
    results = {
        "task_type": task_type,
        "checkpoint_dir": str(checkpoint_mgr.checkpoint_dir),
        "best_checkpoint": str(checkpoint_mgr.checkpoint_dir / "best.ckpt"),
    }

    results_path = output_dir / "training_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training complete! Results saved to {results_path}")
    logger.info(f"Best checkpoint: {checkpoint_mgr.checkpoint_dir / 'best.ckpt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train timeseries flare forecasting model with PyTorch Lightning")

    parser.add_argument(
        "--task_type",
        type=str,
        default=None,
        choices=["multiclass", "regression"],
        help="Task type (default: from config.TASK_TYPE)",
    )
    parser.add_argument(
        "--data_root", type=str, default="/ARCAFF/data/04_final/data", help="Root directory containing sample folders"
    )
    parser.add_argument("--manifest_path", type=str, default=None, help="Path to pre-built manifest file (optional)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/ARCAFF/ARCCnet/outputs/timeseries",
        help="Directory for outputs (checkpoints, logs, etc.)",
    )

    args = parser.parse_args()
    main(args)
