"""Training script for timeseries flare forecasting using PyTorch Lightning."""

import os
import sys
import json
import logging
import argparse
from pathlib import Path

import numpy as np
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
    import arccnet.models.timeseries.config as ts_config
    from arccnet.models.timeseries.data_module import FlareDataModule
    from arccnet.models.timeseries.dataset import SDOTimeseriesDataset
    from arccnet.models.timeseries.lightning_module import FlareForecasterLightning
    from arccnet.models.timeseries.manifest import build_dataset
    from arccnet.models.timeseries.splitters import get_split
else:
    from . import config as ts_config
    from .data_module import FlareDataModule
    from .dataset import SDOTimeseriesDataset
    from .lightning_module import FlareForecasterLightning
    from .manifest import build_dataset
    from .splitters import get_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SEED = ts_config.SEED
RESIZE = ts_config.RESIZE
NUM_CHANNELS = ts_config.NUM_CHANNELS
NUM_TIMESTEPS = ts_config.NUM_TIMESTEPS
GPU_ID = ts_config.GPU_ID
TASK_TYPE = ts_config.TASK_TYPE
SPLIT_STRATEGY = ts_config.SPLIT_STRATEGY
TRAIN_FRAC = ts_config.TRAIN_FRAC
VAL_FRAC = ts_config.VAL_FRAC
TRAIN_YEARS = ts_config.TRAIN_YEARS
VAL_YEARS = ts_config.VAL_YEARS
TEST_YEARS = ts_config.TEST_YEARS
NUM_CLASSES = ts_config.NUM_CLASSES
FLARE_CLASS_NAMES = list(getattr(ts_config, "FLARE_CLASS_NAMES", ["No-flare", "C", "M+"]))
BATCH_SIZE = ts_config.BATCH_SIZE
NUM_WORKERS = ts_config.NUM_WORKERS
USE_AUGMENTATION = ts_config.USE_AUGMENTATION
HFLIP_PROB = ts_config.HFLIP_PROB
VFLIP_PROB = ts_config.VFLIP_PROB
ROTATION_DEGREES = ts_config.ROTATION_DEGREES
REGRESSION_TARGETS = ts_config.REGRESSION_TARGETS
LEARNING_RATE = ts_config.LEARNING_RATE
WEIGHT_DECAY = ts_config.WEIGHT_DECAY
SPATIAL_FEATURE_DIM = ts_config.SPATIAL_FEATURE_DIM
TEMPORAL_NUM_LAYERS = ts_config.TEMPORAL_NUM_LAYERS
TEMPORAL_NUM_HEADS = ts_config.TEMPORAL_NUM_HEADS
TEMPORAL_DIM_FEEDFORWARD = ts_config.TEMPORAL_DIM_FEEDFORWARD
TEMPORAL_DROPOUT = ts_config.TEMPORAL_DROPOUT
TEMPORAL_POOLING = ts_config.TEMPORAL_POOLING
PRETRAINED_SPATIAL = ts_config.PRETRAINED_SPATIAL
FREEZE_SPATIAL = ts_config.FREEZE_SPATIAL
HIDDEN_DIMS = ts_config.HIDDEN_DIMS
DROPOUT = ts_config.DROPOUT
EARLY_STOPPING_PATIENCE = ts_config.EARLY_STOPPING_PATIENCE
FIND_LR = ts_config.FIND_LR
LR_FIND_MIN = ts_config.LR_FIND_MIN
LR_FIND_MAX = ts_config.LR_FIND_MAX
LR_FIND_NUM_STEPS = ts_config.LR_FIND_NUM_STEPS
ACCELERATOR = ts_config.ACCELERATOR
MAX_EPOCHS = ts_config.MAX_EPOCHS
DEVICES = ts_config.DEVICES
GRAD_CLIP_MAX_NORM = ts_config.GRAD_CLIP_MAX_NORM
LOG_EVERY_N_STEPS = ts_config.LOG_EVERY_N_STEPS
TIMESERIES_ROOT = ts_config.TIMESERIES_ROOT
MANIFEST_PATH = ts_config.MANIFEST_PATH
LOSS_FUNCTION = ts_config.LOSS_FUNCTION
FOCAL_LOSS_ALPHA = ts_config.FOCAL_LOSS_ALPHA
FOCAL_LOSS_GAMMA = ts_config.FOCAL_LOSS_GAMMA
PRECISION = getattr(ts_config, "PRECISION", "auto")
SAFE_GPU_MODE = bool(getattr(ts_config, "SAFE_GPU_MODE", True))

VALID_PRECISIONS = ("auto", "16-mixed", "bf16-mixed", "32-true", "64-true")


def _resolve_data_root(data_root):
    """
    Resolve data root across legacy and namespaced layouts.

    Supports:
    - /ARCAFF/data/04_final/data
    - /ARCAFF/data/timeseries/04_final/data
    """
    requested = Path(data_root)
    if requested.exists():
        return requested

    candidates = []
    requested_str = str(requested)

    if "/timeseries/04_final/data" in requested_str:
        candidates.append(Path(requested_str.replace("/timeseries/04_final/data", "/04_final/data")))
    elif requested_str.endswith("/04_final/data"):
        candidates.append(Path(requested_str.replace("/04_final/data", "/timeseries/04_final/data")))

    candidates.extend(
        [
            Path(ts_config.TIMESERIES_ROOT),
            Path("/ARCAFF/data/timeseries/04_final/data"),
            Path("/ARCAFF/data/04_final/data"),
        ]
    )

    seen = set()
    for candidate in candidates:
        candidate_resolved = str(candidate)
        if candidate_resolved in seen:
            continue
        seen.add(candidate_resolved)
        if candidate.exists():
            logger.warning(f"Data root not found at {requested}; using existing path {candidate} instead.")
            return candidate

    raise FileNotFoundError(
        f"Data root not found: {requested}. Checked fallbacks: {', '.join(str(c) for c in candidates)}"
    )


def _preflight_data_availability(manifest_df, task_type, sample_count=32):
    """
    Check whether referenced FITS files are physically available before training.

    This catches cases where 04_final cutout files are broken symlinks to missing
    03_processed archives and avoids multi-worker error spam during training.
    """
    if len(manifest_df) == 0:
        raise RuntimeError("Manifest is empty after dataset build/load.")

    probe = manifest_df.sample(min(sample_count, len(manifest_df)), random_state=SEED)
    dataset_probe = SDOTimeseriesDataset(
        probe.reset_index(drop=True),
        split="test",
        task_type=task_type,
        resize=RESIZE,
        augment=False,
        norm_stats={"mean": [0.0] * NUM_CHANNELS, "std": [1.0] * NUM_CHANNELS},
    )

    checked = 0
    missing = 0
    for _, row in probe.iterrows():
        paths = dataset_probe._parse_paths(row["paths"])
        for t_paths in paths[:NUM_TIMESTEPS]:
            for c_path in t_paths[:NUM_CHANNELS]:
                if c_path is None or c_path == "None":
                    continue
                checked += 1
                resolved = dataset_probe._resolve_existing_path(c_path)
                if not Path(resolved).exists():
                    missing += 1

    if checked == 0:
        raise RuntimeError("Preflight could not find any FITS paths to validate.")

    missing_ratio = missing / checked
    logger.info(f"Data preflight: checked={checked}, missing={missing}, missing_ratio={missing_ratio:.3f}")

    if missing_ratio >= 0.2:
        raise RuntimeError(
            "Data preflight failed: too many missing FITS files.\n"
            f"Checked {checked} paths, {missing} missing ({missing_ratio:.1%}).\n"
            "Likely cause: broken symlinks from 04_final/data/* to missing 03_processed files.\n"
            "If your processed archive exists elsewhere, set:\n"
            "  export ARCAFF_TIMESERIES_PROCESSED_ROOT=/absolute/path/to/03_processed\n"
            "or restore/create the expected path:\n"
            "  /ARCAFF/data/timeseries/03_processed"
        )


def _resolve_precision(requested_precision, trainer_accelerator):
    """
    Resolve trainer precision with a simple device-aware default.
    """
    precision = str(requested_precision).strip().lower()
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"Invalid precision '{requested_precision}'. Expected one of: {', '.join(VALID_PRECISIONS)}")

    use_cuda = torch.cuda.is_available() and str(trainer_accelerator).lower() != "cpu"

    if precision == "auto":
        return "16-mixed" if use_cuda else "32-true"

    return precision


def _looks_like_worker_permission_error(exc):
    """Detect restricted multiprocessing environments (e.g., SemLock permission denied)."""
    msg = str(exc)
    return "[Errno 13]" in msg and "Permission denied" in msg


def main(args):
    """Main training function using PyTorch Lightning."""

    # Set GPU device from config
    if GPU_ID is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
        logger.info(f"Using GPU {GPU_ID}")

    # Set random seed for reproducibility without enforcing deterministic kernels.
    train_utils.set_global_seed(SEED, deterministic=False)
    logger.info(f"Random seed set to {SEED} (deterministic kernels disabled)")

    # Task type
    task_type = args.task_type if args.task_type else TASK_TYPE
    logger.info(f"Task type: {task_type}")

    # Setup checkpoint manager early so run artifacts can default to checkpoint directory.
    loss_fn = LOSS_FUNCTION if task_type == "multiclass" else "mse"
    checkpoint_mgr = CheckpointManager(
        root_name=f"timeseries/{task_type}",
        data_folder="/ARCAFF/data",
        model_name="resnet34_transformer",
        loss_function=loss_fn,
    )

    # Build dataset manifest for this run
    data_root = _resolve_data_root(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_mgr.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using data root: {data_root}")
    logger.info(f"Run artifacts directory: {output_dir}")

    manifest_path = Path(args.manifest_path)
    if manifest_path.exists():
        logger.info(f"Manifest output already exists and will be overwritten: {manifest_path}")
    logger.info(f"Building dataset manifest from {data_root}")
    manifest_df = build_dataset(data_root, output_path=manifest_path)

    logger.info(f"Total samples: {len(manifest_df)}")
    _preflight_data_availability(manifest_df, task_type=task_type)

    # Split dataset
    split_strategy = SPLIT_STRATEGY
    split_kwargs = {"seed": SEED}
    if split_strategy == "noaa":
        split_kwargs.update({"train_frac": TRAIN_FRAC, "val_frac": VAL_FRAC})
    elif split_strategy == "time":
        split_kwargs.update({"train_years": TRAIN_YEARS, "val_years": VAL_YEARS, "test_years": TEST_YEARS})
    else:
        raise ValueError(f"Unsupported split strategy in config: {split_strategy}")

    split_data = get_split(manifest_df, strategy=split_strategy, **split_kwargs)
    train_mask = split_data["train_mask"]
    val_mask = split_data["val_mask"]
    test_mask = split_data["test_mask"]

    logger.info(f"{split_strategy.upper()} split:")
    logger.info(f"  Train: {train_mask.sum()} samples from {manifest_df[train_mask]['noaa_ar'].nunique()} ARs")
    logger.info(f"  Val:   {val_mask.sum()} samples from {manifest_df[val_mask]['noaa_ar'].nunique()} ARs")
    logger.info(f"  Test:  {test_mask.sum()} samples from {manifest_df[test_mask]['noaa_ar'].nunique()} ARs")

    if split_strategy == "noaa":
        train_noaa = set(manifest_df[train_mask]["noaa_ar"].unique().tolist())
        val_noaa = set(manifest_df[val_mask]["noaa_ar"].unique().tolist())
        test_noaa = set(manifest_df[test_mask]["noaa_ar"].unique().tolist())
        assert train_noaa.isdisjoint(val_noaa), "Leakage detected: train/val NOAA overlap"
        assert train_noaa.isdisjoint(test_noaa), "Leakage detected: train/test NOAA overlap"
        assert val_noaa.isdisjoint(test_noaa), "Leakage detected: val/test NOAA overlap"

    split_assignments = manifest_df[["sample_id", "noaa_ar", "date"]].copy()
    split_assignments["split"] = "unassigned"
    split_assignments.loc[train_mask, "split"] = "train"
    split_assignments.loc[val_mask, "split"] = "val"
    split_assignments.loc[test_mask, "split"] = "test"
    split_assignments_path = output_dir / "split_assignments.parquet"
    split_assignments.to_parquet(split_assignments_path, index=False)
    logger.info(f"Saved split assignments to {split_assignments_path}")

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
            class_name = FLARE_CLASS_NAMES[cls] if cls < len(FLARE_CLASS_NAMES) else f"class_{cls}"
            logger.info(f"  Class {cls} ({class_name}): {count} samples ({pct:.1f}%), weight: {weight:.3f}")

    # Compute normalization stats from training set
    logger.info("Computing normalization stats from training set...")
    train_dataset_temp = SDOTimeseriesDataset(
        manifest_df[train_mask].reset_index(drop=True),
        split="train",
        norm_stats=None,
        task_type=task_type,
        resize=RESIZE,
        augment=False,
    )
    norm_stats = train_dataset_temp.get_norm_stats()

    # Save normalization stats
    norm_stats_path = output_dir / "norm_stats.json"
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    logger.info(f"Normalization stats saved to {norm_stats_path}")

    effective_num_workers = args.num_workers if args.num_workers is not None else NUM_WORKERS

    def _build_datamodule(num_workers):
        return FlareDataModule(
            manifest_df=manifest_df,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            data_dir=data_root,
            norm_stats=norm_stats,
            task_type=task_type,
            batch_size=BATCH_SIZE,
            num_workers=num_workers,
            resize=RESIZE,
            use_augmentation=USE_AUGMENTATION,
            hflip_prob=HFLIP_PROB,
            vflip_prob=VFLIP_PROB,
            rotation_degrees=ROTATION_DEGREES,
        )

    datamodule = _build_datamodule(effective_num_workers)

    # Create Lightning model
    model = FlareForecasterLightning(
        task_type=task_type,
        num_channels=NUM_CHANNELS,
        output_dim=NUM_CLASSES if task_type == "multiclass" else REGRESSION_TARGETS,
        flare_class_names=FLARE_CLASS_NAMES if task_type == "multiclass" else None,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        class_weights=class_weights_computed if task_type == "multiclass" else None,
        loss_function=LOSS_FUNCTION if task_type == "multiclass" else "mse",
        focal_alpha=FOCAL_LOSS_ALPHA,
        focal_gamma=FOCAL_LOSS_GAMMA,
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

    # Callbacks
    checkpoint_callback = checkpoint_mgr.get_checkpoint_callback(
        monitor="val/primary_metric",
        mode="max",
    )

    callbacks = [
        checkpoint_callback,
        EarlyStopping(
            monitor="val/primary_metric",
            patience=EARLY_STOPPING_PATIENCE,
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

    trainer_accelerator = ACCELERATOR
    if trainer_accelerator == "gpu" and not torch.cuda.is_available():
        logger.warning("ACCELERATOR set to 'gpu' but CUDA not available; falling back to CPU")
        trainer_accelerator = "cpu"

    requested_precision = args.precision if args.precision is not None else PRECISION
    trainer_precision = _resolve_precision(requested_precision, trainer_accelerator)

    if trainer_accelerator != "cpu" and torch.cuda.is_available() and SAFE_GPU_MODE:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
        if trainer_precision in {"16-mixed", "bf16-mixed"}:
            logger.warning(
                "SAFE_GPU_MODE is enabled: forcing precision=32-true and disabling cuDNN to avoid CUDA engine errors."
            )
            trainer_precision = "32-true"
        else:
            logger.warning("SAFE_GPU_MODE is enabled: cuDNN disabled for training stability.")

    logger.info(
        "Trainer runtime: accelerator=%s, devices=%s, precision=%s, num_workers=%s, safe_gpu_mode=%s",
        trainer_accelerator,
        DEVICES,
        trainer_precision,
        effective_num_workers,
        SAFE_GPU_MODE,
    )

    def _build_trainer():
        return pl.Trainer(
            max_epochs=MAX_EPOCHS,
            callbacks=callbacks,
            logger=tb_logger,
            accelerator=trainer_accelerator,
            devices=DEVICES,
            precision=trainer_precision,
            gradient_clip_val=GRAD_CLIP_MAX_NORM,
            log_every_n_steps=LOG_EVERY_N_STEPS,
            deterministic=False,
            benchmark=True,
        )

    trainer = _build_trainer()

    # Train
    logger.info("Starting training...")
    try:
        trainer.fit(model, datamodule)
    except PermissionError as exc:
        if effective_num_workers > 0 and _looks_like_worker_permission_error(exc):
            logger.warning(
                "DataLoader multiprocessing is not available in this environment; retrying with num_workers=0."
            )
            effective_num_workers = 0
            datamodule = _build_datamodule(effective_num_workers)
            trainer = _build_trainer()
            trainer.fit(model, datamodule)
        else:
            raise

    # Test on best model
    logger.info("Evaluating best model on test set...")
    trainer.test(model, datamodule, ckpt_path="best")

    best_checkpoint_path = checkpoint_callback.best_model_path or str(checkpoint_mgr.checkpoint_dir / "best.ckpt")

    # Save final results summary
    results = {
        "task_type": task_type,
        "split_strategy": split_strategy,
        "manifest_path": str(manifest_path),
        "norm_stats_path": str(norm_stats_path),
        "split_assignments_path": str(split_assignments_path),
        "checkpoint_dir": str(checkpoint_mgr.checkpoint_dir),
        "best_checkpoint": str(best_checkpoint_path),
    }

    results_path = output_dir / "training_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Training complete! Results saved to {results_path}")
    logger.info(f"Best checkpoint: {best_checkpoint_path}")


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
        "--data_root",
        type=str,
        default=TIMESERIES_ROOT,
        help="Root directory containing sample folders",
    )
    parser.add_argument(
        "--manifest_path",
        type=str,
        default=MANIFEST_PATH,
        help="Path where train.py writes the manifest built from --data_root for this run",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for run artifacts (split assignments, norm stats, logs, summary). "
            "Defaults to the checkpoint run folder under /ARCAFF/data/checkpoints/timeseries."
        ),
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=VALID_PRECISIONS,
        help=(
            "Lightning precision mode. "
            "Use 'auto' to choose based on device availability (default from config.PRECISION)."
        ),
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override DataLoader worker count (defaults to config.NUM_WORKERS).",
    )

    args = parser.parse_args()
    main(args)
