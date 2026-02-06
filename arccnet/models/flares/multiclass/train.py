# Import comet_ml first to resolve logging warnings
try:
    from pytorch_lightning.loggers import CometLogger
except ImportError:
    CometLogger = None

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from arccnet.models.dataset_utils import split_data
from arccnet.models.flares.multiclass import config
from arccnet.models.flares.multiclass.datamodule import FlareDataModule
from arccnet.models.flares.multiclass.model import FlareClassifier
from arccnet.utils.logging import get_logger

logger = get_logger(__name__)


def get_strongest_flare_class(row):
    """
    Determines the strongest flare class for a given row based on flare counts.
    First filters out A and B class flares, then applies precedence: M_X > C > Quiet.
    M and X classes are merged into M_X for better classification performance.
    """
    # Check for A or B class flares first - these will be filtered out regardless of other flares
    if ("A" in row and row["A"] > 0) or ("B" in row and row["B"] > 0):
        return "Filter_out"  # Mark for removal

    # Apply precedence for remaining classes (M and X are merged)
    if row["X"] > 0 or row["M"] > 0:
        return "M_X"
    if row["C"] > 0:
        return "C"
    return "Quiet"  # Default to Quiet for regions with no C, M, or X flares


def preprocess_flare_data():
    """
    Load and preprocess the raw flare dataset on the fly.
    Merges M and X classes into M_X for better classification performance.
    """
    logger.info("Loading and preprocessing raw flare data...")

    # Load raw flare dataset
    input_flares_path = os.path.join(config.DATA_FOLDER, config.FLARES_PARQ)
    logger.info(f"Loading flare data from {input_flares_path}")

    if not os.path.exists(input_flares_path):
        logger.error(f"Input file not found: {input_flares_path}")
        raise FileNotFoundError(f"Input file not found: {input_flares_path}")

    df = pd.read_parquet(input_flares_path)
    logger.info(f"Loaded {len(df)} flare events.")

    logger.info(f"Classifying events into {config.CLASSES} based on strongest flare (M and X merged)...")
    # Fill NaN values with 0 in flare count columns
    flare_columns = ["A", "B", "C", "M", "X"]
    available_columns = [col for col in flare_columns if col in df.columns]
    df[available_columns] = df[available_columns].fillna(0)

    df[config.TARGET_COLUMN] = df.apply(get_strongest_flare_class, axis=1)

    # Filter out A and B class flares
    initial_count = len(df)
    df_filtered = df[df[config.TARGET_COLUMN] != "Filter_out"].copy()
    filtered_count = len(df_filtered)
    logger.info(f"Filtered out {initial_count - filtered_count} events with A or B class flares.")
    logger.info(f"Remaining events: {filtered_count}")

    logger.info("Class distribution (M and X merged):")
    logger.info(f"\n{df_filtered[config.TARGET_COLUMN].value_counts()}")

    return df_filtered


def filter_solar_limb(df):
    """
    Filter out observations too close to the solar limb based on longitude/latitude.
    Uses longitude_hmi/longitude_mdi columns based on which instrument has data available.

    Args:
        df (pd.DataFrame): DataFrame with solar observations

    Returns:
        pd.DataFrame: Filtered DataFrame with limb observations removed
    """
    if not config.FILTER_SOLAR_LIMB:
        logger.info("Solar limb filtering disabled.")
        return df

    initial_count = len(df)
    logger.info(f"Applying solar limb filtering (max longitude: {config.MAX_LONGITUDE}°)...")

    # Check for instrument-specific longitude columns (HMI/MDI)
    if "longitude_hmi" in df.columns and "longitude_mdi" in df.columns:
        # Use HMI if available, otherwise MDI
        # Check which instrument has image data available for each row
        logger.info("Using instrument-specific longitude columns (longitude_hmi/longitude_mdi)")

        # Create a combined longitude column based on available instrument data
        # Default to NaN, then fill based on available paths
        df["limb_filter_longitude"] = np.nan

        # For rows with HMI data available, use longitude_hmi
        hmi_mask = pd.notna(df.get("path_image_cutout_hmi", pd.Series(dtype=object))) & (
            df.get("path_image_cutout_hmi", "") != ""
        )
        if hmi_mask.any():
            df.loc[hmi_mask, "limb_filter_longitude"] = df.loc[hmi_mask, "longitude_hmi"]
            logger.info(f"Using HMI longitude for {hmi_mask.sum()} observations")

        # For rows with MDI data (and no HMI), use longitude_mdi
        mdi_mask = (
            pd.notna(df.get("path_image_cutout_mdi", pd.Series(dtype=object)))
            & (df.get("path_image_cutout_mdi", "") != "")
        ) & ~hmi_mask
        if mdi_mask.any():
            df.loc[mdi_mask, "limb_filter_longitude"] = df.loc[mdi_mask, "longitude_mdi"]
            logger.info(f"Using MDI longitude for {mdi_mask.sum()} observations")

        lon_col = "limb_filter_longitude"
        lat_col = None  # Latitude filtering not implemented

    # Apply longitude filtering
    if config.MAX_LONGITUDE is not None:
        # Remove rows with NaN longitude values first
        valid_lon_mask = pd.notna(df[lon_col])
        if not valid_lon_mask.all():
            invalid_count = (~valid_lon_mask).sum()
            logger.info(f"Filtering out {invalid_count} observations with missing longitude data")
            df_filtered = df[valid_lon_mask].copy()
        else:
            df_filtered = df.copy()

        # Apply longitude threshold filtering
        mask = np.abs(df_filtered[lon_col]) <= config.MAX_LONGITUDE
        df_filtered = df_filtered[mask].copy()
        lon_filtered = len(df) - len(df_filtered)
        logger.info(f"Filtered out {lon_filtered} observations with |longitude| > {config.MAX_LONGITUDE}°")
    else:
        df_filtered = df.copy()
        lon_filtered = 0

    # Apply latitude filtering if specified
    if config.MAX_LATITUDE is not None and lat_col is not None:
        initial_lat_count = len(df_filtered)
        mask = np.abs(df_filtered[lat_col]) <= config.MAX_LATITUDE
        df_filtered = df_filtered[mask].copy()
        lat_filtered = initial_lat_count - len(df_filtered)
        logger.info(f"Filtered out {lat_filtered} observations with |latitude| > {config.MAX_LATITUDE}°")

    # Clean up temporary columns
    temp_columns = ["approx_lon", "approx_lat", "limb_filter_longitude"]
    for temp_col in temp_columns:
        if temp_col in df_filtered.columns:
            df_filtered = df_filtered.drop(columns=[temp_col])

    total_filtered = initial_count - len(df_filtered)
    percentage_filtered = (total_filtered / initial_count) * 100
    logger.info(f"Total limb filtering: {total_filtered} observations removed ({percentage_filtered:.1f}%)")
    logger.info(f"Remaining observations: {len(df_filtered)}")

    return df_filtered


def log_dataset_histograms(train_df, val_df, test_df, class_names, comet_logger):
    """
    Log histograms of class distributions for train, validation, and test sets to Comet.
    """
    if not comet_logger or not hasattr(comet_logger, "experiment"):
        logger.warning("Comet logger not available, skipping histogram logging.")
        return

    logger.info("Logging dataset histograms to Comet...")

    # Get class distributions
    train_counts = train_df[config.TARGET_COLUMN].value_counts()
    val_counts = val_df[config.TARGET_COLUMN].value_counts()
    test_counts = test_df[config.TARGET_COLUMN].value_counts()

    # Ensure all classes are represented (fill missing with 0)
    for class_idx, class_name in enumerate(class_names):
        train_count = train_counts.get(class_idx, 0)
        val_count = val_counts.get(class_idx, 0)
        test_count = test_counts.get(class_idx, 0)

        # Log individual counts
        comet_logger.experiment.log_metric(f"dataset_count_train_{class_name}", train_count)
        comet_logger.experiment.log_metric(f"dataset_count_val_{class_name}", val_count)
        comet_logger.experiment.log_metric(f"dataset_count_test_{class_name}", test_count)

        # Log percentages
        train_pct = (train_count / len(train_df)) * 100 if len(train_df) > 0 else 0
        val_pct = (val_count / len(val_df)) * 100 if len(val_df) > 0 else 0
        test_pct = (test_count / len(test_df)) * 100 if len(test_df) > 0 else 0

        comet_logger.experiment.log_metric(f"dataset_pct_train_{class_name}", train_pct)
        comet_logger.experiment.log_metric(f"dataset_pct_val_{class_name}", val_pct)
        comet_logger.experiment.log_metric(f"dataset_pct_test_{class_name}", test_pct)

    # Create comparative histogram
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Prepare data for plotting
    class_indices = list(range(len(class_names)))
    train_values = [train_counts.get(i, 0) for i in class_indices]
    val_values = [val_counts.get(i, 0) for i in class_indices]
    test_values = [test_counts.get(i, 0) for i in class_indices]

    # Train set histogram
    ax1.bar(class_names, train_values, color="skyblue", alpha=0.7)
    ax1.set_title(f"Training Set Distribution (n={len(train_df)})")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)

    # Validation set histogram
    ax2.bar(class_names, val_values, color="lightgreen", alpha=0.7)
    ax2.set_title(f"Validation Set Distribution (n={len(val_df)})")
    ax2.set_ylabel("Count")
    ax2.tick_params(axis="x", rotation=45)

    # Test set histogram
    ax3.bar(class_names, test_values, color="lightcoral", alpha=0.7)
    ax3.set_title(f"Test Set Distribution (n={len(test_df)})")
    ax3.set_ylabel("Count")
    ax3.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Log the figure to Comet
    comet_logger.experiment.log_figure("dataset_distributions", fig)
    plt.close()

    # Create a combined comparison chart
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(class_names))
    width = 0.25

    ax.bar(x - width, train_values, width, label="Train", color="skyblue", alpha=0.7)
    ax.bar(x, val_values, width, label="Validation", color="lightgreen", alpha=0.7)
    ax.bar(x + width, test_values, width, label="Test", color="lightcoral", alpha=0.7)

    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_title("Dataset Distribution Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_yscale("log")  # Use log scale due to class imbalance

    plt.tight_layout()
    comet_logger.experiment.log_figure("dataset_comparison", fig)
    plt.close()

    logger.info("Dataset histograms logged to Comet successfully.")


def create_transforms():
    """Create training and validation transforms based on config."""
    if config.USE_AUGMENTATION:
        train_transform = T.Compose(
            [
                T.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
                T.RandomVerticalFlip(p=config.VERTICAL_FLIP_PROB),
                T.RandomRotation(degrees=config.ROTATION_DEGREES),
            ]
        )
    else:
        train_transform = T.Compose([])

    # No augmentation for validation/test
    val_test_transform = T.Compose([])

    return train_transform, val_test_transform


def train():
    """
    Main training routine for the multi-class flare classifier.
    """
    torch.set_float32_matmul_precision("high")
    pl.seed_everything(config.RANDOM_SEED, workers=True)

    # --- Data Preparation ---
    logger.info("Starting data preparation...")

    # Preprocess the raw flare data on the fly
    df = preprocess_flare_data()

    # Apply solar limb filtering
    df = filter_solar_limb(df)

    # 1. Filter out rows with missing image paths
    path_cols = ["path_image_cutout_hmi", "path_image_cutout_mdi"]
    initial_count = len(df)
    df.dropna(subset=path_cols, how="all", inplace=True)
    logger.info(f"Filtered {initial_count - len(df)} records with no image path.")

    # 2. Verify file existence
    logger.info("Verifying image file existence...")
    initial_count = len(df)
    base_path = os.path.join(config.DATA_FOLDER, config.CUTOUT_DATASET_FOLDER, "fits")

    def get_path(row):
        p = row.get("path_image_cutout_hmi")
        return p if pd.notna(p) else row.get("path_image_cutout_mdi")

    df["temp_path"] = df.apply(get_path, axis=1)
    tqdm.pandas(desc="Verifying files")
    exists_mask = df["temp_path"].progress_apply(lambda p: os.path.exists(os.path.join(base_path, os.path.basename(p))))
    df = df[exists_mask].drop(columns=["temp_path"])
    logger.info(f"Filtered {initial_count - len(df)} records with missing FITS files.")

    # 3. Encode labels
    label_encoder = LabelEncoder()
    df[config.TARGET_COLUMN] = label_encoder.fit_transform(df[config.TARGET_COLUMN])
    class_names = list(label_encoder.classes_)
    num_classes = len(class_names)
    logger.info(f"Encoded labels. Found classes: {class_names}")

    # 4. Split data
    df_with_folds, _ = split_data(df, label_col=config.TARGET_COLUMN, group_col="number")
    train_df = df_with_folds[df_with_folds["Fold 1"] == "train"]
    val_df = df_with_folds[df_with_folds["Fold 1"] == "val"]
    test_df = df_with_folds[df_with_folds["Fold 1"] == "test"]
    logger.info(f"Data split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # 5. Calculate class weights if enabled
    class_weights = None
    if config.USE_WEIGHTED_LOSS:
        logger.info("Calculating class weights for weighted loss...")
        # Use training data for class weight calculation
        train_labels = train_df[config.TARGET_COLUMN].values
        class_weights_values = compute_class_weight("balanced", classes=np.unique(train_labels), y=train_labels)
        class_weights = torch.tensor(class_weights_values, dtype=torch.float32)
        logger.info(f"Class names: {class_names}")
        logger.info(f"Class weights: {class_weights.tolist()}")
    else:
        logger.info("Using unweighted loss.")

    # --- End Data Preparation ---

    # Create transforms
    train_transform, val_test_transform = create_transforms()
    logger.info(f"Augmentation enabled: {config.USE_AUGMENTATION}")

    logger.info("Setting up data module...")
    data_module = FlareDataModule(
        data_folder=config.DATA_FOLDER,
        dataset_folder=config.CUTOUT_DATASET_FOLDER,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        target_column=config.TARGET_COLUMN,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        img_target_height=config.IMG_TARGET_HEIGHT,
        img_target_width=config.IMG_TARGET_WIDTH,
        img_divisor=config.IMG_DIVISOR,
        img_min_val=config.IMG_MIN_VAL,
        img_max_val=config.IMG_MAX_VAL,
        train_transform=train_transform,
        val_test_transform=val_test_transform,
    )

    logger.info(f"Initializing model: {config.MODEL_NAME}")
    logger.info(f"Using loss function: {config.LOSS_TYPE}")
    if config.LOSS_TYPE in ["focal", "weighted_focal"]:
        logger.info(f"Focal loss parameters - Alpha: {config.FOCAL_ALPHA}, Gamma: {config.FOCAL_GAMMA}")
    model = FlareClassifier(num_classes=num_classes, class_names=class_names, class_weights=class_weights)

    comet_logger = None
    if config.ENABLE_COMET_LOGGING and CometLogger:
        logger.info("Setting up Comet.ml logger...")
        comet_logger = CometLogger(
            api_key=os.environ.get("COMET_API_KEY"),
            workspace=config.COMET_WORKSPACE,
            project_name=config.COMET_PROJECT_NAME,
        )
        comet_logger.log_hyperparams(vars(config))
        comet_logger.log_hyperparams({"class_names": class_names})
        comet_logger.log_hyperparams(
            {
                "use_augmentation": config.USE_AUGMENTATION,
                "horizontal_flip_prob": config.HORIZONTAL_FLIP_PROB if config.USE_AUGMENTATION else 0.0,
                "vertical_flip_prob": config.VERTICAL_FLIP_PROB if config.USE_AUGMENTATION else 0.0,
                "rotation_degrees": config.ROTATION_DEGREES if config.USE_AUGMENTATION else 0.0,
                "loss_type": config.LOSS_TYPE,
                "focal_alpha": config.FOCAL_ALPHA if hasattr(config, "FOCAL_ALPHA") else None,
                "focal_gamma": config.FOCAL_GAMMA if hasattr(config, "FOCAL_GAMMA") else None,
                "log_confusion_matrix": config.LOG_CONFUSION_MATRIX,
                "log_misclassified_examples": config.LOG_MISCLASSIFIED_EXAMPLES,
                "log_classification_report": config.LOG_CLASSIFICATION_REPORT,
                "max_misclassified_examples": config.MAX_MISCLASSIFIED_EXAMPLES,
            }
        )

        # Log dataset histograms
        log_dataset_histograms(train_df, val_df, test_df, class_names, comet_logger)

    logger.info("Setting up callbacks...")
    checkpoint_callback = ModelCheckpoint(
        monitor=config.CHECKPOINT_METRIC,
        dirpath="trained_models/multiclass/",
        filename=f"{config.MODEL_NAME}-{{epoch:02d}}-{{{config.CHECKPOINT_METRIC}:.3f}}",
        save_top_k=1,
        mode="max",
        auto_insert_metric_name=False,
    )
    early_stop_callback = EarlyStopping(
        monitor=config.CHECKPOINT_METRIC, patience=config.PATIENCE, verbose=True, mode="max"
    )

    logger.info("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator="auto",
        devices=config.DEVICES,
        precision="16-mixed",
        logger=comet_logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        deterministic=True,
    )

    logger.info("Starting training...")
    trainer.fit(model, datamodule=data_module)

    logger.info("Starting testing on the best model...")
    # Use the actual best checkpoint path from the callback
    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path and os.path.exists(best_ckpt_path):
        logger.info(f"Loading best checkpoint: {best_ckpt_path}")
        trainer.test(model, datamodule=data_module, ckpt_path=best_ckpt_path)
    else:
        logger.warning("No valid checkpoint found, testing with current model weights")
        trainer.test(model, datamodule=data_module)

    logger.info("Training and testing complete.")


if __name__ == "__main__":
    train()
