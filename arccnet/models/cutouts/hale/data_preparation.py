"""
Data preparation pipeline for ARCCNet Hale classification model.

This module provides a complete data preparation pipeline for training the ARCCNet Hale
classification model. It handles dataset loading, cleaning, label mapping, filtering,
and cross-validation fold creation with proper AR number separation.

The pipeline ensures that Active Region (AR) numbers don't overlap between train/validation/test
sets to prevent data leakage, which is crucial for temporal solar data.

Functions:
    load_and_clean_dataset: Load raw dataset and apply basic cleaning
    apply_label_mapping_and_filter: Map Hale classes and apply filtering/undersampling
    create_and_validate_cv_folds: Create CV folds with AR number validation
    prepare_dataset: Complete end-to-end preparation pipeline
    main: CLI entry point for data preparation

Example: python data_preparation.py --n_splits 5 --random_state 42
"""

import logging
import argparse
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import arccnet.models.cutouts.hale.config as config
from arccnet.models import dataset_utils as ut_d

# ==============================================================================
# CONFIGURABLE PARAMETERS
# ==============================================================================

# NaN filtering parameters
DEFAULT_NAN_THRESHOLD = 0.2  # Maximum allowed fraction of NaN values (20%)
SHOW_PROGRESS = True  # Show progress bar during NaN filtering

# Column names for cutout paths (prioritized in order)
CUTOUT_PATH_COLUMNS = [
    "path_image_cutout_hmi",  # Primary: HMI cutout paths
    "path_image_cutout_mdi",  # Secondary: MDI cutout paths
]

# File format configuration
FILE_FORMAT = "fits"  # Options: "fits", "numpy", "hdf5"
FITS_HDU_INDEX = 0  # FITS HDU index for data (usually 0 or 1)

# Logging configuration
ENABLE_NAN_FILTERING_LOG = True
LOG_DETAILED_STATS = True


def filter_by_nan_threshold(
    df: pd.DataFrame,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
    data_folder: Path = None,
    show_progress: bool = SHOW_PROGRESS,
) -> pd.DataFrame:
    """
    Filter out magnetogram cutouts with excessive NaN values.

    Args:
        df: DataFrame with cutout references
        nan_threshold: Maximum allowed fraction of NaN values (0-1)
        data_folder: Base folder for cutout files (defaults to config.DATA_FOLDER)
        show_progress: Show progress bar during filtering

    Returns:
        pd.DataFrame: Filtered dataframe with low-NaN cutouts only
    """
    if data_folder is None:
        data_folder = Path(config.DATA_FOLDER)

    if not ENABLE_NAN_FILTERING_LOG:
        return df

    logging.info(f"Filtering cutouts with NaN threshold: {nan_threshold * 100:.1f}%")

    valid_indices = []
    nan_stats = []

    iterator = tqdm(df.iterrows(), total=len(df), desc="Filtering NaN cutouts") if show_progress else df.iterrows()

    for idx, row in iterator:
        try:
            cutout = _load_cutout(row, data_folder)
            if cutout is not None:
                nan_fraction = np.isnan(cutout).sum() / cutout.size
                nan_stats.append(nan_fraction)
                if nan_fraction <= nan_threshold:
                    valid_indices.append(idx)
            else:
                # Skip if no valid cutout found
                nan_stats.append(1.0)
        except Exception as e:
            logging.warning(f"Error loading cutout at index {idx}: {e}")
            nan_stats.append(1.0)

    df_filtered = df.loc[valid_indices].copy()

    if LOG_DETAILED_STATS and len(nan_stats) > 0:
        nan_stats = np.array(nan_stats)
        logging.info("NaN filtering results:")
        logging.info(f"  Original: {len(df):,} cutouts")
        logging.info(f"  Filtered: {len(df_filtered):,} cutouts ({len(df_filtered) / len(df) * 100:.1f}% retained)")
        logging.info(f"  Removed:  {len(df) - len(df_filtered):,} cutouts")
        logging.info(f"  Mean NaN fraction: {nan_stats.mean() * 100:.2f}%")
        logging.info(f"  Max NaN fraction:  {nan_stats.max() * 100:.2f}%")

    return df_filtered


def _load_cutout(row, data_folder):
    """
    Load a single magnetogram cutout from available path columns.

    Tries cutout path columns in priority order and loads based on file format.

    Args:
        row: DataFrame row containing path information
        data_folder: Base folder for cutout files

    Returns:
        numpy.ndarray or None: Loaded cutout data or None if no valid path found
    """
    # Find the first available path
    file_path = None
    for path_col in CUTOUT_PATH_COLUMNS:
        if path_col in row and row[path_col] and row[path_col] not in ["", "None"]:
            file_path = Path(data_folder) / row[path_col]
            if file_path.exists():
                break

    if file_path is None or not file_path.exists():
        return None

    # Load based on file format
    try:
        if FILE_FORMAT.lower() == "fits":
            from astropy.io import fits

            with fits.open(file_path) as hdul:
                cutout = hdul[FITS_HDU_INDEX].data
        elif FILE_FORMAT.lower() == "numpy":
            cutout = np.load(file_path)
        elif FILE_FORMAT.lower() == "hdf5":
            import h5py

            with h5py.File(file_path, "r") as f:
                # Assume data is in the first dataset - adjust as needed
                cutout = f[list(f.keys())[0]][:]
        else:
            raise ValueError(f"Unsupported file format: {FILE_FORMAT}")

        return cutout

    except Exception as e:
        logging.warning(f"Failed to load cutout from {file_path}: {e}")
        return None


def load_and_clean_dataset(nan_threshold: float = DEFAULT_NAN_THRESHOLD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw solar magnetogram dataset and apply cleaning + NaN filtering.

    Args:
        nan_threshold: Maximum allowed NaN fraction (default from config)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Raw dataset and cleaned dataset
    """
    logging.info("Loading and cleaning dataset...")

    df, _, _ = ut_d.make_dataframe(config.DATA_FOLDER, config.DATASET_FOLDER, config.DF_FILE_NAME)
    logging.info(f"Original DataFrame shape: {df.shape}")

    df_clean = ut_d.cleanup_df(df)
    logging.info(f"After cleanup: {df_clean.shape} ({len(df_clean) / len(df) * 100:.1f}% retained)")

    # Add NaN filtering step
    if ENABLE_NAN_FILTERING_LOG:
        df_clean = filter_by_nan_threshold(df_clean, nan_threshold=nan_threshold, data_folder=Path(config.DATA_FOLDER))

    return df, df_clean


def apply_label_mapping_and_filter(
    df_clean: pd.DataFrame, label_mapping: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply Hale classification label mapping and filtering operations.

    Maps original Hale class labels to grouped categories (e.g., Alpha, Beta, Beta-Gamma)
    and applies filtering based on longitude limits and undersampling configuration.

    Args:
        df_clean: Cleaned dataframe from load_and_clean_dataset()
        label_mapping: Dictionary mapping original labels to grouped categories
                      (e.g., {'Alpha': 'Alpha', 'Beta-Delta': 'Beta'})

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Original and processed dataframes
            - First element: Dataframe after label mapping only
            - Second element: Final processed dataframe after all filtering

    Note:
        Applies longitude filtering and undersampling based on config parameters.
        Logs detailed class distribution statistics.
    """

    df_original, df_processed = ut_d.undersample_group_filter(
        df_clean, label_mapping=label_mapping, long_limit_deg=config.LONG_LIMIT_DEG, undersample=config.UNDERSAMPLE
    )

    logging.info(f"Label mapping: {label_mapping}")
    logging.info(f"Label mapping applied: {len(df_original):,} → {len(df_processed):,}")

    # Log detailed class distribution for verification
    class_dist = df_processed["grouped_labels"].value_counts()
    logging.info("Final class distribution:")
    for label, count in class_dist.items():
        pct = count / len(df_processed) * 100
        logging.info(f"  {label}: {count:,} ({pct:.1f}%)")

    return df_original, df_processed


def create_and_validate_cv_folds(
    df_processed: pd.DataFrame, n_splits: int = config.N_FOLDS, random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """
    Create cross-validation folds with AR number separation validation.

    Creates stratified cross-validation folds ensuring that Active Region (AR) numbers
    don't overlap between train/validation/test sets. This prevents data leakage that
    could occur when the same AR appears in multiple sets.

    Args:
        df_processed: Processed dataframe with grouped labels
        n_splits: Number of CV folds to create (default from config)
        random_state: Random seed for reproducible splits (default from config)

    Returns:
        pd.DataFrame: Input dataframe with added fold assignment columns
                     (e.g., 'Fold 1', 'Fold 2', etc.)

    Raises:
        Logs error if AR number overlaps are detected between sets

    Note:
        Each fold column contains 'train'/'val'/'test' assignments.
        Validation ensures no AR appears in multiple sets within same fold.
    """
    logging.info("Creating cross-validation folds...")

    # Create stratified folds grouped by AR number to prevent data leakage
    ut_d.split_data(
        df_processed,
        label_col="grouped_labels",
        group_col="number",  # Critical: group by AR number
        n_splits=n_splits,
        random_state=random_state,
    )

    # Validate AR number separation to ensure no data leakage
    _validate_fold_separation(df_processed)

    return df_processed


def _validate_fold_separation(df_processed: pd.DataFrame) -> None:
    """
    Validate that AR numbers don't overlap between train/validation/test sets.

    Performs critical validation to ensure no Active Region number appears in
    multiple sets (train/val/test) within the same fold, preventing data leakage.

    Args:
        df_processed: Dataframe with fold assignment columns

    Logs:
        - Set sizes and unique AR counts for each fold
        - Error messages if overlaps are detected
        - Success confirmation if no overlaps found

    Note:
        This validation is essential for temporal solar data where the same AR
        can have multiple observations over time.
    """
    fold_columns = [col for col in df_processed.columns if col.startswith("Fold ")]

    for fold_column_name in fold_columns:
        fold_num = fold_column_name.split()[-1]

        # Get masks and counts for each set
        train_mask, val_mask, test_mask = (
            df_processed[fold_column_name] == split for split in ["train", "val", "test"]
        )
        train_count, val_count, test_count = train_mask.sum(), val_mask.sum(), test_mask.sum()

        # Get unique AR numbers in each set
        train_ars, val_ars, test_ars = (
            set(df_processed[mask]["number"].unique()) for mask in [train_mask, val_mask, test_mask]
        )

        # Check for overlaps
        overlaps = {
            "Train-Val": train_ars & val_ars,
            "Train-Test": train_ars & test_ars,
            "Val-Test": val_ars & test_ars,
        }

        logging.info(f"Fold {fold_num}: Train={train_count:,}, Val={val_count:,}, Test={test_count:,}")
        logging.info(f"  Unique ARs - Train: {len(train_ars)}, Val: {len(val_ars)}, Test: {len(test_ars)}")

        if any(overlaps.values()):
            logging.error(f"  ERROR - AR number overlaps detected in Fold {fold_num}:")
            for overlap_type, overlap_set in overlaps.items():
                if overlap_set:
                    logging.error(f"    {overlap_type} overlap: {sorted(overlap_set)}")
        else:
            logging.info("  ✓ No AR number overlaps on different sets")


def prepare_dataset(
    save_path: str = None,
    n_splits: int = config.N_FOLDS,
    random_state: int = config.RANDOM_STATE,
    label_mapping: dict[str, Any] = None,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
) -> pd.DataFrame:
    """
    Complete end-to-end dataset preparation pipeline with NaN filtering.
    Args:
        save_path: Path to save processed dataset
        n_splits: Number of cross-validation folds
        random_state: Random seed for reproducibility
        label_mapping: Custom label mapping dict
        nan_threshold: Maximum allowed NaN fraction (0-1)
    Returns:
        pd.DataFrame: Fully processed dataset with fold assignments
    """
    # Step 1: Load, clean, and filter NaNs
    df_raw, df_clean = load_and_clean_dataset(nan_threshold=nan_threshold)
    # Rest of the pipeline remains the same...
    if label_mapping is None:
        label_mapping = config.label_mapping
    df_original, df_processed = apply_label_mapping_and_filter(df_clean, label_mapping=label_mapping)
    df_with_folds = create_and_validate_cv_folds(df_processed, n_splits=n_splits, random_state=random_state)
    if save_path:
        df_with_folds.to_parquet(save_path, index=False)
        logging.info(f"Processed dataset saved to: {save_path}")
    return df_with_folds


def main():
    """
    CLI entry point for the data preparation pipeline.

    Provides command-line interface for running the complete dataset preparation
    pipeline with configurable parameters. Automatically generates output filename
    if not specified.

    Command Line Arguments:
        --n_splits: Number of CV folds (default from config)
        --random_state: Random seed for reproducibility (default from config)
        --save_path: Output file path (default: auto-generated based on config)

    Generated Filename Format:
        processed_dataset_{classes}_{n_splits}-splits_rs-{random_state}.parquet

    Example Usage:
        python data_preparation.py --n_splits 10 --random_state 123
        python data_preparation.py --save_path /data/custom_name.parquet
    """
    parser = argparse.ArgumentParser(description="Prepare dataset for ARCCNet model training.")
    parser.add_argument("--n_splits", type=int, default=config.N_FOLDS, help="Number of splits for cross-validation.")
    parser.add_argument(
        "--random_state", type=int, default=config.RANDOM_STATE, help="Random state for reproducibility."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Path to save the processed dataset. If not provided, a default name will be generated.",
    )
    args = parser.parse_args()

    # Configure logging for pipeline execution
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    save_path = args.save_path
    if save_path is None:
        # Generate descriptive filename based on configuration
        save_path = (
            Path(config.DATA_FOLDER)
            / f"processed_dataset_{config.classes}_{args.n_splits}-splits_rs-{args.random_state}.parquet"
        )

    # Execute the complete data preparation pipeline
    prepare_dataset(
        save_path=save_path, n_splits=args.n_splits, random_state=args.random_state, label_mapping=config.label_mapping
    )


if __name__ == "__main__":
    main()
