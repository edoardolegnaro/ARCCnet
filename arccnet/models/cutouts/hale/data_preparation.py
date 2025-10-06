# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: ARCAFF
#     language: python
#     name: python3
# ---

# %%
"""
Data preparation for ARCCNet Hale classification:
load, clean, map labels, filter, and create cross-validation folds
with AR number separation.

Example:
    python data_preparation.py --n_splits 5 --random_state 42
"""

# %%
import logging
import argparse
from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from p_tqdm import p_map

import arccnet.models.cutouts.hale.config as config
import arccnet.models.cutouts.hale.dataset as ds
from arccnet.models import dataset_utils as ut_d

# %%
# NaN filtering parameters
DEFAULT_NAN_THRESHOLD = 0.1  # Maximum allowed fraction of NaN values (10%)
ENABLE_NAN_FILTERING_LOG = True
LOG_DETAILED_STATS = True

# %%
df, _, _ = ut_d.make_dataframe("/home/edoardo/Code/ARCAFF/data", config.DATASET_FOLDER, config.DF_FILE_NAME)
logging.info(f"Original DataFrame shape: {df.shape}")
df_clean = ut_d.cleanup_df(df)
logging.info(f"After cleanup: {df_clean.shape} ({len(df_clean) / len(df) * 100:.1f}% retained)")


# %%
def filter_by_nan_threshold(
    df: pd.DataFrame,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
) -> pd.DataFrame:
    """
    Filter out magnetogram cutouts with excessive NaN values.

    Args:
        df: DataFrame with cutout references
        nan_threshold: Maximum allowed fraction of NaN values (0-1)

    Returns:
        pd.DataFrame: Filtered dataframe with low-NaN cutouts only
    """
    logging.info(f"Filtering cutouts with NaN threshold: {nan_threshold * 100:.1f}%")

    def check_nan(idx_row):
        idx, row = idx_row
        cutout = ds.load_image(row)
        if cutout is not None:
            nan_fraction = np.isnan(cutout).sum() / cutout.size
            if nan_fraction <= nan_threshold:
                return nan_fraction, idx
            else:
                return nan_fraction, None
        return 1.0, None

    results = p_map(check_nan, list(df.iterrows()))
    nan_stats, valid_indices = zip(*results)
    valid_indices = [idx for idx in valid_indices if idx is not None]
    removed_indices = [idx for idx in range(len(df)) if idx not in valid_indices]
    df_filtered = df.loc[valid_indices].copy()
    if LOG_DETAILED_STATS and nan_stats:
        nan_stats = np.array(nan_stats)
        logging.info("NaN filtering results:")
        logging.info(f"  Original: {len(df):,} cutouts")
        logging.info(f"  Filtered: {len(df_filtered):,} cutouts ({len(df_filtered) / len(df) * 100:.1f}% retained)")
        logging.info(f"  Removed:  {len(df) - len(df_filtered):,} cutouts")
        logging.info(f"  Mean NaN fraction: {nan_stats.mean() * 100:.2f}%")
        logging.info(f"  Max NaN fraction:  {nan_stats.max() * 100:.2f}%")
        if removed_indices:
            logging.debug(f"  Indices of removed images due to NaN: {removed_indices}")
    return df_filtered


# %%
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

    # NaN filtering
    df_clean = filter_by_nan_threshold(df_clean, nan_threshold=nan_threshold)

    return df, df_clean


# %%
def apply_label_mapping_and_filter(
    df_clean: pd.DataFrame, label_mapping: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map Hale class labels to grouped categories and
    apply longitude filtering/undersampling.
    Args:
        df_clean: Cleaned dataframe
        label_mapping: Dict mapping original to grouped labels
    Returns:
        (DataFrame after label mapping, DataFrame after all filtering)
    """

    df_original, df_processed = ut_d.undersample_group_filter(
        df_clean, label_mapping=label_mapping, long_limit_deg=config.LONG_LIMIT_DEG, undersample=config.UNDERSAMPLE
    )
    logging.info(f"Label mapping: {label_mapping}")
    logging.info(f"Label mapping applied: {len(df_original):,} → {len(df_processed):,}")
    class_dist = df_processed["grouped_labels"].value_counts()
    logging.info(
        "Final class distribution:"
        + "".join(
            f"\n  {label}: {count:,} ({count / len(df_processed) * 100:.1f}%)" for label, count in class_dist.items()
        )
    )
    return df_original, df_processed


# %%
def create_and_validate_cv_folds(
    df_processed: pd.DataFrame, n_splits: int = config.N_FOLDS, random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """
    Create stratified CV folds ensuring AR numbers do not overlap between train/val/test sets.
    Args:
        df_processed: DataFrame with grouped labels
        n_splits: Number of folds
        random_state: Random seed
    Returns:
        DataFrame with fold assignment columns
    """
    logging.info("Creating cross-validation folds...")

    # Create stratified folds grouped by AR number to prevent data leakage
    ut_d.split_data(
        df_processed,
        label_col="grouped_labels",
        group_col="number",  # group by AR number
        n_splits=n_splits,
        random_state=random_state,
    )

    # Validate AR number separation to ensure no data leakage
    _validate_fold_separation(df_processed)

    return df_processed


# %%
def _validate_fold_separation(df_processed: pd.DataFrame) -> None:
    """
    Ensure AR numbers do not overlap between train/val/test sets in any fold.
    Args:
        df_processed: DataFrame with fold assignment columns
    """
    for fold_column_name in [col for col in df_processed.columns if col.startswith("Fold ")]:
        fold_num = fold_column_name.split()[-1]
        masks = [df_processed[fold_column_name] == split for split in ["train", "val", "test"]]
        counts = [mask.sum() for mask in masks]
        ars = [set(df_processed[mask]["number"].unique()) for mask in masks]
        overlaps = {
            name: ars[i] & ars[j] for (name, i, j) in [("Train-Val", 0, 1), ("Train-Test", 0, 2), ("Val-Test", 1, 2)]
        }
        logging.info(f"Fold {fold_num}: Train={counts[0]:,}, Val={counts[1]:,}, Test={counts[2]:,}")
        logging.info(f"  Unique ARs - Train: {len(ars[0])}, Val: {len(ars[1])}, Test: {len(ars[2])}")
        if any(overlaps.values()):
            logging.error(f"  ERROR - AR number overlaps detected in Fold {fold_num}:")
            [logging.error(f"    {name} overlap: {sorted(overlap)}") for name, overlap in overlaps.items() if overlap]
        else:
            logging.info("  ✓ No AR number overlaps on different sets")


# %%
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
    _, df_clean = load_and_clean_dataset(nan_threshold=nan_threshold)
    # Rest of the pipeline remains the same...
    if label_mapping is None:
        label_mapping = config.label_mapping
    _, df_processed = apply_label_mapping_and_filter(df_clean, label_mapping=label_mapping)
    df_with_folds = create_and_validate_cv_folds(df_processed, n_splits=n_splits, random_state=random_state)
    if save_path:
        df_with_folds.to_parquet(save_path, index=False)
        logging.info(f"Processed dataset saved to: {save_path}")
    return df_with_folds


# %%
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


# %%
if __name__ == "__main__":
    main()
