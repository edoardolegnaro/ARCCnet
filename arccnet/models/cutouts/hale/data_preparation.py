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

import pandas as pd

import arccnet.models.cutouts.hale.config as config
from arccnet.models import dataset_utils as ut_d


def load_and_clean_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw solar magnetogram dataset and apply basic cleaning operations.

    Loads the raw dataset from configured paths and applies standard cleaning
    procedures to remove invalid/corrupted entries and standardize data format.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Raw dataset and cleaned dataset
            - First element: Original unprocessed dataframe
            - Second element: Cleaned dataframe with invalid entries removed

    Note:
        Uses configuration from config module for data paths and cleaning parameters.
    """
    logging.info("Loading and cleaning dataset...")

    df, _, _ = ut_d.make_dataframe(config.DATA_FOLDER, config.DATASET_FOLDER, config.DF_FILE_NAME)
    logging.info(f"Original DataFrame shape: {df.shape}")

    df_clean = ut_d.cleanup_df(df)
    logging.info(f"After cleanup: {df_clean.shape} ({len(df_clean) / len(df) * 100:.1f}% retained)")

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
) -> pd.DataFrame:
    """
    Complete end-to-end dataset preparation pipeline for ARCCNet training.

    Orchestrates the full data preparation process: loading, cleaning, label mapping,
    filtering, and cross-validation fold creation. Optionally saves the final dataset.

    Args:
        save_path: Path to save processed dataset (default: None, no saving)
        n_splits: Number of cross-validation folds (default from config)
        random_state: Random seed for reproducibility (default from config)
        label_mapping: Custom label mapping dict (default: use config mapping)

    Returns:
        pd.DataFrame: Fully processed dataset with fold assignments ready for training

    Pipeline Steps:
        1. Load raw dataset and apply cleaning
        2. Apply label mapping and filtering/undersampling
        3. Create CV folds with AR number separation validation
        4. Optionally save processed dataset

    Example:
        df = prepare_dataset(save_path='processed_data.parquet', n_splits=5)
    """
    # Step 1: Load and clean raw data
    df_raw, df_clean = load_and_clean_dataset()

    # Step 2: Use default label mapping if none provided
    if label_mapping is None:
        label_mapping = config.label_mapping
        logging.info("Using default label mapping from config")

    # Step 3: Apply label mapping and filtering/undersampling
    df_original, df_processed = apply_label_mapping_and_filter(df_clean, label_mapping=label_mapping)

    # Step 4: Create cross-validation folds with AR separation validation
    df_with_folds = create_and_validate_cv_folds(df_processed, n_splits=n_splits, random_state=random_state)

    # Step 5: Save processed dataset if path provided
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
