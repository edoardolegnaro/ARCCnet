import logging
import argparse
from typing import Any
from pathlib import Path

import pandas as pd

import arccnet.models.cutouts.hale.config as config
from arccnet.models import dataset_utils as ut_d


def load_and_clean_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean the raw dataset."""
    logging.info("Loading and cleaning dataset...")

    df, _, _ = ut_d.make_dataframe(config.DATA_FOLDER, config.DATASET_FOLDER, config.DF_FILE_NAME)
    logging.info(f"Original DataFrame shape: {df.shape}")

    df_clean = ut_d.cleanup_df(df)
    logging.info(f"After cleanup: {df_clean.shape} ({len(df_clean) / len(df) * 100:.1f}% retained)")

    return df, df_clean


def apply_label_mapping_and_filter(
    df_clean: pd.DataFrame, label_mapping: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply label mapping and filtering."""

    df_original, df_processed = ut_d.undersample_group_filter(
        df_clean, label_mapping=label_mapping, long_limit_deg=config.LONG_LIMIT_DEG, undersample=config.UNDERSAMPLE
    )

    logging.info(f"Label mapping: {label_mapping}")
    logging.info(f"Label mapping applied: {len(df_original):,} → {len(df_processed):,}")

    # Log class distribution
    class_dist = df_processed["grouped_labels"].value_counts()
    logging.info("Final class distribution:")
    for label, count in class_dist.items():
        pct = count / len(df_processed) * 100
        logging.info(f"  {label}: {count:,} ({pct:.1f}%)")

    return df_original, df_processed


def create_and_validate_cv_folds(
    df_processed: pd.DataFrame, n_splits: int = config.N_FOLDS, random_state: int = config.RANDOM_STATE
) -> pd.DataFrame:
    """Create cross-validation folds and validate AR number separation."""
    logging.info("Creating cross-validation folds...")

    ut_d.split_data(
        df_processed,
        label_col="grouped_labels",
        group_col="number",
        n_splits=n_splits,
        random_state=random_state,
    )

    # Validate AR number separation
    _validate_fold_separation(df_processed)

    return df_processed


def _validate_fold_separation(df_processed: pd.DataFrame) -> None:
    """Validate that AR numbers don't overlap between train/val/test sets."""
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
    """Complete dataset preparation pipeline."""
    # Load and clean data
    df_raw, df_clean = load_and_clean_dataset()

    # Use config.label_mapping if label_mapping is None
    if label_mapping is None:
        label_mapping = config.label_mapping
        logging.info("Using default label mapping from config")

    # Apply label mapping and filtering
    df_original, df_processed = apply_label_mapping_and_filter(df_clean, label_mapping=label_mapping)

    # Create cross-validation folds
    df_with_folds = create_and_validate_cv_folds(df_processed, n_splits=n_splits, random_state=random_state)

    # Save processed dataset if path provided
    if save_path:
        df_with_folds.to_parquet(save_path, index=False)
        logging.info(f"Processed dataset saved to: {save_path}")

    return df_with_folds


def main():
    """Main function to run the data preparation pipeline."""
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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    save_path = args.save_path
    if save_path is None:
        save_path = (
            Path(config.DATA_FOLDER)
            / f"processed_dataset_{config.classes}_{args.n_splits}-splits_rs-{args.random_state}.parquet"
        )

    prepare_dataset(
        save_path=save_path, n_splits=args.n_splits, random_state=args.random_state, label_mapping=config.label_mapping
    )


if __name__ == "__main__":
    main()
