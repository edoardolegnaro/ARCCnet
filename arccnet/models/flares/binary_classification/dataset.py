"""
Module for loading, preparing, and splitting flare dataset.
"""
import os
import logging
from typing import Optional

import pandas as pd

from arccnet.models.flares import utils as ut_f

# Configure basic logging (can be configured externally in a real application)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- Main Data Loading and Splitting Function ---


def load_and_split_data(
    data_folder: str,
    df_flares_name: str,
    dataset_folder: str,
    target_column: str,
    test_size: float,
    val_size: float,
    random_state: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data, creates binary flare columns, splits into train/val/test,
    logs distributions, and returns the splits.

    Args:
        data_path (str): Path to the input Parquet file.
        target_column (str): Column name for the primary binary flare target used
                             for logging distribution (e.g., 'flares_above_C').
                             Note: Splitting uses stratify_col_for_split.
        test_size (float): Proportion of the dataset to include in the test split.
        val_size (float): Proportion of the *remaining* dataset (after test split)
                          to include in the validation split.
        random_state (Optional[int]): Random seed for the splitting function. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train_df, val_df, test_df

    Raises:
        FileNotFoundError: If the data_path does not exist.
        ValueError: If target_column or necessary magnetic_class column is missing.
        Exception: Propagates exceptions from the split_function.
    """
    data_path = os.path.join(data_folder, df_flares_name)
    logger.info(f"Loading data from: {data_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    df_flares = pd.read_parquet(data_path)
    logger.info(f"Loaded dataframe with shape: {df_flares.shape}")
    df_flares_exists, none_idxs = ut_f.check_fits_file_existence(df_flares.copy(), data_folder, dataset_folder)
    logger.info(
        f"""Checked existence of fits locally.
                        {len(none_idxs)} have no path.
                        {df_flares_exists['file_exists'].sum()} found, {len(df_flares_exists) - df_flares_exists['file_exists'].sum()} not present.
                        """
    )
    df_flares = df_flares_exists[df_flares_exists["file_exists"]].copy()

    # --- Feature Engineering: Binary Flare Columns ---
    logger.info("Creating binary 'flares_above_ ' columns...")
    for i in range(len(ut_f.FLARE_CLASSES)):
        threshold_class = ut_f.FLARE_CLASSES[i]
        # Select columns from current class onward
        columns_to_check = [f for f in ut_f.FLARE_CLASSES[i:] if f in df_flares.columns]
        if not columns_to_check:
            logger.warning(
                f"No columns found for flare classes {ut_f.FLARE_CLASSES[i:]} in dataframe. Skipping 'flares_above_{threshold_class}'."
            )
            continue

        new_col_name = f"flares_above_{threshold_class}"
        df_flares[new_col_name] = (df_flares[columns_to_check].fillna(0) > 0).any(axis=1).astype(int)
        logger.debug(f"Created column: {new_col_name}")

    # --- Splitting Data ---
    logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}, random_state={random_state}")
    try:
        train_df, val_df, test_df = ut_f.split_dataframe(
            df=df_flares,
            stratify_col=target_column,
            test_size=test_size,
            val_size=val_size,  #
            random_state=random_state,
        )
    except Exception as e:
        logger.error(f"Error during data splitting: {e}")
        raise  # Re-raise the exception

    logger.info(f"Split complete. Shapes: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}")

    # --- Logging Distributions ---
    flare_dist = pd.concat(
        [
            train_df["flares_above_C"].value_counts().rename("Train"),
            val_df["flares_above_C"].value_counts().rename("Validation"),
            test_df["flares_above_C"].value_counts().rename("Test"),
        ],
        axis=1,
    )
    for col in flare_dist.columns:
        percentages = (flare_dist[col] / flare_dist[col].sum() * 100).round(1)
        flare_dist[col] = flare_dist[col].astype(str) + " (" + percentages.astype(str) + "%)"
    logger.info(f"\nFlare Classification Distribution ({target_column}):\n{flare_dist.to_string()}")

    # For magnetic class - first map the class names
    mag_dist = pd.concat(
        [
            train_df["magnetic_class"].map(ut_f.MAG_CLASS_MAPPING).value_counts().rename("Train"),
            val_df["magnetic_class"].map(ut_f.MAG_CLASS_MAPPING).value_counts().rename("Validation"),
            test_df["magnetic_class"].map(ut_f.MAG_CLASS_MAPPING).value_counts().rename("Test"),
        ],
        axis=1,
    )

    # Reindex to ensure correct order
    mag_dist = mag_dist.reindex(ut_f.MAG_CLASS_ORDER)
    # Combine counts and percentages in the same columns
    for col in mag_dist.columns:
        percentages = (mag_dist[col] / mag_dist[col].sum() * 100).round(1)
        mag_dist[col] = mag_dist[col].astype(str) + " (" + percentages.astype(str) + "%)"
    # Fill NaN values with "0 (0%)" if any classes are missing
    mag_dist = mag_dist.fillna("0 (0%)")
    # No need for fillna here as _calculate_and_format_distribution handles it with `order`
    logger.info(f"\nMagnetic Class Distribution:\n{mag_dist.to_string()}")

    return train_df, val_df, test_df
