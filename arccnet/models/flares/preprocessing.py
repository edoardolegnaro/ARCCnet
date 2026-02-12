"""
Flares data preprocessing module that reuses filtering from cutouts training.

This module provides data cleaning and filtering utilities for flare datasets,
reusing quality checks and location filtering from the cutouts pipeline.

Filtering Steps:
1. Quality flag filtering (HMI and MDI quality flags)
2. Missing path removal (both image paths)
3. Longitude/location filtering (front-hemisphere only)
4. Optional: NaN filtering on magnetogram cutouts
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from p_tqdm import p_map

logger = logging.getLogger(__name__)

# Filtering parameters
DEFAULT_NAN_THRESHOLD = 0.05  # Maximum allowed fraction of NaN values (5%)
DEFAULT_LONGITUDE_LIMIT = 65.0  # Maximum absolute longitude for front-hemisphere


def filter_by_nan_threshold(
    df: pd.DataFrame,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
    data_folder: str = "/ARCAFF/data",
    dataset_folder: str = "arcnet-v20251017/04_final",
) -> pd.DataFrame:
    """
    Filter out magnetogram cutouts with excessive NaN values.

    Args:
        df: DataFrame with cutout references and image paths
        nan_threshold: Maximum allowed fraction of NaN values (0-1)
        data_folder: Base data folder path
        dataset_folder: Path to cutout dataset folder

    Returns:
        Filtered dataframe with low-NaN cutouts only
    """
    logger.info(f"Filtering cutouts with NaN threshold: {nan_threshold * 100:.1f}%")

    def check_nan(idx_row):
        """Check NaN fraction for a magnetogram cutout."""
        idx, row = idx_row
        try:
            # Try loading from HMI first, then MDI
            path = None
            if pd.notna(row.get("path_image_cutout_hmi")) and row.get("path_image_cutout_hmi"):
                path = row["path_image_cutout_hmi"]
            elif pd.notna(row.get("path_image_cutout_mdi")) and row.get("path_image_cutout_mdi"):
                path = row["path_image_cutout_mdi"]

            if path:
                full_path = Path(data_folder) / dataset_folder / "fits" / Path(path).name
                if full_path.exists():
                    # Load FITS file
                    from astropy.io import fits

                    with fits.open(full_path) as hdul:
                        cutout = hdul[0].data
                        if cutout is not None:
                            nan_fraction = np.isnan(cutout).sum() / cutout.size
                            if nan_fraction <= nan_threshold:
                                return nan_fraction, idx
                            else:
                                return nan_fraction, None
        except Exception as e:
            logger.debug(f"Error loading cutout at idx {idx}: {e}")
            return 1.0, None

        return 1.0, None

    logger.info("Computing NaN statistics (this may take a while)...")
    results = p_map(check_nan, list(df.iterrows()))
    nan_stats, valid_indices = zip(*results)
    valid_indices = [idx for idx in valid_indices if idx is not None]

    df_filtered = df.loc[valid_indices].copy()
    removed_count = len(df) - len(df_filtered)

    if nan_stats:
        nan_stats_array = np.array(nan_stats)
        logger.info("NaN filtering results:")
        logger.info(f"  Original: {len(df):,} cutouts")
        logger.info(f"  Filtered: {len(df_filtered):,} cutouts ({len(df_filtered) / len(df) * 100:.1f}% retained)")
        logger.info(f"  Removed:  {removed_count:,} cutouts")
        logger.info(f"  Mean NaN fraction: {nan_stats_array.mean() * 100:.2f}%")
        logger.info(f"  Max NaN fraction:  {nan_stats_array.max() * 100:.2f}%")

    return df_filtered


def apply_quality_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality flag filtering to remove low-quality magnetograms.

    Uses the standard cutout pipeline quality checking:
    - HMI quality flags: "", "0x00000000", "0x00000400"
    - MDI quality flags: "", "00000000", "00000200"

    Args:
        df: Input dataframe

    Returns:
        Filtered dataframe with only good-quality magnetograms
    """
    logger.info("Applying quality flag filtering...")

    # Define quality flags (same as cutouts pipeline)
    hmi_good_flags = {"", "0x00000000", "0x00000400"}
    mdi_good_flags = {"", "00000000", "00000200"}

    # Filter by quality flags
    df_clean = df[df["QUALITY_hmi"].isin(hmi_good_flags) & df["QUALITY_mdi"].isin(mdi_good_flags)].copy()

    removed = len(df) - len(df_clean)
    logger.info(f"Quality filtering: {len(df):,} → {len(df_clean):,} ({removed:,} removed)")

    return df_clean


def apply_path_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where both HMI and MDI image paths are missing.

    Args:
        df: Input dataframe

    Returns:
        Filtered dataframe with at least one valid image path
    """
    logger.info("Applying path filtering...")

    def is_missing(series):
        return series.isna() | (series == "") | (series == "None")

    both_missing = is_missing(df["path_image_cutout_hmi"]) & is_missing(df["path_image_cutout_mdi"])

    df_filtered = df[~both_missing].reset_index(drop=True)
    removed = both_missing.sum()

    logger.info(f"Path filtering: {len(df):,} → {len(df_filtered):,} ({removed:,} removed)")

    stats = {
        "total": len(df_filtered),
        "hmi_only": (~is_missing(df_filtered["path_image_cutout_hmi"])).sum(),
        "mdi_only": (~is_missing(df_filtered["path_image_cutout_mdi"])).sum(),
        "both": (
            ~is_missing(df_filtered["path_image_cutout_hmi"]) & ~is_missing(df_filtered["path_image_cutout_mdi"])
        ).sum(),
    }

    logger.debug(f"Path distribution: HMI-only={stats['hmi_only']}, MDI-only={stats['mdi_only']}, Both={stats['both']}")

    return df_filtered


def apply_longitude_filtering(
    df: pd.DataFrame,
    max_longitude: float = DEFAULT_LONGITUDE_LIMIT,
) -> pd.DataFrame:
    """
    Filter to keep only front-hemisphere observations based on longitude.

    Removes observations with |longitude| > max_longitude to focus on
    front-facing solar regions with better magnetic field measurements.

    Args:
        df: Input dataframe with longitude columns
        max_longitude: Maximum absolute longitude in degrees (default: 65°)

    Returns:
        Filtered dataframe with front-hemisphere observations only
    """
    logger.info(f"Applying longitude filtering (max |lon| = {max_longitude}°)...")

    initial_count = len(df)

    # Use HMI longitude if available, otherwise MDI
    def get_longitude(row):
        if pd.notna(row.get("longitude_hmi")) and row.get("longitude_hmi"):
            return row["longitude_hmi"]
        elif pd.notna(row.get("longitude_mdi")) and row.get("longitude_mdi"):
            return row["longitude_mdi"]
        return np.nan

    df = df.copy()
    df["combined_longitude"] = df.apply(get_longitude, axis=1)

    # Filter by longitude
    valid_lon = df["combined_longitude"].notna()
    within_limit = np.abs(df["combined_longitude"]) <= max_longitude

    df_filtered = df[valid_lon & within_limit].drop(columns=["combined_longitude"]).reset_index(drop=True)

    removed = initial_count - len(df_filtered)
    logger.info(f"Longitude filtering: {initial_count:,} → {len(df_filtered):,} ({removed:,} removed)")

    return df_filtered


def preprocess_flare_data(
    df: pd.DataFrame,
    apply_quality_filter: bool = True,
    apply_path_filter: bool = True,
    apply_longitude_filter: bool = True,
    apply_nan_filter: bool = False,
    max_longitude: float = DEFAULT_LONGITUDE_LIMIT,
    nan_threshold: float = DEFAULT_NAN_THRESHOLD,
    data_folder: str = "/ARCAFF/data",
    dataset_folder: str = "arcnet-v20251017/04_final",
) -> pd.DataFrame:
    """
    Apply complete preprocessing pipeline to flare dataset.

    This pipeline reuses cutouts-based filtering to ensure consistency:
    1. Quality flag filtering (HMI/MDI)
    2. Missing path removal
    3. Longitude filtering (front-hemisphere)
    4. Optional NaN filtering on magnetograms

    Args:
        df: Input flare dataframe
        apply_quality_filter: Apply quality flag filtering (default: True)
        apply_path_filter: Remove rows with missing image paths (default: True)
        apply_longitude_filter: Apply longitude filtering (default: True)
        apply_nan_filter: Filter by NaN content in magnetograms (default: False, expensive)
        max_longitude: Maximum absolute longitude for filtering
        nan_threshold: Maximum NaN fraction allowed in magnetograms
        data_folder: Base data folder
        dataset_folder: Path to cutout dataset folder

    Returns:
        Preprocessed dataframe
    """
    logger.info("=" * 60)
    logger.info("FLARE DATA PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Starting dataset size: {len(df):,} records")

    df_processed = df.copy()

    if apply_quality_filter:
        df_processed = apply_quality_filtering(df_processed)

    if apply_path_filter:
        df_processed = apply_path_filtering(df_processed)

    if apply_longitude_filter:
        df_processed = apply_longitude_filtering(df_processed, max_longitude)

    if apply_nan_filter:
        df_processed = filter_by_nan_threshold(
            df_processed,
            nan_threshold=nan_threshold,
            data_folder=data_folder,
            dataset_folder=dataset_folder,
        )

    total_removed = len(df) - len(df_processed)
    retention_rate = (len(df_processed) / len(df) * 100) if len(df) > 0 else 0

    logger.info("=" * 60)
    logger.info(f"Final dataset size: {len(df_processed):,} records")
    logger.info(f"Total removed: {total_removed:,} records ({100 - retention_rate:.1f}%)")
    logger.info(f"Retention rate: {retention_rate:.1f}%")
    logger.info("=" * 60)

    return df_processed
