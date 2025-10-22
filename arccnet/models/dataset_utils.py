import os
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils import resample

from astropy.time import Time

from arccnet import load_config
from arccnet.models import labels

config = load_config()


def make_dataframe(data_folder, dataset_folder, file_name):
    """
    Load and process the ARCCNet cutout dataset.

    - Reads a parquet file and converts Julian dates to datetime.
    - Removes rows with problematic quicklook magnetograms (by filename).
    - Adds 'label' (from magnetic_class or region_type) and 'date_only' columns.
    - Returns the full DataFrame, a filtered DataFrame with only AR/IA regions, and the removed problematic rows.

    Returns
    -------
    df : pandas.DataFrame
        The processed DataFrame.
    AR_df : pandas.DataFrame
        DataFrame with only active regions (AR) and intermediate regions (IA).
    filtered_df : pandas.DataFrame
        DataFrame of removed problematic quicklook rows.
    """
    df = _load_and_process_data(data_folder, dataset_folder, file_name)
    df, filtered_df = _remove_problematic_quicklooks(df)
    df, AR_df = _add_labels_and_filter_regions(df)
    return df, AR_df, filtered_df


def _load_and_process_data(data_folder, dataset_folder, file_name):
    """Load data and convert dates."""
    df = pd.read_parquet(os.path.join(data_folder, dataset_folder, file_name))
    return _convert_jd_to_datetime(df)


def _convert_jd_to_datetime(df):
    """Convert Julian dates to datetime objects."""
    df["time"] = df["target_time.jd1"] + df["target_time.jd2"]
    times = Time(df["time"], format="jd")
    df["dates"] = pd.to_datetime(times.iso)
    return df


def _remove_problematic_quicklooks(df):
    """Remove problematic magnetograms from the dataset."""
    problematic_quicklooks = [ql.strip() for ql in config.get("magnetograms", "problematic_quicklooks").split(",")]
    mask = df["quicklook_path_mdi"].apply(lambda x: os.path.basename(x) in problematic_quicklooks)
    filtered_df = df[mask]
    df = df[~mask].reset_index(drop=True)
    return df, filtered_df


def _add_labels_and_filter_regions(df):
    """Label the data and filter for AR and IA regions."""
    df["label"] = np.where(df["magnetic_class"] == "", df["region_type"], df["magnetic_class"])
    df["date_only"] = df["dates"].dt.date
    AR_df = pd.concat([df[df["region_type"] == "AR"], df[df["region_type"] == "IA"]])
    return df, AR_df


def cleanup_df(df, log_level=logging.INFO):
    """
    Clean dataframe by removing bad quality data and rows with missing paths.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to clean.
    log_level : int, optional
        Logging level for output messages. Use logging.DEBUG, logging.INFO, etc.
        Set to None to disable logging. Defaults to logging.INFO.

    Returns
    -------
    pandas.DataFrame
        Cleaned dataframe with quality filtering and missing path removal applied.
    """
    logger = logging.getLogger(__name__)

    # Define quality flags
    hmi_good_flags = {"", "0x00000000", "0x00000400"}
    mdi_good_flags = {"", "00000000", "00000200"}

    # Filter by quality flags
    df_clean = df[df["QUALITY_hmi"].isin(hmi_good_flags) & df["QUALITY_mdi"].isin(mdi_good_flags)].copy()

    if log_level is not None:
        _log_filtering_stats(df, df_clean, logger, log_level)

    # Remove rows where both paths are missing
    def is_missing(series):
        return series.isna() | (series == "") | (series == "None")

    both_missing = is_missing(df_clean["path_image_cutout_hmi"]) & is_missing(df_clean["path_image_cutout_mdi"])

    if log_level is not None:
        _log_path_analysis(df_clean, both_missing, logger, log_level)

    return df_clean[~both_missing].reset_index(drop=True)


def _log_filtering_stats(df_orig, df_clean, logger, log_level):
    """Log data filtering statistics."""
    df_HMI = df_orig[df_orig["path_image_cutout_mdi"] == ""]
    df_MDI = df_orig[df_orig["path_image_cutout_hmi"] == ""]

    hmi_good_flags = {"", "0x00000000", "0x00000400"}
    mdi_good_flags = {"", "00000000", "00000200"}

    df_HMI_clean = df_HMI[df_HMI["QUALITY_hmi"].isin(hmi_good_flags)]
    df_MDI_clean = df_MDI[df_MDI["QUALITY_mdi"].isin(mdi_good_flags)]

    logger.log(log_level, "DATA FILTERING Stats")
    logger.log(log_level, "-" * 40)

    for name, orig, clean in [
        ("HMI", len(df_HMI), len(df_HMI_clean)),
        ("MDI", len(df_MDI), len(df_MDI_clean)),
        ("Total", len(df_orig), len(df_clean)),
    ]:
        pct = clean / orig * 100 if orig > 0 else 0
        logger.log(log_level, f"{name}: {clean:,}/{orig:,} ({pct:.1f}% retained)")

    logger.log(log_level, "-" * 40)


def _log_path_analysis(df_clean, both_missing, logger, log_level):
    """Log path analysis statistics."""
    stats = {
        "total": len(df_clean),
        "hmi_none": (df_clean["path_image_cutout_hmi"] == "None").sum(),
        "hmi_empty": (df_clean["path_image_cutout_hmi"] == "").sum(),
        "mdi_none": (df_clean["path_image_cutout_mdi"] == "None").sum(),
        "mdi_empty": (df_clean["path_image_cutout_mdi"] == "").sum(),
        "both_missing": both_missing.sum(),
    }

    logger.log(log_level, "PATH ANALYSIS:")
    logger.log(log_level, "-" * 40)
    logger.log(log_level, f"Total rows in df_clean: {stats['total']:,}")
    logger.log(log_level, f"HMI paths - None: {stats['hmi_none']:,}, Empty: {stats['hmi_empty']:,}")
    logger.log(log_level, f"MDI paths - None: {stats['mdi_none']:,}, Empty: {stats['mdi_empty']:,}")
    logger.log(
        log_level,
        f"Both paths missing: {stats['both_missing']:,} ({stats['both_missing'] / stats['total'] * 100:.1f}%)",
    )


def undersample_group_filter(df, label_mapping, long_limit_deg=60, undersample=True, buffer_percentage=0.1):
    """
    Filter data based on longitude limit, group labels according to mapping, and optionally undersample.

    This function performs a multi-step data processing pipeline:
    1. Assigns front/rear location based on longitude limits
    2. Maps original labels to grouped labels using the provided mapping
    3. Filters out rows with unmapped labels (None values)
    4. Optionally undersamples the majority class for balanced training
    5. Keeps only front-hemisphere samples for final output

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing solar region data with 'label', 'longitude_hmi',
        'longitude_mdi' columns.
    label_mapping : dict
        Dictionary mapping original labels to grouped labels. Unmapped labels
        (mapped to None) will be filtered out.
    long_limit_deg : int, optional
        Longitude limit in degrees for front/rear classification. Regions with
        |longitude| <= limit are considered 'front' (default: 60).
    undersample : bool, optional
        Whether to undersample the majority class to balance dataset (default: True).
    buffer_percentage : float, optional
        Buffer percentage added to second-largest class size when undersampling
        majority class (default: 0.1 = 10%).

    Returns
    -------
    df_original : pandas.DataFrame
        Original dataframe with added columns: 'location', 'grouped_labels',
        'encoded_labels'.
    df_processed : pandas.DataFrame
        Processed dataframe with label mapping applied, optional undersampling,
        and only front-hemisphere samples retained.

    Raises
    ------
    ValueError
        If input DataFrame is empty or no data remains after label mapping.
    """
    logger = logging.getLogger(__name__)

    if df.empty:
        raise ValueError("Input DataFrame is empty")

    logger.debug(f"Input: {len(df):,} rows")

    df = df.copy()
    df = _assign_location(df, long_limit_deg, logger)
    df = _apply_label_mapping(df, label_mapping, logger)

    if len(df) == 0:
        raise ValueError("No data remaining after applying label mapping")

    df["encoded_labels"] = df["grouped_labels"].map(labels.LABEL_TO_INDEX)
    _log_unmapped_encoded(df, logger)

    df_processed = _perform_undersampling(df, undersample, buffer_percentage, logger) if undersample else df.copy()
    df_processed = _filter_front_hemisphere(df_processed, logger)

    _log_final_summary(df_processed, logger)

    return df, df_processed


def _assign_location(df, lon_limit_deg, logger):
    """Assigns front/rear location based on longitude limits."""
    original_labels = df["label"].value_counts()
    logger.debug(f"Original label distribution:\n{original_labels}")

    location_results = filter_by_location(df, lon_limit_deg=lon_limit_deg)
    df["location"] = "rear"
    df.loc[location_results["mask_front"], "location"] = "front"

    front_count = (df["location"] == "front").sum()
    rear_count = len(df) - front_count
    logger.debug(
        f"Location assignment: {front_count:,} front ({front_count / len(df) * 100:.1f}%), {rear_count:,} rear ({rear_count / len(df) * 100:.1f}%)"
    )

    front_labels = df[df["location"] == "front"]["label"].value_counts()
    logger.debug(f"Front hemisphere label distribution:\n{front_labels}")
    return df


def _apply_label_mapping(df, label_mapping, logger):
    """Apply label mapping and filter unmapped labels."""
    df["grouped_labels"] = df["label"].map(label_mapping)
    initial_count = len(df)

    mapped_count = df["grouped_labels"].notna().sum()
    unmapped_count = df["grouped_labels"].isna().sum()
    logger.debug(
        f"Label mapping results: {mapped_count:,} mapped ({mapped_count / initial_count * 100:.1f}%), {unmapped_count:,} unmapped ({unmapped_count / initial_count * 100:.1f}%)"
    )

    unmapped_labels = df[df["grouped_labels"].isna()]["label"].value_counts()
    if len(unmapped_labels) > 0:
        logger.debug(f"Unmapped labels being filtered out:\n{unmapped_labels}")

    df = df.dropna(subset=["grouped_labels"]).reset_index(drop=True)
    grouped_labels = df["grouped_labels"].value_counts()
    logger.debug(f"After label mapping - grouped label distribution:\n{grouped_labels}")
    return df


def _log_unmapped_encoded(df, logger):
    """Check for unmapped encoded labels."""
    unmapped_encoded = df["encoded_labels"].isna().sum()
    if unmapped_encoded > 0:
        logger.warning(f"Found {unmapped_encoded} rows with unmapped encoded labels")
        unmapped_grouped = df[df["encoded_labels"].isna()]["grouped_labels"].value_counts()
        logger.warning(f"Unmapped grouped labels:\n{unmapped_grouped}")


def _perform_undersampling(df, undersample, buffer_percentage, logger):
    """Undersample the majority class."""
    if not undersample:
        logger.debug("Undersampling disabled")
        return df.copy()

    class_counts = df["grouped_labels"].value_counts()
    logger.debug(f"Before undersampling - class distribution:\n{class_counts}")

    if len(class_counts) < 2:
        logger.warning("Less than 2 classes available, skipping undersampling")
        return df.copy()

    majority_class = class_counts.idxmax()
    n_samples = min(int(class_counts.iloc[1] * (1 + buffer_percentage)), class_counts.iloc[0])

    logger.debug("Undersampling strategy:")
    logger.debug(f"  Majority class: {majority_class} ({class_counts.iloc[0]:,} samples)")
    logger.debug(f"  Second largest class: {class_counts.index[1]} ({class_counts.iloc[1]:,} samples)")
    logger.debug(f"  Target size for majority: {n_samples:,} (with {buffer_percentage * 100:.1f}% buffer)")

    df_majority_resampled = resample(
        df[df["grouped_labels"] == majority_class], replace=False, n_samples=n_samples, random_state=42
    )
    df_others = [df[df["grouped_labels"] == label] for label in class_counts.index if label != majority_class]
    df_du = pd.concat([*df_others, df_majority_resampled], ignore_index=True)

    logger.debug("After undersampling:")
    after_undersample = df_du["grouped_labels"].value_counts()
    for label in after_undersample.index:
        count = after_undersample[label]
        pct = count / len(df_du) * 100
        logger.debug(f"  {label}: {count:,} ({pct:.1f}%)")
    return df_du


def _filter_front_hemisphere(df, logger):
    """Keep only front samples."""
    before_front_filter = len(df)
    df_filtered = df[df["location"] == "front"].reset_index(drop=True)
    after_front_filter = len(df_filtered)

    if before_front_filter > 0:
        logger.debug(
            f"Front hemisphere filtering: {before_front_filter:,} -> {after_front_filter:,} ({after_front_filter / before_front_filter * 100:.1f}% retained)"
        )
    return df_filtered


def _log_final_summary(df, logger):
    """Log final summary."""
    final_counts = df["grouped_labels"].value_counts()
    logger.debug(f"Final output: {len(df):,} rows")
    logger.debug("Final class distribution:")
    for label in final_counts.index:
        count = final_counts[label]
        pct = count / len(df) * 100 if len(df) > 0 else 0
        logger.debug(f"  {label}: {count:,} ({pct:.1f}%)")


def split_data(df, label_col, group_col, n_splits=5, random_state=42):
    """
    Split the data into training, validation, and test sets for cross-validation
    using Stratified Group K-Fold approach.

    This implementation ensures:
    1.  Each group appears in the test set exactly once across all folds.
    2.  Each group appears in the validation set exactly once across all folds.
    3.  There is no overlap between train, validation, and test sets within any given fold.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be split.
    label_col : str
        The column to be used for stratification.
    group_col : str
        The column to group by, ensuring groups are not split across sets.
    n_splits : int, optional
        The number of folds for cross-validation. Defaults to 5.
    random_state : int, optional
        The random seed for reproducibility. Defaults to 42.

    Returns
    -------
    list of tuples
        A list where each tuple contains (fold_number, train_df, val_df, test_df).
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    X = df
    y = df[label_col]
    groups = df[group_col]

    # Generate indices for all folds at once
    fold_indices = list(sgkf.split(X, y, groups))

    fold_dataframes = []
    for i in range(n_splits):
        # Assign test, validation, and training folds
        test_idx = fold_indices[i][1]
        val_idx = fold_indices[(i + 1) % n_splits][1]  # Use the next fold for validation

        # All other folds are used for training
        train_folds_indices = [j for j in range(n_splits) if j != i and j != (i + 1) % n_splits]
        train_idx = np.concatenate([fold_indices[j][1] for j in train_folds_indices])

        # Create the dataframes
        train_df = X.iloc[train_idx]
        val_df = X.iloc[val_idx]
        test_df = X.iloc[test_idx]

        fold_dataframes.append((i + 1, train_df, val_df, test_df))

        # Annotate the original DataFrame
        X.loc[train_df.index, f"Fold {i + 1}"] = "train"
        X.loc[val_df.index, f"Fold {i + 1}"] = "val"
        X.loc[test_df.index, f"Fold {i + 1}"] = "test"

    return fold_dataframes


def filter_by_location(df, lon_limit_deg=65, lat_limit_deg=None, limb_r_max=None):
    """
    Return masks and filtered DataFrames using |lon|, optional |lat| and optional inner-disc radius.
    Chooses HMI coords when an HMI path exists, else MDI.
    """

    # Build an HMI-available mask from known path columns
    def nonempty_mask(s):
        return (~s.isna()) & (s != "") & (s != "None")

    hmi_candidates = ["path_image_cutout_hmi", "processed_path_image_hmi", "quicklook_path_hmi"]
    hmi_mask = np.zeros(len(df), dtype=bool)
    for col in hmi_candidates:
        if col in df.columns:
            hmi_mask |= nonempty_mask(df[col]).to_numpy()

    lon_deg = np.where(hmi_mask, df["longitude_hmi"].to_numpy(), df["longitude_mdi"].to_numpy())
    lat_deg = np.where(hmi_mask, df["latitude_hmi"].to_numpy(), df["latitude_mdi"].to_numpy())

    # Vector masks (avoid Python bools)
    lon_ok = (np.abs(lon_deg) <= lon_limit_deg) if lon_limit_deg is not None else np.ones(len(df), dtype=bool)
    lat_ok = (np.abs(lat_deg) <= lat_limit_deg) if lat_limit_deg is not None else np.ones(len(df), dtype=bool)

    # Optional limb exclusion via projected radius r = sqrt(y^2 + z^2)
    if limb_r_max is not None:
        lon = np.deg2rad(lon_deg)
        lat = np.deg2rad(lat_deg)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        r = np.sqrt(y**2 + z**2)
        limb_ok = r <= limb_r_max
    else:
        limb_ok = np.ones(len(df), dtype=bool)

    front_mask = lon_ok & lat_ok & limb_ok
    rear_mask = (
        (np.logical_not(lon_ok) & lat_ok & limb_ok) if lon_limit_deg is not None else np.zeros(len(df), dtype=bool)
    )

    return {
        "mask_front": front_mask,
        "mask_rear": rear_mask,
        "df_front": df[front_mask].copy(),
        "df_rear": df[rear_mask].copy(),
    }
