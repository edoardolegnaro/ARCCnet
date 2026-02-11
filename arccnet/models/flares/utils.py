import os
import re
import glob
import logging

from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

FLARE_CLASSES = ["A", "B", "C", "M", "X"]
MAG_CLASS_MAPPING = {
    "Alpha": "α",
    "Beta": "β",
    "Beta-Delta": "β-δ",
    "Beta-Gamma": "β-γ",
    "Beta-Gamma-Delta": "β-γ-δ",
    "Gamma": "γ",
    "Gamma-Delta": "γ-δ",
}  # Mapping to Greek letters
MAG_CLASS_ORDER = ["α", "β", "β-δ", "β-γ", "β-γ-δ", "γ", "γ-δ"]  # Order for display


def check_fits_file_existence(df, data_folder, dataset_folder):
    """
    Iterates through rows of a DataFrame, checks if a corresponding FITS file exists,
    adds a column indicating existence, and collects indices where path info is missing.

    Args:
        df (pd.DataFrame): The DataFrame containing potential paths to FITS files
                                        in columns like 'path_image_cutout_hmi' or 'path_image_cutout_mdi'.
        data_folder (str): The base directory where the data is stored.
        dataset_folder (str): The subdirectory within data_folder where the dataset is located.

    Returns:
        tuple: A tuple containing:
                - pd.DataFrame: The DataFrame with a new column 'file_exists' (boolean).
                - list: A list of indices from the original DataFrame where both
                            'path_image_cutout_hmi' and 'path_image_cutout_mdi' were missing.
    """

    def is_missing(value):
        """Check if a path value is missing or invalid."""
        if value is None:
            return True
        if isinstance(value, str):
            return value == "" or value == "None"
        # Check for NaN values
        try:
            import pandas as pd

            return pd.isna(value)
        except ImportError:
            return False

    def extract_key_info(filename):
        """Extract date, AR number, and instrument from filename.
        Returns tuple of (date, ar_number, instrument) or None if pattern doesn't match.
        """
        # Pattern: YYYYMMDD_HHMMSS_[prefix]-[AR_number]_..._[instrument][_SIDE1].fits
        # Examples:
        #   19960604_000130_I-7968_mag_MDI.fits
        #   19961212_235945_I-8003_MDI.fits
        #   20100407_235819_I-11060_HMI_SIDE1.fits
        # Match and extract: date, AR number, and instrument (before optional _SIDE1)
        match = re.match(r"(\d{8})_\d{6}_[A-Z]+-(\d+).*_([A-Z]+)(?:_SIDE\d+)?\.fits", filename)
        if match:
            date, ar_number, instrument = match.groups()
            return (date, ar_number, instrument)
        return None

    # Pre-compute mapping based on (date, ar_number, instrument) to actual files
    fits_dir = os.path.join(data_folder, dataset_folder, "data/cutout_classification/fits")
    all_fits_files = glob.glob(os.path.join(fits_dir, "*.fits"))

    logger.info(f"Looking for FITS files in: {fits_dir}")
    logger.info(f"Found {len(all_fits_files)} FITS files")

    key_to_file = {}
    for fits_file in all_fits_files:
        filename = os.path.basename(fits_file)
        key = extract_key_info(filename)

        if key:
            # Prefer _mag_ files over _cont_ files
            if key not in key_to_file or "_mag_" in filename:
                key_to_file[key] = filename
        else:
            logger.debug(f"Could not extract key from filename: {filename}")

    logger.info(f"Successfully mapped {len(key_to_file)} unique (date, AR, instrument) keys")

    df["file_exists"] = False
    missing_path_indices = []

    for index, row in df.iterrows():
        hmi_path = row.get("path_image_cutout_hmi")
        mdi_path = row.get("path_image_cutout_mdi")

        hmi_missing = is_missing(hmi_path)
        mdi_missing = is_missing(mdi_path)

        # If both paths are missing, record the index and continue
        if hmi_missing and mdi_missing:
            missing_path_indices.append(index)
            continue

        # Prefer HMI path if available, otherwise use MDI path
        if not hmi_missing:
            path_value = hmi_path
            path_column = "path_image_cutout_hmi"
        else:
            path_value = mdi_path
            path_column = "path_image_cutout_mdi"

        base_filename = os.path.basename(path_value)
        key = extract_key_info(base_filename)

        if not key:
            logger.debug(f"Row {index}: Could not extract key from dataframe filename: {base_filename}")

        # Look up the file by key (date, ar_number, instrument)
        if key and key in key_to_file:
            df.loc[index, "file_exists"] = True
            # Update the path column to point to the actual file found
            df.loc[index, path_column] = key_to_file[key]
        elif key:
            logger.debug(f"Row {index}: No matching file for key {key} (from {base_filename})")

    files_found = df["file_exists"].sum()
    logger.info(f"Found existing files for {files_found}/{len(df)} rows")

    return df, missing_path_indices


def split_dataframe(df, stratify_col, test_size=0.1, val_size=0.2, random_state=42):
    """
    Split dataframe into train, validation, and test sets with stratification,
    while keeping Active Regions (AR) together.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to be split
    stratify_col : str
        Column name to use for stratification (e.g., 'flares_above_C')
    test_size : float, optional (default=0.1)
        Proportion of dataset to include in test split
    val_size : float, optional (default=0.2)
        Proportion of dataset to include in validation split
    random_state : int, optional (default=42)
        Controls the shuffling applied to the data before splitting

    Returns:
    --------
    tuple of pandas.DataFrame
        (train_df, val_df, test_df)
    """
    # Validate inputs
    if test_size + val_size >= 1:
        raise ValueError("Combined test and validation sizes must be less than 1")
    if stratify_col not in df.columns:
        raise ValueError(f"Stratification column '{stratify_col}' not found in dataframe")

    # Get unique ARs and their stratification labels
    ar_groups = df["number"].unique()
    ar_labels = df.groupby("number")[stratify_col].max().loc[ar_groups].values

    # First split: train_val (will be split further) vs test
    train_val_ars, test_ars = train_test_split(
        ar_groups, test_size=test_size, stratify=ar_labels, random_state=random_state
    )

    # Second split: split train_val into train and validation
    train_val_labels = df.groupby("number")[stratify_col].max().loc[train_val_ars].values
    adjusted_val_size = val_size / (1 - test_size)  # Relative to train_val size

    train_ars, val_ars = train_test_split(
        train_val_ars, test_size=adjusted_val_size, stratify=train_val_labels, random_state=random_state
    )

    # Create final splits using AR numbers
    train_df = df[df["number"].isin(train_ars)]
    val_df = df[df["number"].isin(val_ars)]
    test_df = df[df["number"].isin(test_ars)]

    # Verification
    logger.info("Unique ARs in Train: %s", train_df["number"].nunique())
    logger.info("Unique ARs in Validation: %s", val_df["number"].nunique())
    logger.info("Unique ARs in Test: %s", test_df["number"].nunique())

    logger.info("AR Overlap Check:")
    logger.info("Train vs Val AR Overlap: %s", len(set(train_ars) & set(val_ars)))
    logger.info("Train vs Test AR Overlap: %s", len(set(train_ars) & set(test_ars)))
    logger.info("Val vs Test AR Overlap: %s", len(set(val_ars) & set(test_ars)))

    return train_df, val_df, test_df
