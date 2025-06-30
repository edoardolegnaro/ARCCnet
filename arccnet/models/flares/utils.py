import os

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
    Vectorized version with progress bar for better performance.
    """
    df["file_exists"] = False
    missing_path_indices = []

    # Get valid paths and their indices
    valid_mask = (df["path_image_cutout_hmi"].notna()) | (df["path_image_cutout_mdi"].notna())

    missing_path_indices = df[~valid_mask].index.tolist()
    valid_df = df[valid_mask].copy()

    # Prepare paths for checking
    paths_to_check = []
    indices_to_update = []

    print("Preparing file paths...")
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Preparing paths"):
        if pd.notna(row["path_image_cutout_hmi"]):
            path_value = row["path_image_cutout_hmi"]
        else:
            path_value = row["path_image_cutout_mdi"]

        base_filename = os.path.basename(path_value)
        fits_file_path = os.path.join(data_folder, dataset_folder, "fits", base_filename)

        paths_to_check.append(fits_file_path)
        indices_to_update.append(idx)

    # Check file existence with progress bar
    print("Checking file existence...")
    for i, file_path in enumerate(tqdm(paths_to_check, desc="Checking files")):
        if os.path.exists(file_path):
            df.loc[indices_to_update[i], "file_exists"] = True

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
    print("Unique ARs in Train:", train_df["number"].nunique())
    print("Unique ARs in Validation:", val_df["number"].nunique())
    print("Unique ARs in Test:", test_df["number"].nunique())

    print("\nAR Overlap Check:")
    print("Train vs Val AR Overlap:", len(set(train_ars) & set(val_ars)))
    print("Train vs Test AR Overlap:", len(set(train_ars) & set(test_ars)))
    print("Val vs Test AR Overlap:", len(set(val_ars) & set(test_ars)))

    return train_df, val_df, test_df
