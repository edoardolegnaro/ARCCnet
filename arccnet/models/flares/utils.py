import os

from sklearn.model_selection import train_test_split

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
                            'path_image_cutout_hmi' and 'path_image_cutout_mdi' were None.
    """

    df["file_exists"] = False
    missing_path_indices = []

    for index, row in df.iterrows():
        path_key = None  # Initialize path_key for the current row
        if "path_image_cutout_hmi" in row and row["path_image_cutout_hmi"] is not None:
            path_key = "path_image_cutout_hmi"
        elif "path_image_cutout_mdi" in row and row["path_image_cutout_mdi"] is not None:
            path_key = "path_image_cutout_mdi"

        if path_key is None:
            missing_path_indices.append(index)
            continue

        path_value = row[path_key]  # At this point, path_key is set to either 'hmi' or 'mdi' column name

        base_filename = os.path.basename(path_value)
        fits_file_path = os.path.join(data_folder, dataset_folder, "fits", base_filename)

        # Check if the constructed file path exists
        if os.path.exists(fits_file_path):
            df.loc[index, "file_exists"] = True

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
