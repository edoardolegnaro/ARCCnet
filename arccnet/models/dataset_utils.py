import os

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
    # Read the parquet file
    df = pd.read_parquet(os.path.join(data_folder, dataset_folder, file_name))

    # Convert Julian dates to datetime objects
    df["time"] = df["target_time.jd1"] + df["target_time.jd2"]
    times = Time(df["time"], format="jd")
    dates = pd.to_datetime(times.iso)  # Convert to datetime objects
    df["dates"] = dates

    # Remove problematic magnetograms from the dataset
    problematic_quicklooks = [ql.strip() for ql in config.get("magnetograms", "problematic_quicklooks").split(",")]
    mask = df["quicklook_path_mdi"].apply(lambda x: os.path.basename(x) in problematic_quicklooks)
    filtered_df = df[mask]
    df = df[~mask].reset_index(drop=True)

    # Label the data
    df["label"] = np.where(df["magnetic_class"] == "", df["region_type"], df["magnetic_class"])
    df["date_only"] = df["dates"].dt.date

    # Filter AR and IA regions
    AR_df = pd.concat([df[df["region_type"] == "AR"], df[df["region_type"] == "IA"]])

    return df, AR_df, filtered_df


def undersample_group_filter(df, label_mapping, long_limit_deg=60, undersample=True, buffer_percentage=0.1):
    """
    Filter data based on a specified longitude limit, assign 'front' or 'rear' locations, and group labels
    according to a provided mapping. Optionally undersample the majority class.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data to be undersampled, grouped, and filtered.
    label_mapping : dict
        A dictionary mapping original labels to grouped labels.
    long_limit_deg : int, optional
        The longitude limit for filtering to determine 'front' or 'rear' location. Defaults to 60 degrees.
    undersample : bool, optional
        Flag to enable or disable undersampling of the majority class. Defaults to True.
    buffer_percentage : float, optional
        The percentage buffer added to the second-largest class size when undersampling the majority class.
        Defaults to 0.1 (10%).

    Returns
    -------
    pandas.DataFrame
        The modified original dataframe with 'location', 'grouped_labels', and 'encoded_labels' columns added.
    pandas.DataFrame
        The undersampled and grouped dataframe, with rows from the 'rear' location filtered out.

    Notes
    -----
    - This function assigns 'front' or 'rear' location based on a longitude limit.
    - Labels are grouped according to the `label_mapping` provided.
    - If `undersample` is True, the majority class is reduced to the size of the second-largest class,
      plus a specified buffer percentage.
    - The function returns two dataframes: the modified original dataframe and an undersampled version where
      the 'rear' locations are filtered out.

    Examples
    --------
    label_mapping = {'A': 'group1', 'B': 'group1', 'C': 'group2'}
    df, undersampled_df = undersample_group_filter(
         df=my_dataframe,
         label_mapping=label_mapping,
         long_limit_deg=60,
         undersample=True,
        buffer_percentage=0.1
        )
    """
    lonV = np.deg2rad(np.where(df["processed_path_image_hmi"] != "", df["longitude_hmi"], df["longitude_mdi"]))
    condition = (lonV < -np.deg2rad(long_limit_deg)) | (lonV > np.deg2rad(long_limit_deg))
    df_filtered = df[~condition]
    df_rear = df[condition]
    df.loc[df_filtered.index, "location"] = "front"
    df.loc[df_rear.index, "location"] = "rear"

    # Apply label mapping to the dataframe
    df["grouped_labels"] = df["label"].map(label_mapping)
    df["encoded_labels"] = df["grouped_labels"].map(labels.LABEL_TO_INDEX)

    if undersample:
        class_counts = df["grouped_labels"].value_counts()
        majority_class = class_counts.idxmax()
        second_largest_class_count = class_counts.iloc[1]
        n_samples = int(second_largest_class_count * (1 + buffer_percentage))

        # Perform undersampling on the majority class
        df_majority = df[df["grouped_labels"] == majority_class]
        df_majority_undersampled = resample(df_majority, replace=False, n_samples=n_samples, random_state=42)

        df_list = [df[df["grouped_labels"] == label] for label in class_counts.index if label != majority_class]
        df_list.append(df_majority_undersampled)

        df_du = pd.concat(df_list)
    else:
        df_du = df.copy()

    # Filter out rows with 'rear' location
    df_du = df_du[df_du["location"] != "rear"]

    return df, df_du


def split_data(df_du, label_col, group_col, random_state=42):
    """
    Split the data into training, validation, and test sets using stratified group k-fold cross-validation.

    Parameters
    ----------
    df_du : pandas.DataFrame
        The dataframe to be split. It must contain the columns specified by `label_col` and `group_col`.
    label_col : str
        The name of the column to be used for stratification, ensuring balanced class distribution across folds.
    group_col : str
        The name of the column to be used for grouping, ensuring that all instances of a group are in the same fold.
    random_state : int, optional
        The random seed for reproducibility of the splits. Defaults to 42.

    Returns
    -------
    list of tuples
        A list of tuples, each containing the following for each fold:
        - fold : int
            The fold number (1 to n_splits).
        - train_df : pandas.DataFrame
            The training set for the fold.
        - val_df : pandas.DataFrame
            The validation set for the fold.
        - test_df : pandas.DataFrame
            The test set for the fold.

    Notes
    -----
    - The function uses `StratifiedGroupKFold` to perform k-fold cross-validation with both stratification and
      group-wise splits.
    - `label_col` is used to ensure balanced class distributions across folds, while `group_col` ensures that
      all instances of a group remain in the same fold.
    - An inner 10-fold split is performed on the training set to create the test set.

    Examples
    --------
    fold_splits = split_data(
       df_du=my_dataframe,
         label_col='grouped_labels',
         group_col='number',
        random_state=42
     )
    """
    fold_df = []
    inner_fold_choice = [0, 1, 2, 3, 4]
    sgkf = StratifiedGroupKFold(n_splits=5, random_state=random_state, shuffle=True)
    X = df_du

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(df_du, df_du[label_col], df_du[group_col]), 1):
        temp_df = X.iloc[train_idx]
        val_df = X.iloc[val_idx]
        inner_sgkf = StratifiedGroupKFold(n_splits=10)
        inner_splits = list(inner_sgkf.split(temp_df, temp_df[label_col], temp_df[group_col]))
        inner_train_idx, inner_test_idx = inner_splits[inner_fold_choice[fold - 1]]
        train_df = temp_df.iloc[inner_train_idx]
        test_df = temp_df.iloc[inner_test_idx]

        fold_df.append((fold, train_df, val_df, test_df))

    for fold, train_df, val_df, test_df in fold_df:
        X.loc[train_df.index, f"Fold {fold}"] = "train"
        X.loc[val_df.index, f"Fold {fold}"] = "val"
        X.loc[test_df.index, f"Fold {fold}"] = "test"

    return fold_df


def assign_fold_sets(df, fold_df):
    """
    Assign training, validation, and test sets to the dataframe based on fold information.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to be annotated with set information.
    fold_df : list of tuples
        A list containing tuples for each fold. Each tuple consists of:
        - fold : int
            The fold number.
        - train_df : pandas.DataFrame
            The training set for the fold.
        - val_df : pandas.DataFrame
            The validation set for the fold.
        - test_df : pandas.DataFrame
            The test set for the fold.

    Returns
    -------
    pandas.DataFrame
        The original dataframe with an additional 'set' column, which indicates whether a row belongs
        to the training, validation, or test set for each fold.

    Notes
    -----
    - The function iterates through each fold, adding a 'set' column to the dataframe that assigns rows to
    either the 'train', 'val', or 'test' sets based on the information in `fold_df`.

    Examples
    --------
    df = assign_fold_sets(
        df=df,
        fold_df=fold_splits)
    """
    for fold, train_set, val_set, test_set in fold_df:
        df.loc[train_set.index, f"Fold {fold}"] = "train"
        df.loc[val_set.index, f"Fold {fold}"] = "val"
        df.loc[test_set.index, f"Fold {fold}"] = "test"
    return df
