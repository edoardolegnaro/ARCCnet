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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from arccnet.visualisation import utils as ut_v
from arccnet.visualisation import EDA_flares_utils as ut_f
import arccnet.models.dataset_utils as ut_d
import seaborn as sns
import logging
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GroupShuffleSplit

sns.set_style("darkgrid")

# %%
pd.set_option("display.max_columns", None)
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../../data")
df_file_name = "mag-pit-flare-dataset_1996-01-01_2023-01-01_dev.parq"
df_flares = pd.read_parquet(os.path.join(data_folder, df_file_name))

# %%
dataset_folder = "arccnet-cutout-dataset-v20240715"
df_file_name = "cutout-mcintosh-catalog-v20240715.parq"
df, AR_df = ut_d.make_dataframe(data_folder, dataset_folder, df_file_name)

# %%
def check_fits_file_existence(df, data_folder, dataset_folder):
    """
    Iterates through rows of a DataFrame, checks if a corresponding FITS file exists,
    and adds a new column indicating the file's existence.

    Args:
        df (pd.DataFrame): The DataFrame containing the paths to FITS files.
        data_folder (str): The base directory where the data is stored.
        dataset_folder (str): The subdirectory within data_folder where the dataset is located.

    Returns:
        pd.DataFrame: The DataFrame with a new column 'file_exists' indicating whether the file exists.
    """

    df['file_exists'] = False  # Initialize the new column with False

    for index, row in df.iterrows():
        # Determine the path key based on the presence of "path_image_cutout_hmi"
        if row["path_image_cutout_hmi"] is not None:
            path_key = "path_image_cutout_hmi"
        elif row["path_image_cutout_mdi"] is not None:
            path_key = "path_image_cutout_mdi"
        else:
            # Handle the case where both path columns are None
            logging.warning(f"Both 'path_image_cutout_hmi' and 'path_image_cutout_mdi' are None for row index {index}. Skipping.")
            continue  # Skip to the next row

        # Check if the path value is None or not a string
        if row[path_key] is None or not isinstance(row[path_key], str):
            print(f"Warning: Invalid path value '{row[path_key]}' in column '{path_key}' for row index {index}. Skipping.")
            continue

        fits_file_path = os.path.join(data_folder, dataset_folder, 'fits', os.path.basename(row[path_key]))
        
        # Check if the file exists
        if os.path.exists(fits_file_path):
            df.loc[index, 'file_exists'] = True

    return df

# %%
df_flares_exists = check_fits_file_existence(df_flares.copy(), data_folder, dataset_folder).copy()
df_flares_exists

# %%
plt.figure(figsize=(6, 6))
ax = sns.countplot(x='file_exists', data=df_flares_exists)
plt.title('File Existence')
plt.xlabel('File Exists Locally')
plt.ylabel('Count')
plt.xticks([0, 1], ['False', 'True'])

# Add counts and percentages to the bars
total = len(df_flares_exists)
for p in ax.patches:
    count = p.get_height()
    percentage = '{:.1f}%'.format(100 * count / total)
    x = p.get_x() + p.get_width() / 2
    y = count + 200 
    ax.annotate(f'{count} ({percentage})', (x, y), ha='center')

plt.show()

# %%
df_flares_data = df_flares_exists[df_flares_exists['file_exists']]
df_flares_data


# %%
def categorize_flare(row, threshold='C'):
    """
    Categorizes a row as 'flares' or 'quiet' based on the flare levels.

    Args:
        row (pd.Series): A row of the dataframe.
        threshold (str): The flare level threshold ('A', 'B', 'C', 'M', 'X').

    Returns:
        str: 'flares' if any flare level is above the threshold, 'quiet' otherwise.
    """
    flare_levels = {'A': 1, 'B': 2, 'C': 3, 'M': 4, 'X': 5}
    if threshold not in flare_levels:
        raise ValueError("Invalid threshold. Must be 'A', 'B', 'C', 'M', or 'X'.")

    threshold_value = flare_levels[threshold]

    for level, value in flare_levels.items():
        if value > threshold_value and not pd.isna(row[level]):
            return 'flares'
    return 'quiet'

# %%
flare_threshold = 'C'
df_flares_data.loc[:, 'flaring_flag'] = df_flares_data.apply(lambda row: categorize_flare(row, threshold=flare_threshold), axis=1)

# %%
plt.figure(figsize=(8, 6))
ax = sns.countplot(x='flaring_flag', data=df_flares_data)
plt.title('Percentage and Count of ARs Flaring')
plt.xlabel('Flaring Status')
plt.ylabel('Count')
plt.xticks([0, 1], ['Quiet', 'Flares'])

total = len(df_flares_data)
for p in ax.patches:
    count = int(p.get_height())
    percentage = '{:.1f}%'.format(100 * count / total)
    x = p.get_x() + p.get_width() / 2
    y = p.get_height()
    ax.annotate(f'{count} ({percentage})', (x, y), ha='center', va='bottom')

plt.show()


# %%
train_df, temp_df = train_test_split(df_flares_data,
                                        test_size=0.3, # Adjust as needed
                                        stratify=df_flares_data['flaring_flag'],
                                        random_state=42) # For reproducibility

# %%
def split_dataframe(df, test_size=0.1, val_size=0.2, random_state=42):
    """
    Split dataframe into train, validation, and test sets while keeping 
    Active Regions (AR) together.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to be split
    test_size : float, optional (default=0.2)
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
    # Ensure test_size + val_size is less than 1
    if test_size + val_size >= 1:
        raise ValueError("Combined test and validation sizes must be less than 1")
    
    # First, split into train+val and test sets
    gss_test = GroupShuffleSplit(n_splits=1, 
                                 test_size=test_size, 
                                 random_state=random_state)
    
    # Get indices for train+val and test sets
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['number']))
    
    train_val_df = df.iloc[train_val_idx]
    test_df = df.iloc[test_idx]
    
    # Now split train+val into train and validation
    # Adjust validation size relative to train+val set
    adjusted_val_size = val_size / (1 - test_size)
    
    gss_val = GroupShuffleSplit(n_splits=1, 
                                test_size=adjusted_val_size, 
                                random_state=random_state)
    
    # Get indices for train and validation sets
    train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df['number']))
    
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    
    # Verification
    print("Unique ARs in Train:", train_df['number'].nunique())
    print("Unique ARs in Validation:", val_df['number'].nunique())
    print("Unique ARs in Test:", test_df['number'].nunique())
    
    # Verify no AR overlaps between sets
    print("\nAR Overlap Check:")
    print("Train vs Val AR Overlap:", 
          len(set(train_df['number']) & set(val_df['number'])))
    print("Train vs Test AR Overlap:", 
          len(set(train_df['number']) & set(test_df['number'])))
    print("Val vs Test AR Overlap:", 
          len(set(val_df['number']) & set(test_df['number'])))
    
    return train_df, val_df, test_df

# %%
train_set, val_set, test_set = split_dataframe(df_flares_data)

# %%
train_set['flaring_flag']

# %%
def plot_ar_flaring_distribution_percentages(train_set, val_set, test_set):
    """
    Create a bar plot showing percentages of quiet and flaring ARs
    across train, validation, and test sets.

    Parameters:
    -----------
    train_set : pandas.DataFrame
        Training dataset
    val_set : pandas.DataFrame
        Validation dataset
    test_set : pandas.DataFrame
        Test dataset
    """
    # Prepare data
    sets = {
        'Train': train_set,
        'Validation': val_set,
        'Test': test_set
    }

    # Create figure with one subplot
    fig, ax2 = plt.subplots(1, 1, figsize=(8, 6))

    # Color palette
    colors = ['#1E90FF', '#FF6347']  # Dodger Blue, Tomato Red

    # Percentages subplot
    percentages_data = []
    for set_name, dataset in sets.items():
        flaring_counts = dataset['flaring_flag'].value_counts()
        total_count = len(dataset)
        quiet_percentage = (flaring_counts.get('quiet', 0) / total_count) * 100 if total_count > 0 else 0
        flaring_percentage = (flaring_counts.get('flares', 0) / total_count) * 100 if total_count > 0 else 0
        percentages_data.append([quiet_percentage, flaring_percentage])

    percentages_data = np.array(percentages_data)

    x = np.arange(len(sets))
    width = 0.35

    ax2.bar(x - width/2, percentages_data[:, 0], width, color=colors[0], label='Quiet')
    ax2.bar(x + width/2, percentages_data[:, 1], width, color=colors[1], label='Flaring')

    ax2.set_title('AR Percentages by Flaring Status')
    ax2.legend()

    ax2.set_xticks(x)
    ax2.set_xticklabels(sets.keys())
    ax2.set_ylabel('Percentage of Active Regions')

    # Add percentage labels on bars
    for i in range(len(sets)):
        ax2.text(x[i] - width/2, percentages_data[i, 0], f'{percentages_data[i, 0]:.1f}%',
                 ha='center', va='bottom')
        ax2.text(x[i] + width/2, percentages_data[i, 1], f'{percentages_data[i, 1]:.1f}%',
                 ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

# %%
plot_ar_flaring_distribution_percentages(train_set, val_set, test_set)

# %%



