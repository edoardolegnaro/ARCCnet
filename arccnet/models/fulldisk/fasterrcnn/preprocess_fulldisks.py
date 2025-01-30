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

import numpy as np
import pandas as pd
from p_tqdm import p_map
from scipy.ndimage import rotate

from astropy.io import fits
from astropy.time import Time

from arccnet.visualisation import utils as ut_v

img_size_dic = {"MDI": 1024, "HMI": 4096}

# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data/")
dataset_folder = "arccnet-fulldisk-dataset-v20240917"
preprocessed_folder = "faster_rcnn_preprocessed"
df_name = "fulldisk-detection-catalog-v20240917.parq"
local_path_root = os.path.join(data_folder, dataset_folder)

# %%
df = pd.read_parquet(os.path.join(data_folder, dataset_folder, df_name))
df["time"] = df["datetime.jd1"] + df["datetime.jd2"]
times = Time(df["time"], format="jd")
df["datetime"] = pd.to_datetime(times.iso)

selected_df = df[~df["filtered"]]

lon_trshld = 70
front_df = selected_df[(selected_df["longitude"] < lon_trshld) & (selected_df["longitude"] > -lon_trshld)]

min_size = 0.024

cleaned_df = front_df.copy()
for idx, row in cleaned_df.iterrows():
    x_min, y_min = row["bottom_left_cutout"]
    x_max, y_max = row["top_right_cutout"]

    img_sz = img_size_dic.get(row["instrument"])
    width = (x_max - x_min) / img_sz
    height = (y_max - y_min) / img_sz

    cleaned_df.at[idx, "width"] = width
    cleaned_df.at[idx, "height"] = height

cleaned_df = cleaned_df[(cleaned_df["width"] >= min_size) & (cleaned_df["height"] >= min_size)]

# %%
label_mapping = {
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": "None",
    "Gamma-Delta": "None",
}

unique_labels = cleaned_df["magnetic_class"].map(label_mapping).unique()
label_to_index = {label: idx for idx, label in enumerate(unique_labels, start=1)}  # Start from 1

# Update DataFrame
cleaned_df["grouped_label"] = cleaned_df["magnetic_class"].map(label_mapping)
cleaned_df = cleaned_df[cleaned_df["grouped_label"] != "None"].copy()  # Exclude 'None' labels if necessary
cleaned_df["encoded_label"] = cleaned_df["grouped_label"].map(label_to_index)


# %%
def preprocess_FD(row):
    arccnet_path_root = row["path"].split("/fits")[0]
    image_path = row["path"].replace(arccnet_path_root, local_path_root)

    with fits.open(image_path) as img_fit:
        data = img_fit[1].data
        header = img_fit[1].header

    data = np.nan_to_num(data, nan=0.0)
    data = ut_v.hardtanh_transform_npy(data)
    crota2 = header.get("CROTA2", 0)
    data = rotate(data, crota2, reshape=False, mode="constant", cval=0)
    # data = ut_v.pad_resize_normalize(data, target_height=final_size, target_width=final_size)
    filename = image_path.replace(dataset_folder, preprocessed_folder).replace(".fits", ".npy")
    return data, filename


# %%
results = p_map(preprocess_FD, [row for _, row in cleaned_df.iterrows()])

# %%
for data, filename in results:
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.save(filename, data)

cleaned_df.to_parquet(os.path.join(data_folder, preprocessed_folder, "cleaned_fulldisk_df.parq"))
