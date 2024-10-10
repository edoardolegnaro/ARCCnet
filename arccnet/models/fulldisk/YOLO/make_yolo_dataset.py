# %% Clean up Dataframe
import os

import pandas as pd
from p_tqdm import p_map

from astropy.time import Time

from arccnet.models.fulldisk.YOLO import utilities as ut

# %% Clean up Dataframe
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../data/")
dataset_folder = "arccnet-fulldisk-dataset-v20240917"
df_name = "fulldisk-detection-catalog-v20240917.parq"

local_path_root = os.path.join(data_folder, dataset_folder)

df = pd.read_parquet(os.path.join(data_folder, dataset_folder, df_name))
df["time"] = df["datetime.jd1"] + df["datetime.jd2"]
times = Time(df["time"], format="jd")
df["datetime"] = pd.to_datetime(times.iso)

selected_df = df[~df["filtered"]]

lon_trshld = 70
front_df = selected_df[(selected_df["longitude"] < lon_trshld) & (selected_df["longitude"] > -lon_trshld)]

min_size = 0.024
img_size_dic = {"MDI": 1024, "HMI": 4096}

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

# %% YOLO Labels
cleaned_df["yolo_label"] = cleaned_df.apply(
    lambda row: ut.to_yolo(
        row["magnetic_class"],
        row["top_right_cutout"],
        row["bottom_left_cutout"],
        img_size_dic.get(row["instrument"]),
        img_size_dic.get(row["instrument"]),
    ),
    axis=1,
)

df_yolo = cleaned_df.groupby("path")["yolo_label"].apply(lambda x: "\n".join(x)).reset_index()

# %% temporal dataset split
split_idx = int(0.8 * len(df_yolo))
train_df = df_yolo[:split_idx]
val_df = df_yolo[split_idx:]

YOLO_root_path = os.path.join(data_folder, "YOLO_dataset")


def process_train(row):
    return ut.process_fits_row(row, local_path_root, YOLO_root_path, "train", resize_dim=(1024, 1024))


def process_val(row):
    return ut.process_fits_row(row, local_path_root, YOLO_root_path, "val", resize_dim=(1024, 1024))


p_map(process_train, [row for _, row in train_df.iterrows()], num_cpus=16)
p_map(process_val, [row for _, row in val_df.iterrows()], num_cpus=16)
