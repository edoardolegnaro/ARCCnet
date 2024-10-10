# %%
import os

import comet_ml
import pandas as pd
from ultralytics import YOLO

from astropy.time import Time

# %% Clean up Dataframe
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../data/")
dataset_folder = "arccnet-fulldisk-dataset-v20240917"
df_name = "fulldisk-detection-catalog-v20240917.parq"

local_path_root = os.path.join(data_folder, dataset_folder)

df = pd.read_parquet(os.path.join(data_folder, dataset_folder, df_name))
df["time"] = df["datetime.jd1"] + df["datetime.jd2"]
times = Time(df["time"], format="jd")
df["datetime"] = pd.to_datetime(times.iso)

selected_df = df[df["filtered"] is False]

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

# %%
comet_ml.init(project_name="fulldisk-detection-arcaff", workspace="arcaff")

# %%
model = YOLO("yolov8l.pt")  # load a pretrained model

# Define training arguments
train_args = {
    "data": "fulldisk640.yaml",
    "imgsz": 1024,  # Image size
    "batch": 64,
    "epochs": 1000,
    "device": [0],
    "patience": 200,
    "dropout": 0.1,
    "fliplr": 0.5,
}

results = model.train(**train_args)
