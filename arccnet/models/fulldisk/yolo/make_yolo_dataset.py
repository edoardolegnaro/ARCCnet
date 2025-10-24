"""Generate YOLO datasets from full-disk solar observations (mag/cont)."""

from pathlib import Path

import pandas as pd
from p_tqdm import p_map

from arccnet.models.fulldisk import utils as fd_utils
from arccnet.models.fulldisk.yolo import dataset_config as cfg
from arccnet.models.fulldisk.yolo import yolo_utils as ut

# Load and filter data
df = fd_utils.prepare_fulldisk_dataset(
    cfg.DATA_FOLDER,
    cfg.DATASET_ROOT,
    cfg.DATASET_FOLDER,
    cfg.DATAFRAME_NAME,
    longitude_threshold=cfg.LONGITUDE_THRESHOLD,
    min_size=cfg.MIN_SIZE,
    filter_selected=cfg.FILTER_SELECTED,
)


def check_file_exists(path):
    if pd.isna(path):
        return False
    local_path = path.replace("/mnt/ARCAFF/v0.3.0/", str(cfg.DATA_FOLDER / cfg.DATASET_ROOT) + "/")
    return Path(local_path).exists()


initial_count = len(df)
df["mag_exists"] = df["processed_path_image_mag"].apply(check_file_exists)
df["cont_exists"] = df["processed_path_image_cont"].apply(check_file_exists)
df = df[df["mag_exists"] & df["cont_exists"]].copy()
print(f"File check: kept {len(df)}/{initial_count} ({len(df) / initial_count * 100:.1f}%)")

# Map and encode labels
unique_labels = df["magnetic_class"].map(cfg.LABEL_MAPPING).unique()
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
df["grouped_label"] = df["magnetic_class"].map(cfg.LABEL_MAPPING)
df = df[df["grouped_label"] != "None"].copy()
df["encoded_label"] = df["grouped_label"].map(label_to_index)

# Create YOLO labels
img_size_dic = fd_utils.IMG_SIZE_BY_INSTRUMENT
df["yolo_label"] = df.apply(
    lambda row: ut.to_yolo(
        row["encoded_label"],
        row["top_right_cutout"],
        row["bottom_left_cutout"],
        img_size_dic[row["instrument"]],
        img_size_dic[row["instrument"]],
    ),
    axis=1,
)

# Group by full-disk image
df_yolo = (
    df.groupby("processed_path_image_mag")
    .agg(
        {
            "yolo_label": "\n".join,
            "processed_path_image_cont": "first",  # All rows in a group have the same continuum path
        }
    )
    .reset_index()
)
df_yolo = df_yolo.rename(columns={"processed_path_image_mag": "path_mag", "processed_path_image_cont": "path_cont"})

# Split train/val
split_idx = int(cfg.TRAIN_SPLIT_RATIO * len(df_yolo))
train_df, val_df = df_yolo[:split_idx], df_yolo[split_idx:]

# Process and save
local_root = str(cfg.DATA_FOLDER / cfg.DATASET_ROOT)
mag_root, cont_root = str(cfg.YOLO_OUTPUT_MAG), str(cfg.YOLO_OUTPUT_CONT)


def process(row, split):
    return ut.process_fits_pair(row, local_root, mag_root, cont_root, split, cfg.RESIZE_DIM, cfg.USE_COLORMAP_MAG)


p_map(lambda r: process(r, "train"), [r for _, r in train_df.iterrows()], num_cpus=cfg.NUM_CPUS)
p_map(lambda r: process(r, "val"), [r for _, r in val_df.iterrows()], num_cpus=cfg.NUM_CPUS)
