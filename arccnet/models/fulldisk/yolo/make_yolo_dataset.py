"""Generate YOLO datasets from full-disk solar observations (mag/cont)."""

import logging
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import yaml
from p_tqdm import p_map

from arccnet.models.fulldisk import utils as fd_utils
from arccnet.models.fulldisk.yolo import dataset_config as cfg
from arccnet.models.fulldisk.yolo import yolo_utils as ut

print("=" * 80)
print("YOLO DATASET GENERATION")
print("=" * 80)

# Load and filter data
print("\n[1/6] Loading and filtering dataset...")
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
print(f"  File existence check: kept {len(df)}/{initial_count} ({len(df) / initial_count * 100:.1f}%)")

# Store all unique images BEFORE label filtering (for negative examples)
print("\n[2/6] Processing labels and creating YOLO annotations...")
all_images_before_filter = (
    df.groupby("processed_path_image_mag")
    .agg(
        {
            "processed_path_image_cont": "first",
            "datetime": "first",
            "instrument": "first",
        }
    )
    .reset_index()
)
all_images_before_filter = all_images_before_filter.rename(
    columns={"processed_path_image_mag": "path_mag", "processed_path_image_cont": "path_cont"}
)
print(f"  Total unique images (before label filter): {len(all_images_before_filter)}")

# Map and encode labels
df["grouped_label"] = df["magnetic_class"].map(cfg.LABEL_MAPPING)
print("\n  Label distribution (before filtering 'None'):")
for label, count in df["grouped_label"].value_counts().items():
    print(f"    {label}: {count} ({count / len(df) * 100:.1f}%)")

# Filter out "None" labels BEFORE creating the encoding
df_with_labels = df[df["grouped_label"] != "None"].copy()
print(f"\n  After removing 'None' labels: {len(df_with_labels)} regions")

# Get unique labels and sort them for consistent ordering
unique_labels = sorted(df_with_labels["grouped_label"].unique())
label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
df_with_labels["encoded_label"] = df_with_labels["grouped_label"].map(label_to_index)

print("\n  Label mapping (grouped_label -> YOLO class index):")
for label, idx in label_to_index.items():
    count = (df_with_labels["grouped_label"] == label).sum()
    print(f"    {label}: {idx} ({count} regions)")

# Validate bounding boxes
print("\n  Validating bounding boxes...")
img_size_dic = fd_utils.IMG_SIZE_BY_INSTRUMENT


def validate_bbox(row):
    """Validate that bbox is within image bounds and has positive dimensions."""
    x1, y1 = row["bottom_left_cutout"]
    x2, y2 = row["top_right_cutout"]
    img_size = img_size_dic[row["instrument"]]

    # Check bounds
    if x1 < 0 or y1 < 0 or x2 > img_size or y2 > img_size:
        return False
    # Check positive dimensions
    if x2 <= x1 or y2 <= y1:
        return False
    # Check normalized values are in [0, 1]
    x_center = ((x1 + x2) / 2) / img_size
    y_center = ((y1 + y2) / 2) / img_size
    width = (x2 - x1) / img_size
    height = (y2 - y1) / img_size
    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 0 < width <= 1 and 0 < height <= 1):
        return False
    return True


df_with_labels["valid_bbox"] = df_with_labels.apply(validate_bbox, axis=1)
invalid_count = (~df_with_labels["valid_bbox"]).sum()
if invalid_count > 0:
    print(f"  WARNING: {invalid_count} invalid bounding boxes detected and will be removed!")
    df_with_labels = df_with_labels[df_with_labels["valid_bbox"]].copy()
else:
    print(f"  ✓ All {len(df_with_labels)} bounding boxes are valid")

# Create YOLO labels
df_with_labels["yolo_label"] = df_with_labels.apply(
    lambda row: ut.to_yolo(
        row["encoded_label"],
        row["top_right_cutout"],
        row["bottom_left_cutout"],
        img_size_dic[row["instrument"]],
        img_size_dic[row["instrument"]],
    ),
    axis=1,
)

# Group by full-disk image - images WITH labels
df_yolo_with_labels = (
    df_with_labels.groupby("processed_path_image_mag")
    .agg(
        {
            "yolo_label": "\n".join,
            "processed_path_image_cont": "first",
            "datetime": "first",
            "instrument": "first",
        }
    )
    .reset_index()
)
df_yolo_with_labels = df_yolo_with_labels.rename(
    columns={"processed_path_image_mag": "path_mag", "processed_path_image_cont": "path_cont"}
)

# Handle images with NO valid labels (negative examples)
if cfg.INCLUDE_EMPTY_LABELS:
    images_with_labels = set(df_yolo_with_labels["path_mag"])
    images_without_labels = all_images_before_filter[
        ~all_images_before_filter["path_mag"].isin(images_with_labels)
    ].copy()
    images_without_labels["yolo_label"] = ""  # Empty label = no detections

    print(f"\n  Images WITH valid labels: {len(df_yolo_with_labels)}")
    print(f"  Images WITHOUT valid labels (negative examples): {len(images_without_labels)}")

    # Combine
    df_yolo = pd.concat([df_yolo_with_labels, images_without_labels], ignore_index=True)
else:
    print(f"\n  Using only images with labels: {len(df_yolo_with_labels)}")
    df_yolo = df_yolo_with_labels

df_yolo = df_yolo.sort_values("datetime").reset_index(drop=True)
print(f"\n  Total images in dataset: {len(df_yolo)}")

# Split train/val WITH temporal gap
print("\n[3/6] Splitting dataset with temporal gap...")
print(f"  Date range: {df_yolo['datetime'].min()} to {df_yolo['datetime'].max()}")
print(f"  Duration: {(df_yolo['datetime'].max() - df_yolo['datetime'].min()).days} days")

# Calculate split index based on ratio
initial_split_idx = int(cfg.TRAIN_SPLIT_RATIO * len(df_yolo))

# Find the actual split ensuring temporal gap
train_end_date = df_yolo.iloc[initial_split_idx - 1]["datetime"]
temporal_gap = timedelta(days=cfg.TEMPORAL_GAP_DAYS)
val_start_threshold = train_end_date + temporal_gap

# Find first validation image after the gap
val_start_idx = df_yolo[df_yolo["datetime"] >= val_start_threshold].index[0]

# Adjust split
train_df = df_yolo[:val_start_idx].copy()
val_df = df_yolo[val_start_idx:].copy()

actual_train_end = train_df["datetime"].max()
actual_val_start = val_df["datetime"].min()
actual_gap = (actual_val_start - actual_train_end).days

print(f"\n  Train set: {len(train_df)} images ({len(train_df) / len(df_yolo) * 100:.1f}%)")
print(f"    Date range: {train_df['datetime'].min()} to {train_df['datetime'].max()}")
print(f"  Validation set: {len(val_df)} images ({len(val_df) / len(df_yolo) * 100:.1f}%)")
print(f"    Date range: {val_df['datetime'].min()} to {val_df['datetime'].max()}")
print(f"  Temporal gap: {actual_gap} days")

if actual_gap < cfg.TEMPORAL_GAP_DAYS:
    print(f"  WARNING: Actual gap ({actual_gap} days) < requested ({cfg.TEMPORAL_GAP_DAYS} days)")

# Calculate and log statistics
print("\n[4/6] Computing dataset statistics...")

# Label statistics
train_labels = df_with_labels[df_with_labels["processed_path_image_mag"].isin(train_df["path_mag"])]
val_labels = df_with_labels[df_with_labels["processed_path_image_mag"].isin(val_df["path_mag"])]

print("\n  TRAIN SET STATISTICS:")
print(f"    Total images: {len(train_df)}")
print(f"    Images with labels: {len(train_df[train_df['yolo_label'] != ''])}")
print(f"    Images without labels: {len(train_df[train_df['yolo_label'] == ''])}")
print(f"    Total regions: {len(train_labels)}")
print("    Label distribution:")
for label, idx in sorted(label_to_index.items(), key=lambda x: x[1]):
    count = (train_labels["grouped_label"] == label).sum()
    pct = count / len(train_labels) * 100 if len(train_labels) > 0 else 0
    print(f"      Class {idx} ({label}): {count:5d} ({pct:5.1f}%)")

print("\n  VALIDATION SET STATISTICS:")
print(f"    Total images: {len(val_df)}")
print(f"    Images with labels: {len(val_df[val_df['yolo_label'] != ''])}")
print(f"    Images without labels: {len(val_df[val_df['yolo_label'] == ''])}")
print(f"    Total regions: {len(val_labels)}")
print("    Label distribution:")
for label, idx in sorted(label_to_index.items(), key=lambda x: x[1]):
    count = (val_labels["grouped_label"] == label).sum()
    pct = count / len(val_labels) * 100 if len(val_labels) > 0 else 0
    print(f"      Class {idx} ({label}): {count:5d} ({pct:5.1f}%)")


# Class imbalance check

logger = logging.getLogger("yolo.dataset")
logger.info("CLASS IMBALANCE CHECK:")
for label, idx in sorted(label_to_index.items(), key=lambda x: x[1]):
    train_pct = (train_labels["grouped_label"] == label).sum() / len(train_labels) * 100 if len(train_labels) > 0 else 0
    val_pct = (val_labels["grouped_label"] == label).sum() / len(val_labels) * 100 if len(val_labels) > 0 else 0
    diff = abs(train_pct - val_pct)
    status = "CRITICAL" if diff > 10 else "WARNING" if diff > 5 else "OK"
    logger.info(f"[{status}] Class {idx} ({label}): Train {train_pct:.1f}% vs Val {val_pct:.1f}% (Δ {diff:.1f}%)")


# Save label mapping to YAML config
config_data = {
    "names": {idx: label for label, idx in label_to_index.items()},
    "nc": len(label_to_index),
    "train": str(cfg.YOLO_OUTPUT_MAG / "images" / "train"),
    "val": str(cfg.YOLO_OUTPUT_MAG / "images" / "val"),
}


config_path = Path("/ARCAFF/ARCCnet/arccnet/models/fulldisk/yolo/config.yaml")
with open(config_path, "w") as f:
    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
logger.info(f"Saved configuration to: {config_path}")


# Bounding box statistics
logger.info("BOUNDING BOX STATISTICS:")


def compute_bbox_stats(labels_df):
    if len(labels_df) == 0:
        return {k: 0 for k in ["mean_width", "mean_height", "min_width", "min_height", "max_width", "max_height"]}
    widths = [
        (row["top_right_cutout"][0] - row["bottom_left_cutout"][0]) / img_size_dic[row["instrument"]]
        for _, row in labels_df.iterrows()
    ]
    heights = [
        (row["top_right_cutout"][1] - row["bottom_left_cutout"][1]) / img_size_dic[row["instrument"]]
        for _, row in labels_df.iterrows()
    ]

    return {
        "mean_width": np.mean(widths),
        "mean_height": np.mean(heights),
        "median_width": np.median(widths),
        "median_height": np.median(heights),
        "min_width": np.min(widths),
        "min_height": np.min(heights),
        "max_width": np.max(widths),
        "max_height": np.max(heights),
    }


train_bbox_stats = compute_bbox_stats(train_labels)
val_bbox_stats = compute_bbox_stats(val_labels)

print("    Train set:")
print(
    f"      Width:  mean={train_bbox_stats['mean_width']:.4f}, median={train_bbox_stats['median_width']:.4f}, "
    f"range=[{train_bbox_stats['min_width']:.4f}, {train_bbox_stats['max_width']:.4f}]"
)
print(
    f"      Height: mean={train_bbox_stats['mean_height']:.4f}, median={train_bbox_stats['median_height']:.4f}, "
    f"range=[{train_bbox_stats['min_height']:.4f}, {train_bbox_stats['max_height']:.4f}]"
)

print("    Validation set:")
print(
    f"      Width:  mean={val_bbox_stats['mean_width']:.4f}, median={val_bbox_stats['median_width']:.4f}, "
    f"range=[{val_bbox_stats['min_width']:.4f}, {val_bbox_stats['max_width']:.4f}]"
)
print(
    f"      Height: mean={val_bbox_stats['mean_height']:.4f}, median={val_bbox_stats['median_height']:.4f}, "
    f"range=[{val_bbox_stats['min_height']:.4f}, {val_bbox_stats['max_height']:.4f}]"
)

# Process and save
print("\n[5/6] Processing and saving FITS files...")
local_root = str(cfg.DATA_FOLDER / cfg.DATASET_ROOT)
mag_root, cont_root = str(cfg.YOLO_OUTPUT_MAG), str(cfg.YOLO_OUTPUT_CONT)


def process(row, split):
    return ut.process_fits_pair(row, local_root, mag_root, cont_root, split, cfg.RESIZE_DIM, cfg.USE_COLORMAP_MAG)


print(f"  Processing {len(train_df)} train images...")
p_map(lambda r: process(r, "train"), [r for _, r in train_df.iterrows()], num_cpus=cfg.NUM_CPUS)

print(f"  Processing {len(val_df)} validation images...")
p_map(lambda r: process(r, "val"), [r for _, r in val_df.iterrows()], num_cpus=cfg.NUM_CPUS)

print("\n[6/6] Dataset generation complete!")
print("=" * 80)
print("\nSUMMARY:")
print(f"  Total images: {len(df_yolo)}")
print(f"  Train: {len(train_df)} ({len(train_df) / len(df_yolo) * 100:.1f}%)")
print(f"  Validation: {len(val_df)} ({len(val_df) / len(df_yolo) * 100:.1f}%)")
print(f"  Temporal gap: {actual_gap} days")
print(f"  Classes: {len(label_to_index)}")
print("  Output directories:")
print(f"    Magnetogram: {mag_root}")
print(f"    Continuum: {cont_root}")
print(f"  Configuration saved to: {config_path}")
print("=" * 80)
