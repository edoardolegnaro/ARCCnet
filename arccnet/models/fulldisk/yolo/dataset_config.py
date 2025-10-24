import os
from pathlib import Path

# Data paths
DATA_FOLDER = Path(os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data/"))
DATASET_ROOT = Path("arccnet-v20251017")
DATASET_FOLDER = DATASET_ROOT / "04_final"
DATAFRAME_NAME = "data/region_detection/region_detection_noaa-xarp.parq"

# Output paths: DATA_FOLDER/YOLO/{mag,cont}/{images,labels}/{train,val}
YOLO_OUTPUT_MAG = DATA_FOLDER / "YOLO" / "mag"
YOLO_OUTPUT_CONT = DATA_FOLDER / "YOLO" / "cont"

# Filtering
LONGITUDE_THRESHOLD = 65.0  # Front-side: |longitude| < threshold
MIN_SIZE = 0.03  # Minimum normalized region size
FILTER_SELECTED = True  # Only non-filtered items

# Magnetic class mapping (classes mapped to "None" are excluded)
LABEL_MAPPING = {
    "IA": "None",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": "None",
    "Gamma-Delta": "None",
}

# Dataset split
TRAIN_SPLIT_RATIO = 0.8
TEMPORAL_GAP_DAYS = 14  # Days gap between train and val to prevent temporal leakage
INCLUDE_EMPTY_LABELS = True  # Include images with no valid regions as negative examples

# Image processing
RESIZE_DIM = (1024, 1024)
USE_COLORMAP_MAG = False  # False is grayscale
NUM_CPUS = 30
