"""
Configuration constants and hyperparameters for the multi-class solar flare classification project.
"""

import os

# =============================================================================
# Model & Training Parameters
# =============================================================================
MODEL_NAME = "vit_base_patch32_224"  # Name of the timm model
USE_WEIGHTED_LOSS = True  # Enable class-weighted loss to handle imbalanced classes
LOSS_TYPE = "focal"  # Options: "cross_entropy", "weighted_ce", "focal", "weighted_focal"
FOCAL_ALPHA = 1.0  # Focal loss alpha parameter (class weighting factor)
FOCAL_GAMMA = 2.0  # Focal loss gamma parameter (focusing parameter, higher = more focus on hard examples)
DEVICES = [2, 3]  # Number of GPUs to use (e.g., 1, [0, 1], "auto")

BATCH_SIZE = 64
NUM_WORKERS = os.cpu_count() // 2 if os.cpu_count() else 1
LEARNING_RATE = 1e-4
MAX_EPOCHS = 500
RANDOM_SEED = 42  # Seed for reproducibility in train, val, test splitting
PATIENCE = 10
CHECKPOINT_METRIC = "val_f1"

# =============================================================================
# Paths and File Names
# =============================================================================
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data")  # Base data directory
FLARES_PARQ = "mag-pit-flare-dataset_1996-01-01_2023-01-01_dev.parq"  # Input flare catalog
CUTOUT_DATASET_FOLDER = "arccnet-cutout-dataset-v20240715"

# =============================================================================
# Data Processing Parameters
# =============================================================================
CLASSES = ["Quiet", "C", "M_X"]  # Merged M and X classes into M_X
TARGET_COLUMN = "flare_class"

TEST_SIZE = 0.1
VAL_SIZE = 0.2

# Solar limb filtering
FILTER_SOLAR_LIMB = True  # Enable filtering of observations near solar limb
MAX_LONGITUDE = 65.0  # Maximum absolute longitude in degrees (filter out |lon| > 65°)
MAX_LATITUDE = None  # Maximum absolute latitude in degrees (None = no filtering)
LIMB_DISTANCE_THRESHOLD = None  # Alternative: filter by distance from disk center (None = use lon/lat)

# =============================================================================
# Image Parameters
# =============================================================================
IMG_TARGET_HEIGHT = 224
IMG_TARGET_WIDTH = 224
# HardTanh normalization parameters
IMG_DIVISOR = 800.0
IMG_MIN_VAL = -1.0
IMG_MAX_VAL = 1.0

# =============================================================================
# Data Augmentation Parameters
# =============================================================================
# Augmentation parameters for training dataset
USE_AUGMENTATION = True
HORIZONTAL_FLIP_PROB = 0.5
VERTICAL_FLIP_PROB = 0.5
ROTATION_DEGREES = 10

# =============================================================================
# Logging Parameters
# =============================================================================
ENABLE_COMET_LOGGING = True  # Set to True to enable Comet logging
COMET_PROJECT_NAME = "flares-multiclass"
COMET_WORKSPACE = "arcaff"

# Confusion matrix and misclassification logging
LOG_CONFUSION_MATRIX = True
LOG_MISCLASSIFIED_EXAMPLES = True
MAX_MISCLASSIFIED_EXAMPLES = 20  # Maximum number of misclassified examples to log
LOG_CLASSIFICATION_REPORT = True  # Log detailed classification report with per-class metrics

# =============================================================================
# Data Preprocessing Parameters (from cutouts pipeline)
# =============================================================================
# Apply quality filtering: remove low-quality magnetograms based on quality flags
APPLY_QUALITY_FILTER = True

# Apply path filtering: remove rows where both HMI and MDI paths are missing
APPLY_PATH_FILTER = True

# Apply longitude filtering: keep only front-hemisphere observations
# Note: multiclass also has filter_solar_limb() which does similar filtering
APPLY_LONGITUDE_FILTER = False  # Set to False if using filter_solar_limb

# Apply NaN filtering: remove magnetograms with too many NaN values
# WARNING: This is computationally expensive (requires loading all FITS files)
APPLY_NAN_FILTER = False
NAN_THRESHOLD = 0.05  # Maximum allowed fraction of NaN values (5%)
