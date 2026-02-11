# config.py
"""
Configuration constants and hyperparameters for the solar flare classification project.
"""

import os

# =============================================================================
# Model & Training Parameters
# =============================================================================
MODEL_NAME = "resnet18"  # Name of the timm model

BATCH_SIZE = 32
NUM_WORKERS = 8
LEARNING_RATE = 1e-4
MAX_EPOCHS = 500
RANDOM_SEED = 42  # Seed for reproducibility in train, val, test splitting
PATIENCE = 10
CHECKPOINT_METRIC = "val_f1"

# Trainer runtime selection
# - "auto": try CUDA first, then fall back based on availability checks in train.py
# - "cpu": force CPU training
# - "gpu": force GPU training (will raise if CUDA cannot be initialized)
ACCELERATOR = "auto"
DEVICES = "auto"
PRECISION = "16-mixed"
CPU_PRECISION = "32-true"
FALLBACK_TO_CPU_ON_CUDA_ERROR = False
HARD_DISABLE_CUDA_ON_FALLBACK = True

# =============================================================================
# Loss Function Parameters
# =============================================================================
LOSS_FUNCTION = "weighted_bce"  # Options: "bce", "focal", "weighted_bce"
# - "bce": Standard Binary Cross Entropy loss
# - "focal": Focal Loss for addressing class imbalance (focuses on hard examples)
# - "weighted_bce": Weighted Binary Cross Entropy with class weights

# Focal Loss Parameters (only used when LOSS_FUNCTION = "focal")
FOCAL_ALPHA = 0.25  # Weight for rare class (positive class in binary classification)
# Typical values: 0.25 (more weight to rare class) to 0.75
FOCAL_GAMMA = 2.0  # Focusing parameter - higher values focus more on hard examples
# Typical values: 1.0 to 5.0, with 2.0 being most common

# Weighted BCE Parameters (only used when LOSS_FUNCTION = "weighted_bce")
USE_CLASS_WEIGHTS = True  # Whether to use class weights for BCE loss

# =============================================================================
# Learning Rate Finder Parameters
# =============================================================================
USE_LR_FINDER = False  # Set to True to enable learning rate finder
LR_FINDER_MIN_LR = 1e-6  # Minimum learning rate to test
LR_FINDER_MAX_LR = 1e-1  # Maximum learning rate to test
LR_FINDER_NUM_TRAINING = 100  # Number of training steps to use for the LR finder
LR_FINDER_AUTO_UPDATE = True  # Automatically update the model with suggested LR

# =============================================================================
# Paths and File Names
# =============================================================================
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")  # Base data directory
FLARES_PARQ = "mag-pit-flare-dataset_1996-01-01_2023-01-01_dev.parq"  # Input flare catalog
CUTOUT_DATASET_FOLDER = "arcnet-v20251017/04_final"

# =============================================================================
# Data Processing Parameters
# =============================================================================
THRESHOLD_CLASS = "M"
TARGET_COLUMN = f"flares_above_{THRESHOLD_CLASS}"

TEST_SIZE = 0.1
VAL_SIZE = 0.2

# =============================================================================
# Image Parameters
# =============================================================================
IMG_TARGET_HEIGHT = 224
IMG_TARGET_WIDTH = 224
# HardTanh normalization parameters
IMG_DIVISOR = 800.0
IMG_MIN_VAL = -1.0
IMG_MAX_VAL = 1.0

# DataLoader parameters
PIN_MEMORY = True
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = 1
DATALOADER_MULTIPROCESSING_CONTEXT = None

# =============================================================================
# Logging Parameters
# =============================================================================
ENABLE_COMET_LOGGING = False  # Set to True to enable Comet logging
COMET_PROJECT_NAME = "ars-flare-classification"
COMET_WORKSPACE = "arcaff"
