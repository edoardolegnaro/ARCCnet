# config.py
"""
Configuration constants and hyperparameters for the solar flare classification project.
"""

import os

# =============================================================================
# Model & Training Parameters
# =============================================================================
MODEL_NAME = "vit_small_patch16_224"  # Name of the timm model

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count() // 2
LEARNING_RATE = 1e-4
MAX_EPOCHS = 500
RANDOM_SEED = 42  # Seed for reproducibility in train, val, test splitting
PATIENCE = 10
CHECKPOINT_METRIC = "val_f1"

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
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data")  # Base data directory
FLARES_PARQ = "mag-pit-flare-dataset_1996-01-01_2023-01-01_dev.parq"  # Input flare catalog
CUTOUT_DATASET_FOLDER = "arccnet-cutout-dataset-v20240715"

# =============================================================================
# Data Processing Parameters
# =============================================================================
THRESHOLD_CLASS = "C"
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

# =============================================================================
# Logging Parameters
# =============================================================================
ENABLE_COMET_LOGGING = True  # Set to True to enable Comet logging
COMET_PROJECT_NAME = "ars-flare-classification"
COMET_WORKSPACE = "arcaff"
