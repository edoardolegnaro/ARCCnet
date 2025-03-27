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
NUM_WORKERS = 32
LEARNING_RATE = 1e-4
MAX_EPOCHS = 10
RANDOM_SEED = 42  # Seed for reproducibility in splitting
PATIENCE = 5
CHECKPOINT_METRIC = "val_f1"

# =============================================================================
# Paths and File Names
# =============================================================================
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "../../../../../data")  # Base data directory
FLARES_PARQ = "mag-pit-flare-dataset_1996-01-01_2023-01-01_dev.parq"  # Input flare catalog
CUTOUT_DATASET_FOLDER = "arccnet-cutout-dataset-v20240715"  # Subfolder containing FITS cutouts

# =============================================================================
# Data Processing Parameters
# =============================================================================
THRESHOLD_CLASS = "C"  # Threshold for binary classification (>= C)
TARGET_COLUMN = f"flares_above_{THRESHOLD_CLASS}"  # Target label for binary classification

TEST_SIZE = 0.1  # Fraction for the test set
VAL_SIZE = 0.2  # Fraction for the validation set

# =============================================================================
# Image Parameters
# =============================================================================
IMG_TARGET_HEIGHT = 224
IMG_TARGET_WIDTH = 224
IMG_DIVISOR = 800.0  # Divisor for hardtanh normalization
IMG_MIN_VAL = -1.0  # Min value for hardtanh normalization
IMG_MAX_VAL = 1.0  # Max value for hardtanh normalization
