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
RANDOM_SEED = 42  # Seed for reproducibility in train, val, test splitting
PATIENCE = 5
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
COMET_PROJECT_NAME = 'ars-flare-classification'
COMET_WORKSPACE = 'arcaff'