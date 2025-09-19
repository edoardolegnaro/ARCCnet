import os

from arccnet.models.labels import LABEL_MAPPING_DICT

# ==============================================================================
# USER INPUT PARAMETERS
# ==============================================================================

# Dataset selection
classes = "a-b-bg"  # Options: "qs-ia-ar", "ia-ar", "qs-ia", "qs-ia-a-b-bg", "a-b-bg"

# Data type selection - choose which data to use for training
# Options: "magnetogram", "continuum", "both"
# Note: Currently only "magnetogram" is fully implemented
DATA_TYPE = "magnetogram"

# Model selection
MODEL_NAME = "resnet18"  # Options: "resnet18", "resnet34", "resnet50"

# Training configuration
BATCH_SIZE = 32  # Batch size for training
LEARNING_RATE = 1e-3  # Learning rate for optimizer
MAX_EPOCHS = 50  # Maximum number of epochs to train

# Cross-validation setup
N_FOLDS = 8  # Number of folds for cross-validation
RANDOM_STATE = 42
TRAIN_ALL_FOLDS = True  # Set to False to train only fold 1 for testing

# Hardware configuration
ACCELERATOR = "gpu"  # Options: "gpu", "cpu"
DEVICES = 1  # Number of devices to use
NUM_WORKERS = 16  # Number of workers for data loading

# ==============================================================================
# PREPROCESSING CONSTANTS - Usually don't need to change these
# ==============================================================================

# Image preprocessing parameters
IMAGE_TARGET_HEIGHT = 200  # Target height for resized images
IMAGE_TARGET_WIDTH = 200  # Target width for resized images
IMAGE_DIVISOR = 3000.0  # Divisor for normalizing image pixel values
HARDTANH_MIN_VAL = -1.0  # Minimum value for hardtanh transformation
HARDTANH_MAX_VAL = 1.0  # Maximum value for hardtanh transformation

# Model architecture parameters
LEAKY_RELU_NEGATIVE_SLOPE = 0.01  # Negative slope for LeakyReLU activation

# Regularization parameters
DROPOUT_RATE = 0.3  # Dropout rate for regularization
WEIGHT_DECAY = 1e-4  # L2 regularization weight decay

# Data augmentation parameters (only applied to training data)
USE_AUGMENTATION = True  # Whether to use data augmentation
ROTATION_DEGREES = 20  # Maximum rotation degrees
HORIZONTAL_FLIP_PROB = 0.5  # Probability of horizontal flip
VERTICAL_FLIP_PROB = 0.5  # Probability of vertical flip
PERSPECTIVE_DISTORTION_SCALE = 0.05  # Distortion scale for perspective transform
PERSPECTIVE_PROB = 0.25  # Probability of perspective transform
AFFINE_TRANSLATE = (0.03, 0.03)  # Translation range for affine transform
AFFINE_SCALE = (0.98, 1.03)  # Scale range for affine transform
AFFINE_SHEAR = 3  # Shear range for affine transform

# Training callbacks and optimization parameters
EARLY_STOPPING_PATIENCE = 8  # Patience for early stopping (reduced from 10)
EARLY_STOPPING_MONITOR = "val_acc"  # Metric to monitor for early stopping
EARLY_STOPPING_MODE = "max"  # Mode for early stopping (max for accuracy)
CHECKPOINT_MONITOR = "val_acc"  # Metric to monitor for checkpointing
LR_SCHEDULER_FACTOR = 0.5  # Factor to reduce learning rate
LR_SCHEDULER_PATIENCE = 5  # Patience for learning rate scheduler
LR_SCHEDULER_MODE = "min"  # Mode for learning rate scheduler (min for loss)
LR_SCHEDULER_MONITOR = "val_acc"  # Metric to monitor for learning rate scheduler (use loss for LR)

# DataLoader optimization parameters
DATALOADER_PREFETCH_FACTOR = 2  # Prefetch factor for data loading
DATALOADER_PIN_MEMORY = True  # Whether to use pinned memory
DATALOADER_PERSISTENT_WORKERS = True  # Whether to use persistent workers
DATALOADER_MULTIPROCESSING_CONTEXT = "spawn"  # Multiprocessing context

# Logging and progress tracking
LOG_EVERY_N_STEPS = 10  # How often to log during training
ENABLE_MODEL_SUMMARY = False  # Whether to show model summary (avoid precision warnings)

# Experiment tracking configuration
PROJECT_NAME = "arccnet-hale"  # Base project name for experiment tracking
ENABLE_COMET = False  # Whether to enable Comet ML logging
ENABLE_TENSORBOARD = True  # Whether to enable TensorBoard logging
ENABLE_CSV = True  # Whether to enable CSV logging
TENSORBOARD_LOG_HYPERPARAMS = True  # Whether to attempt hyperparameter logging to TensorBoard

# ==============================================================================
# DATASET PARAMETERS - Usually don't change unless using different dataset
# ==============================================================================

# Dataset creation parameters
LONG_LIMIT_DEG = 65  # Longitude limit in degrees
UNDERSAMPLE = False  # Whether to undersample the dataset

# Data paths
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
DATASET_FOLDER = "arccnet-v20250805/04_final"
DF_FILE_NAME = "data/cutout_classification/region_classification.parq"

# ==============================================================================
# DERIVED PARAMETERS - Automatically calculated, don't modify
# ==============================================================================

# Label mapping based on selected classes
label_mapping = LABEL_MAPPING_DICT[classes]

# Calculate number of unique classes from the mapping (excluding None)
_unique_classes = set(v for v in label_mapping.values() if v is not None)
NUM_CLASSES = len(_unique_classes)  # Should be 3 for "a-b-bg": Alpha, Beta, Beta-Gamma
