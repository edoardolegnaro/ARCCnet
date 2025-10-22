import os

from arccnet.models.labels import CLASS_NAMES_DICT, LABEL_MAPPING_DICT

# Dataset selection
classes = "a-b-bg"  # Options: "qs-ia-ar", "ia-ar", "qs-ia", "qs-ia-a-b-bg", "a-b-bg"
class_names = CLASS_NAMES_DICT.get(classes, ["Unknown"])

# Data type selection - choose which data to use for training
# Options: "magnetogram", "continuum", "both"
DATA_TYPE = "magnetogram"

# Model selection
MODEL_NAME = "resnet50"  # Options: "resnet18", "resnet34", "resnet50"

# Training configuration
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50

# Cross-validation setup
N_FOLDS = 8
RANDOM_STATE = 42
TRAIN_ALL_FOLDS = True  # False trains only fold 1 for testing

# Hardware configuration
ACCELERATOR = "gpu"  # Options: "gpu", "cpu"
DEVICES = 1  # Number of devices to use
NUM_WORKERS = 16  # Number of workers for data loading

# Image preprocessing parameters
IMAGE_TARGET_HEIGHT = 200
IMAGE_TARGET_WIDTH = 200
# HardTanh parameters
IMAGE_DIVISOR = 800.0
HARDTANH_MIN_VAL = -1.0
HARDTANH_MAX_VAL = 1.0

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

ENABLE_MODEL_SUMMARY = False  # Whether to show model summary
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")

# Experiment tracking configuration
PROJECT_NAME = f"arcaff-v20250805-{classes}"
ENABLE_COMET = True
ENABLE_TENSORBOARD = True
ENABLE_CSV = True
TENSORBOARD_LOG_HYPERPARAMS = True

# ==============================================================================
# DATASET PARAMETERS
# ==============================================================================

PROCESSED_DATASET_FILENAME = f"processed_dataset_{classes}_{N_FOLDS}-splits_rs-{RANDOM_STATE}.parquet"

# Dataset creation parameters
LONG_LIMIT_DEG = 65  # Longitude limit in degrees
UNDERSAMPLE = False  # Whether to undersample the dataset

# Data paths
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
DATASET_FOLDER = "arccnet-v20251017/04_final"
DF_FILE_NAME = "data/cutout_classification/region_classification.parq"

# Label mapping based on selected classes
label_mapping = LABEL_MAPPING_DICT[classes]
NUM_CLASSES = len(class_names)
