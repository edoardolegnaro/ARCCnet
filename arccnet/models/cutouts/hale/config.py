import os

from arccnet.models.labels import LABEL_MAPPING_DICT

classes = "a-b-bg"  # Options: "qs-ia-ar", "ia-ar", "qs-ia", "qs-ia-a-b-bg", "a-b-bg"
label_mapping = LABEL_MAPPING_DICT[classes]

# dataset cross validation setup
N_FOLDS = 8
RANDOM_STATE = 42

# dataset creation parameters
LONG_LIMIT_DEG = 65
UNDERSAMPLE = False

# data setup
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
DATASET_FOLDER = "arccnet-v20250805/04_final"
DF_FILE_NAME = "data/cutout_classification/region_classification.parq"

# Data type selection - choose which data to use for training
# Options: "magnetogram", "continuum", "both"
# Note: Currently only "magnetogram" is fully implemented
DATA_TYPE = "magnetogram"

# Lightning training parameters
BATCH_SIZE = 32  # Reduced batch size to avoid memory issues
LEARNING_RATE = 1e-3
MAX_EPOCHS = 50
NUM_WORKERS = 4  # Reduced workers to avoid shared memory issues
ACCELERATOR = "gpu"
DEVICES = 1  # Use single GPU to avoid distributed training issues

# Model parameters
MODEL_NAME = "resnet18"
# Calculate number of unique classes from the mapping (excluding None)
_unique_classes = set(v for v in label_mapping.values() if v is not None)
NUM_CLASSES = len(_unique_classes)  # Should be 3: Alpha, Beta, Beta-Gamma
