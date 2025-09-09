import os

from arccnet.models.labels import LABEL_MAPPING_DICT

classes = "a-b-bg"  # Options: "qs-ia-ar", "ia-ar", "qs-ia", "qs-ia-a-b-bg", "a-b-bg"
label_mapping = LABEL_MAPPING_DICT[classes]

# dataset setup
DATA_FOLDER = os.getenv("ARCAFF_DATA_FOLDER", "/ARCAFF/data")
DATASET_FOLDER = "arccnet-v20250805/04_final"
DF_FILE_NAME = "data/cutout_classification/region_classification.parq"
