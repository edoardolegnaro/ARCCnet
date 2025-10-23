# %%
import os

from arccnet.models.fulldisk.YOLO import utilities as ut

# %%
data_folder = os.getenv("ARCAFF_DATA_FOLDER", "../../../data/")
dataset_folder = "arccnet-fulldisk-dataset-v20240917"
df_name = "fulldisk-detection-catalog-v20240917.parq"

local_path_root = os.path.join(data_folder, dataset_folder)
YOLO_root_path = os.path.join(data_folder, "YOLO_dataset")

# %%
image_name = "hmi.m_720s.20220828_000000_TAI.3.magnetogram.png"
image_path = os.path.join(YOLO_root_path, "images", "train", image_name)

# %%
ut.draw_yolo_labels_on_image(image_path)

# %%
