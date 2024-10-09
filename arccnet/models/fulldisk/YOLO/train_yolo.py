# %%
import comet_ml
from ultralytics import YOLO

# %%
comet_ml.init(project_name="fulldisk-detection-arcaff", workspace="arcaff")

# %%
model = YOLO("yolov8l.pt")  # load a pretrained model

# Define training arguments
train_args = {
    "data": "fulldisk640.yaml",
    "imgsz": 1024,  # Image size
    "batch": 64,
    "epochs": 1000,
    "device": [0],
    "patience": 200,
    "dropout": 0.1,
    "fliplr": 0.5,
}

results = model.train(**train_args)
