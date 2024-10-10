import comet_ml
from ultralytics import YOLO

comet_ml.init(project_name="arcaff-v2-fulldisk-detection-classification", workspace="arcaff")

model = YOLO("yolo11n.pt")  # load a pretrained model


# Define training arguments
train_args = {
    "data": "config.yaml",
    "imgsz": 1024,  # Image size
    "batch": 16,
    "epochs": 1000,
    "device": [0],
    "patience": 200,
    "dropout": 0.1,
    "fliplr": 0.5,
}

results = model.train(**train_args)
