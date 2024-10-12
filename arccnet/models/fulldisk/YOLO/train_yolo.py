import comet_ml
from ultralytics import YOLO

comet_ml.login(project_name="arcaff-v2-fulldisk-detection-classification", workspace="arcaff")

model = YOLO("yolo11x.pt")  # load a pretrained model


# Define training arguments
train_args = {
    "data": "config.yaml",
    "imgsz": 1024,  # Image size
    "batch": 32,
    "epochs": 10000,
    "device": [0, 1, 2, 3],
    "patience": 500,
    "dropout": 0.25,
    "fliplr": 0.5,
}

results = model.train(**train_args)
