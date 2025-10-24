import argparse
from pathlib import Path

import comet_ml
import yaml
from ultralytics import YOLO

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Train YOLO on magnetogram or continuum data")
parser.add_argument(
    "--data-type",
    type=str,
    default="mag",
    choices=["mag", "cont"],
    help="Type of data to train on: 'mag' for magnetogram or 'cont' for continuum (default: mag)",
)
parser.add_argument(
    "--device",
    type=str,
    default="",
    help="Device to use for training: '' (auto-detect GPUs), 'cpu', '0', '0,1,2,3', etc. (default: auto-detect)",
)
args = parser.parse_args()

# Load base config and update paths based on data type
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Update paths for the selected data type
data_root = Path("/ARCAFF/data/YOLO") / args.data_type
config["train"] = str(data_root / "images" / "train")
config["val"] = str(data_root / "images" / "val")

# Save temporary config
temp_config_path = f"config_{args.data_type}_temp.yaml"
with open(temp_config_path, "w") as f:
    yaml.dump(config, f)

print(f"Training on {args.data_type.upper()} data")

print(f"  Train: {config['train']}")
print(f"  Val: {config['val']}")
print(f"  Classes: {config.get('nc', 'unknown')}")

comet_ml.login(project_name="arcaff-v20251710", workspace="arcaff")

model = YOLO("yolo11s.pt")  # load a pretrained model


# Define training arguments
train_args = {
    "data": temp_config_path,
    "imgsz": 1024,  # Image size
    "batch": 32,
    "epochs": 10000,
    "device": args.device,
    "patience": 500,
    "dropout": 0.25,
    "fliplr": 0.5,
}

results = model.train(**train_args)
