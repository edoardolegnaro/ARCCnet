import os
import argparse
from pathlib import Path

import comet_ml
import yaml
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

# Ensure Ultralytics writes all auxiliary files to a temporary workspace
# Use a local temp directory that can be excluded from git
TEMP_WORKSPACE = Path(__file__).parent / "temp"
TEMP_WORKSPACE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(TEMP_WORKSPACE))
os.environ.setdefault("ULTRALYTICS_CACHE_DIR", str(TEMP_WORKSPACE / "weights"))

_temp_configs_dir = TEMP_WORKSPACE / "configs"
_temp_runs_dir = TEMP_WORKSPACE / "runs"
_temp_weights_dir = TEMP_WORKSPACE / "weights"
_temp_datasets_dir = TEMP_WORKSPACE / "datasets"

for _directory in (_temp_configs_dir, _temp_runs_dir, _temp_weights_dir, _temp_datasets_dir):
    _directory.mkdir(parents=True, exist_ok=True)

SETTINGS.update(
    {
        "weights_dir": str(_temp_weights_dir),
        "runs_dir": str(_temp_runs_dir),
        "datasets_dir": str(_temp_datasets_dir),
    }
)


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
temp_config_path = _temp_configs_dir / f"config_{args.data_type}_temp.yaml"
with open(temp_config_path, "w") as f:
    yaml.dump(config, f)

print(f"Training on {args.data_type.upper()} data")

print(f"  Train: {config['train']}")
print(f"  Val: {config['val']}")
print(f"  Classes: {config.get('nc', 'unknown')}")
print(f"  Temp workspace: {TEMP_WORKSPACE}")

comet_ml.login(project_name="arcaff-v20251710", workspace="arcaff")

model = YOLO("yolo11l.pt")  # load a pretrained model


# Define training arguments
train_args = {
    "data": str(temp_config_path),
    "imgsz": 1024,  # Image size
    "batch": 32,
    "epochs": 500,
    "device": args.device,
    "patience": 25,
    "dropout": 0.25,
    "fliplr": 0.5,
    "mosaic": 0.0,
    "project": str(_temp_runs_dir),
    "name": f"{args.data_type}_train",
}

results = model.train(**train_args)
