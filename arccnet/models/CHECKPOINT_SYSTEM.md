# Checkpoint Management System

## Overview

The checkpoint management system provides a standardized way to save model checkpoints with comprehensive metadata and logging during training. All checkpoints are organized hierarchically in the data folder and include configuration, training summaries, and classification reports.

## Directory Structure

Checkpoints are organized with the following hierarchy:

```
/ARCAFF/data/
└── checkpoints/
    ├── flares/
    │   ├── binary_classification/
    │   │   ├── 2025-02-12_14-30-45_resnet18_weighted_bce/
    │   │   │   ├── best-epoch02-val_f1-0.847.ckpt
    │   │   │   ├── config.json
    │   │   │   ├── training_summary.json
    │   │   │   ├── classification_report.json
    │   │   │   └── minimal_logging.json
    │   │   └── 2025-02-12_16-45-22_efficientnet_b0_focal/
    │   │       └── ...
    │   └── multiclass/
    │       ├── 2025-02-12_15-10-33_vit_base_patch32_224_focal/
    │       │   └── ...
    │       └── ...
    └── cutouts/
        ├── hale/
        │   └── ...
        └── mcintosh/
            └── ...
```

### Checkpoint Folder Naming Convention

Each checkpoint folder follows this naming pattern:

```
YYYY-MM-DD_HH-MM-SS_MODEL_NAME_LOSS_FUNCTION
```

**Components:**
- `YYYY-MM-DD_HH-MM-SS`: Training start timestamp (UTC)
- `MODEL_NAME`: Name of the model used (e.g., `resnet18`, `vit_base_patch32_224`)
- `LOSS_FUNCTION`: Loss function used (e.g., `weighted_bce`, `focal`, `cross_entropy`)

**Example:** `2025-02-12_14-30-45_resnet18_weighted_bce`

## Checkpoint Contents

Each checkpoint folder contains the following files:

### 1. **Checkpoint File(s)**
- `best-epoch{XX}-val_f1-{metric}.ckpt` - The best model checkpoint during training
- PyTorch Lightning saves only the top-1 model based on the monitored metric

### 2. **config.json**
Complete configuration used for this training run, including:
- Model architecture details
- Hyperparameters (learning rate, batch size, etc.)
- Data processing parameters
- Augmentation settings
- Loss function parameters
- All variables from the config module

### 3. **training_summary.json**
Training metadata and final metrics:
```json
{
  "start_time": "2025-02-12T14:30:45.123456",
  "end_time": "2025-02-12T16:45:22.654321",
  "model_name": "resnet18",
  "loss_function": "weighted_bce",
  "root_name": "flares/binary_classification",
  "best_epoch": 42,
  "best_val_f1": 0.847,
  "num_epochs_trained": 43,
  "early_stopping_triggered": true,
  "test_results": {
    "test_loss": 0.234,
    "test_f1": 0.851,
    "test_accuracy": 0.89,
    "test_precision": 0.88,
    "test_recall": 0.85
  }
}
```

### 4. **classification_report.json**
Detailed test metrics and per-class performance:
```json
{
  "test_metrics": {
    "test_loss": 0.234,
    "test_f1": 0.851,
    "test_accuracy": 0.89,
    "test_precision": 0.88,
    "test_recall": 0.85
  },
  "model_name": "resnet18",
  "loss_function": "weighted_bce",
  "class_names": null
}
```

**For multiclass models:** Includes `class_names` list and per-class metrics if available.

### 5. **minimal_logging.json**
Quick reference logging data for offline inspection:
```json
{
  "timestamp": "2025-02-12T16:45:22.654321",
  "best_epoch": 42,
  "best_val_f1": 0.847,
  "num_epochs_trained": 43,
  "early_stopping_triggered": true
}
```

## Usage

### Binary Classification Models

```python
from arccnet.models.checkpoint_manager import BinaryClassificationCheckpointManager
from arccnet.models.flares.binary_classification import config

# Initialize checkpoint manager
checkpoint_manager = BinaryClassificationCheckpointManager(
    data_folder=config.DATA_FOLDER,
    model_name=config.MODEL_NAME,  # e.g., "resnet18"
    loss_function=config.LOSS_FUNCTION,  # e.g., "weighted_bce"
)

# Get checkpoint callback for PyTorch Lightning
checkpoint_callback = checkpoint_manager.get_checkpoint_callback(
    monitor="val_f1",
    mode="max"
)

# Save configuration
checkpoint_manager.save_config(vars(config))

# After training, save metadata
checkpoint_manager.save_training_metadata({
    "best_epoch": trainer.current_epoch,
    "num_epochs_trained": trainer.current_epoch + 1,
    "test_results": test_results[0] if test_results else {},
})

# Save classification report
checkpoint_manager.save_classification_report({
    "test_metrics": test_results[0],
    "model_name": config.MODEL_NAME,
    "loss_function": config.LOSS_FUNCTION,
})

# Save minimal logging
checkpoint_manager.save_minimal_logging(
    best_epoch=trainer.current_epoch,
    best_metric_value=best_f1_score,
    best_metric_name="val_f1",
    num_epochs_trained=trainer.current_epoch + 1,
)
```

### Multiclass Models

```python
from arccnet.models.checkpoint_manager import MulticlassFlareCheckpointManager
from arccnet.models.flares.multiclass import config

checkpoint_manager = MulticlassFlareCheckpointManager(
    data_folder=config.DATA_FOLDER,
    model_name=config.MODEL_NAME,  # e.g., "vit_base_patch32_224"
    loss_function=config.LOSS_TYPE,  # e.g., "focal"
)

# Same usage pattern as binary classification
checkpoint_callback = checkpoint_manager.get_checkpoint_callback("val_f1", "max")
```

### Custom Models

For custom model types, use the generic `CheckpointManager`:

```python
from arccnet.models.checkpoint_manager import CheckpointManager

checkpoint_manager = CheckpointManager(
    root_name="custom/model_type",  # Your custom path
    data_folder="/ARCAFF/data",
    model_name="my_model",
    loss_function="my_loss",
)
```

## Logging Philosophy

The checkpoint system complements Comet.ml logging:

- **Comet.ml**: Comprehensive, cloud-based experiment tracking with all metrics, plots, and artifacts
- **Local Checkpoints**: Minimal, self-contained snapshot of key training information
  - Available offline
  - Quick reference without external services
  - Backup of critical metadata
  - Easy access to model weights and configuration

All detailed metrics are logged to Comet. The local checkpoint folder serves as a minimal reference and backup.

## Accessing Saved Checkpoints

### Load Training Metadata
```python
import json
from pathlib import Path

checkpoint_dir = Path("/ARCAFF/data/checkpoints/flares/binary_classification/2025-02-12_14-30-45_resnet18_weighted_bce")

# Load training summary
with open(checkpoint_dir / "training_summary.json") as f:
    training_metadata = json.load(f)

# Load configuration
with open(checkpoint_dir / "config.json") as f:
    config_used = json.load(f)

# Load classification report
with open(checkpoint_dir / "classification_report.json") as f:
    report = json.load(f)
```

### Load Model Checkpoint
```python
import pytorch_lightning as pl
from arccnet.models.flares.binary_classification.lighning_modules import FlareModule

# Find the checkpoint file
ckpt_file = list(checkpoint_dir.glob("best-*.ckpt"))[0]

# Load model
model = FlareModule.load_from_checkpoint(ckpt_file)
```

## Migration from Old System

The updated training scripts now use the checkpoint manager:

### Binary Classification
- **Before:** Checkpoints saved in current working directory or ad-hoc locations
- **After:** Saved in `/ARCAFF/data/checkpoints/flares/binary_classification/TIMESTAMP_MODEL_LOSS/`

### Multiclass
- **Before:** Saved in `trained_models/multiclass/`
- **After:** Saved in `/ARCAFF/data/checkpoints/flares/multiclass/TIMESTAMP_MODEL_LOSS/`

### Cutouts (Hale/McIntosh)
- **Before:** Saved in various locations (logs/, etc.)
- **After:** Saved in `/ARCAFF/data/checkpoints/cutouts/{hale,mcintosh}/TIMESTAMP_MODEL_LOSS/`

## Benefits

1. **Organized**: Hierarchical structure makes finding experiments easy
2. **Self-documenting**: Folder names clearly indicate what was trained
3. **Complete metadata**: Configuration and results stored together with model
4. **Offline access**: No need to query Comet for basic information
5. **Reproducibility**: Full config saved for easy retraining with same parameters
6. **Scalability**: Easy to expand with new model types (cutouts, etc.)

## Future Extensions

The system is designed to be extensible. New model types can be added:

```python
class ActiveRegionCheckpointManager(CheckpointManager):
    def __init__(self, data_folder="/ARCAFF/data", model_name="unknown", loss_function="unknown"):
        super().__init__(
            root_name="active_regions/classification",
            data_folder=data_folder,
            model_name=model_name,
            loss_function=loss_function,
        )
```
