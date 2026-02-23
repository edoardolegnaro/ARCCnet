# Timeseries Training Configuration Guide

## GPU and Training Settings

All GPU selection and training hyperparameters are now centralized in `config.py`.

### GPU Configuration

Edit `/ARCAFF/ARCCnet/arccnet/models/timeseries/config.py`:

```python
# GPU Settings
ACCELERATOR = "gpu"          # "gpu", "cpu", or "auto"
DEVICES = 1                  # Number of devices to use
GPU_ID = 0                   # Which GPU to use (0, 1, 2, etc.)
```

**To use a different GPU:**
- Change `GPU_ID = 0` to `GPU_ID = 1` for GPU 1
- Change `GPU_ID = 0` to `GPU_ID = 2` for GPU 2
- Set to `None` to use all available GPUs

### Learning Rate Finder

```python
# Training settings
FIND_LR = False              # Set to True to run LR finder before training
LR_FIND_MIN = 1e-7           # Minimum learning rate to test
LR_FIND_MAX = 1.0            # Maximum learning rate to test
LR_FIND_NUM_STEPS = 100      # Number of steps for LR finder
```

**To enable LR finder:**
1. Set `FIND_LR = True` in config.py
2. Run training normally: `python3 -m arccnet.models.timeseries.train`
3. Check the suggested LR in the output
4. Update `LEARNING_RATE` in config.py with the suggested value
5. Set `FIND_LR = False` and retrain

### Other Training Hyperparameters

```python
# Optimizer settings
LEARNING_RATE = 1e-4         # Base learning rate
WEIGHT_DECAY = 1e-4          # L2 regularization
BATCH_SIZE = 8               # Batch size

# Training loop
MAX_EPOCHS = 100             # Maximum training epochs
EARLY_STOPPING_PATIENCE = 15 # Stop after N epochs without improvement
GRAD_CLIP_MAX_NORM = 1.0     # Gradient clipping threshold

# Class weights (multiclass only)
CLASS_WEIGHTS = [1.0, 5.0, 15.0]  # Weights for [C, M, X] classes
```

### Model Architecture

```python
# Spatial encoder
MODEL_NAME = "resnet34"           # ResNet backbone
SPATIAL_FEATURE_DIM = 512         # Feature dimension after spatial encoding
PRETRAINED_SPATIAL = True         # Use ImageNet pretrained weights
FREEZE_SPATIAL = False            # Freeze spatial encoder weights

# Temporal transformer
TEMPORAL_NUM_LAYERS = 3           # Number of transformer layers
TEMPORAL_NUM_HEADS = 8            # Number of attention heads
TEMPORAL_DIM_FEEDFORWARD = 2048   # FFN hidden dimension
TEMPORAL_DROPOUT = 0.1            # Dropout in transformer
TEMPORAL_POOLING = "mean"         # Pooling strategy: "mean", "max", "last"

# Prediction head
HIDDEN_DIMS = [256]               # Hidden layer dimensions
DROPOUT = 0.2                     # Dropout in prediction head
```

### Data Configuration

```python
# Data paths
DATA_FOLDER = "/ARCAFF/data"
TIMESERIES_ROOT = "/ARCAFF/data/04_final/data"

# Data splitting
SPLIT_STRATEGY = "noaa"      # "noaa" or "time"
TRAIN_FRAC = 0.7             # Training set fraction
VAL_FRAC = 0.15              # Validation set fraction
SEED = 42                    # Random seed

# Data loading
NUM_WORKERS = 8              # DataLoader workers
PIN_MEMORY = True            # Pin memory for faster GPU transfer
```

### Task Configuration

```python
# Task type
TASK_TYPE = "multiclass"     # "multiclass" or "regression"

# Multiclass: Predict highest flare class
NUM_CLASSES = 3              # C, M, X

# Regression: Predict log flare counts
REGRESSION_TARGETS = 3       # [log(Ca+1), log(Ma+1), log(Xa+1)]
```

## Usage Examples

### Example 1: Use GPU 1 instead of GPU 0

Edit `config.py`:
```python
GPU_ID = 1
```

### Example 2: Find optimal learning rate

Edit `config.py`:
```python
FIND_LR = True
```

Run training:
```bash
python3 -m arccnet.models.timeseries.train
```

Check output for suggested LR, then update config:
```python
LEARNING_RATE = 3e-4  # Use suggested value
FIND_LR = False       # Disable finder
```

### Example 3: Train with larger batch size and more workers

Edit `config.py`:
```python
BATCH_SIZE = 16
NUM_WORKERS = 16
```

### Example 4: Reduce model size for faster training

Edit `config.py`:
```python
MODEL_NAME = "resnet18"           # Smaller backbone
SPATIAL_FEATURE_DIM = 256         # Smaller features
TEMPORAL_NUM_LAYERS = 2           # Fewer layers
HIDDEN_DIMS = [128]               # Smaller head
```

### Example 5: Train regression model instead of classification

Edit `config.py`:
```python
TASK_TYPE = "regression"
```

## Command Line Options

While most settings are in `config.py`, you can override some via command line:

```bash
# Use different data directory
python3 -m arccnet.models.timeseries.train --data_root /path/to/data

# Use existing manifest
python3 -m arccnet.models.timeseries.train --manifest_path /path/to/manifest.parq

# Save outputs to different directory
python3 -m arccnet.models.timeseries.train --output_dir /path/to/outputs

# Override task type
python3 -m arccnet.models.timeseries.train --task_type regression
```

## Monitoring Training

### CSV Logs (Default)

Training metrics are logged to:
```
/ARCAFF/ARCCnet/outputs/timeseries/lightning_logs/version_X/metrics.csv
```

### TensorBoard (Optional)

Install TensorBoard:
```bash
pip install tensorboard
```

View logs:
```bash
tensorboard --logdir /ARCAFF/ARCCnet/outputs/timeseries/lightning_logs
```

### Checkpoints

Checkpoints are saved to:
```
/ARCAFF/data/checkpoints/timeseries/multiclass/YYYY-MM-DD_HH-MM-SS_resnet34_transformer_cross_entropy/
```

Files:
- `best.ckpt` - Best model based on validation metric
- `last.ckpt` - Last epoch checkpoint
- `epoch=X.ckpt` - Periodic checkpoints

## Tips for Best Performance

1. **Start with LR finder** - Set `FIND_LR = True` for first run
2. **Monitor validation metrics** - Watch for overfitting
3. **Use class weights** - Adjust `CLASS_WEIGHTS` based on your dataset distribution
4. **Tune batch size** - Larger batches = more stable but slower training
5. **Use mixed precision** - Already enabled by default for faster training
6. **Adjust early stopping** - Increase `EARLY_STOPPING_PATIENCE` if training is slow to converge
