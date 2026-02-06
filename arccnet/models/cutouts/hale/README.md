# Hale Classification with PyTorch Lightning

This directory contains a minimal PyTorch Lightning implementation for Hale classification (Alpha, Beta, Beta-Gamma).

## Files

- `config.py` - Configuration parameters for training
- `data_preparation.py` - Dataset preparation (existing)
- `dataset.py` - PyTorch dataset class for Hale data
- `lightning_model.py` - Lightning model definition
- `lightning_data.py` - Lightning data module
- `train.py` - Main training script
- `test_setup.py` - Setup verification script

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -e .[models]
   ```

2. **Verify setup:**
   ```bash
   python arccnet/models/cutouts/hale/test_setup.py
   ```

3. **Run training:**
   ```bash
   python arccnet/models/cutouts/hale/train.py
   ```

## Features

- ✅ Simple ResNet-based architecture
- ✅ Cross-validation support (8 folds)
- ✅ Class weight balancing
- ✅ TensorBoard logging
- ✅ Early stopping and checkpointing
- ✅ Proper AR number separation (no data leakage)

## Configuration

Key parameters in `config.py`:
- `BATCH_SIZE`: Batch size for training (default: 32)
- `LEARNING_RATE`: Learning rate (default: 1e-3)
- `MAX_EPOCHS`: Maximum training epochs (default: 50)
- `MODEL_NAME`: ResNet variant (default: "resnet18")

## Dataset Structure

The processed dataset includes:
- `grouped_labels`: Alpha, Beta, Beta-Gamma
- `encoded_labels`: 2, 3, 4 (numeric labels)
- `Fold 1-8`: Cross-validation splits (train/val/test)

## Usage Examples

```python
# Train on single fold
trainer, model = train_single_fold(df_processed, fold_num=1)

# Train on all folds (cross-validation)
results = train_all_folds(df_processed)
```

## Monitoring

Training progress can be monitored via TensorBoard:
```bash
tensorboard --logdir logs
```

## Next Steps

- Add data augmentation transforms
- Implement ensemble methods
- Add confusion matrix logging
- Optimize hyperparameters
