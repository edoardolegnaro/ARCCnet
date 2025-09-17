# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: py_3.11
#     language: python
#     name: python3
# ---

# %%
import logging
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

import arccnet.models.cutouts.hale.config as config
from arccnet.models.cutouts.hale.data_preparation import prepare_dataset
from arccnet.models.cutouts.hale.lightning_data import HaleDataModule
from arccnet.models.cutouts.hale.lightning_model import HaleLightningModel, compute_class_weights

# Set tensor precision for better performance on modern GPUs
torch.set_float32_matmul_precision("medium")

# Try to import RichProgressBar, fallback to TQDMProgressBar
try:
    from pytorch_lightning.callbacks import RichProgressBar

    RICH_AVAILABLE = True
except ImportError:
    from pytorch_lightning.callbacks import TQDMProgressBar

    RICH_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# %%
# Prepare dataset
PROCESSED_DATA_PATH = (
    Path(config.DATA_FOLDER)
    / f"processed_dataset_{config.classes}_{config.N_FOLDS}-splits_rs-{config.RANDOM_STATE}.parquet"
)

if not PROCESSED_DATA_PATH.exists():
    logging.info("Preparing dataset for the first time...")
    df_processed = prepare_dataset(save_path=str(PROCESSED_DATA_PATH))
else:
    logging.info(f"Loading previously prepared dataset: {PROCESSED_DATA_PATH.name}")
    df_processed = pd.read_parquet(str(PROCESSED_DATA_PATH))

logging.info(f"Dataset shape: {df_processed.shape}")
logging.info(f"Label distribution:\n{df_processed['grouped_labels'].value_counts()}")


# %%
def train_single_fold(df: pd.DataFrame, fold_num: int = 1):
    """Train model on a single fold."""

    logging.info(f"Training on fold {fold_num}")

    # Create data module
    data_module = HaleDataModule(df=df, fold_num=fold_num)

    # Setup data to compute class weights
    data_module.setup("fit")
    train_labels = data_module.get_train_labels()
    class_weights = compute_class_weights(train_labels, config.NUM_CLASSES)

    logging.info(f"Train samples: {len(data_module.train_dataset.df)}")
    logging.info(f"Val samples: {len(data_module.val_dataset.df)}")
    logging.info(f"Class weights: {class_weights}")
    logging.info(
        f"Batch size: {config.BATCH_SIZE}, Estimated steps per epoch: {len(data_module.train_dataset.df) // config.BATCH_SIZE}"
    )

    # Create model
    model = HaleLightningModel(
        num_classes=config.NUM_CLASSES,
        learning_rate=config.LEARNING_RATE,
        model_name=config.MODEL_NAME,
        class_weights=class_weights,
    )

    logging.info(f"Model: {config.MODEL_NAME} with {config.NUM_CLASSES} classes")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename=f"hale-fold{fold_num}-{{epoch:02d}}-{{val_acc:.3f}}",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    # Progress bar callback for better visualization
    if RICH_AVAILABLE:
        progress_bar = RichProgressBar(leave=True)
        logging.info("Using RichProgressBar for enhanced progress display")
    else:
        progress_bar = TQDMProgressBar(refresh_rate=10)
        logging.info("Using TQDMProgressBar for progress display")

    # Setup logger (using CSV to avoid TensorBoard/NumPy compatibility issues)
    logger = CSVLogger(
        save_dir="logs",
        name=f"hale_fold_{fold_num}",
    )

    # Create trainer with explicit progress bar settings
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, progress_bar],
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False,  # Set to True for reproducibility but may slow training
    )

    # Train model
    logging.info("Starting training...")
    logging.info(f"Max epochs: {config.MAX_EPOCHS}")
    logging.info(f"Device: {trainer.device_ids if hasattr(trainer, 'device_ids') else 'auto'}")

    trainer.fit(model, data_module)

    # Test model
    logging.info("Starting testing...")
    trainer.test(model, data_module, ckpt_path="best")

    return trainer, model


# %%
def train_all_folds(df: pd.DataFrame):
    """Train model on all folds for cross-validation."""

    results = {}

    for fold_num in range(1, config.N_FOLDS + 1):
        logging.info(f"\n{'=' * 50}")
        logging.info(f"TRAINING FOLD {fold_num}/{config.N_FOLDS}")
        logging.info(f"{'=' * 50}")

        try:
            trainer, model = train_single_fold(df, fold_num)
            results[fold_num] = {
                "trainer": trainer,
                "model": model,
                "best_checkpoint": trainer.checkpoint_callback.best_model_path,
            }
            logging.info(f"Fold {fold_num} completed successfully")

        except Exception as e:
            logging.error(f"Fold {fold_num} failed with error: {e}")
            results[fold_num] = {"error": str(e)}

    return results


# %%
# Train on a single fold first for testing
if __name__ == "__main__":
    # For quick testing, train on fold 1 only
    logging.info("Starting training on fold 1...")
    trainer, model = train_single_fold(df_processed, fold_num=1)

    # Uncomment below to train on all folds
    # logging.info("Starting cross-validation training...")
    # results = train_all_folds(df_processed)
