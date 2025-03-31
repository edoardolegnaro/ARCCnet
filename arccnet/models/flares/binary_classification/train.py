import os
import logging

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.tuner import Tuner

from arccnet.models.flares.binary_classification import config, dataset
from arccnet.models.flares.binary_classification import lighning_modules as lm
from arccnet.models.flares.binary_classification import model

torch.set_float32_matmul_precision("medium")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CometModelCheckpointCallback(Callback):
    """Custom callback to log model to Comet when best checkpoint is saved."""

    def __init__(self, comet_logger):
        super().__init__()
        self.comet_logger = comet_logger

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Skip if this is not a regular training checkpoint
        if "checkpoint_callback_best_model_path" not in checkpoint:
            return

        # Only log when it's the best model
        if trainer.checkpoint_callback.best_model_path == checkpoint["checkpoint_callback_best_model_path"]:
            logger.info("Logging best model to Comet...")
            self.comet_logger.experiment.log_model(
                "best_model", checkpoint["checkpoint_callback_best_model_path"], overwrite=True
            )
            logger.info("Best model logged to Comet successfully.")


# --- Load data once ---
def load_data():
    """Load and split data in train, val and test sets."""
    logger.info("Loading and splitting data (one-time)...")
    train_df, val_df, test_df = dataset.load_and_split_data(
        data_folder=config.DATA_FOLDER,
        df_flares_name=config.FLARES_PARQ,
        dataset_folder=config.CUTOUT_DATASET_FOLDER,
        target_column=f"flares_above_{config.THRESHOLD_CLASS}",
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=42,
    )
    logger.info("Data loading and splitting complete.")
    return train_df, val_df, test_df


# Load data once at the beginning
train_df, val_df, test_df = load_data()

# ---  Initialize DataModule and Model ---
logger.info("Initializing DataModule...")
data_module = lm.FlareDataModule(
    data_folder=config.DATA_FOLDER,
    dataset_folder=config.CUTOUT_DATASET_FOLDER,
    train_df=train_df,
    val_df=val_df,
    test_df=test_df,
    target_column=config.TARGET_COLUMN,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    img_target_height=config.IMG_TARGET_HEIGHT,
    img_target_width=config.IMG_TARGET_WIDTH,
    img_divisor=config.IMG_DIVISOR,
    img_min_val=config.IMG_MIN_VAL,
    img_max_val=config.IMG_MAX_VAL,
)
logger.info("DataModule initialized.")

logger.info(f"Initializing model '{config.MODEL_NAME}'...")

# Calculate class weights
pos_weight = torch.tensor(
    [(len(train_df) - train_df[config.TARGET_COLUMN].sum()) / train_df[config.TARGET_COLUMN].sum()], dtype=torch.float32
)

logger.info(f"Calculated positive class weight: {pos_weight.item():.2f}")

flare_model = model.FlareClassifier(
    model_name=config.MODEL_NAME,
    num_classes=1,
    in_chans=1,
    learning_rate=config.LEARNING_RATE,
    pretrained=False,
    pos_weight=pos_weight,
)
logger.info(f"Model '{config.MODEL_NAME}' initialized with weighted loss.")

# --- Define Callbacks ---
logger.info("Setting up callbacks...")
checkpoint_callback = ModelCheckpoint(
    monitor=config.CHECKPOINT_METRIC,
    mode="max",
    save_top_k=1,
    filename="best-model-{epoch:02d}-{val_f1:.3f}",  # Check config for metric name if different
)

early_stopping_callback = EarlyStopping(
    monitor=config.CHECKPOINT_METRIC, patience=config.PATIENCE, mode="max", verbose=True  # Log when stopping occurs
)

# Initialize CometLogger if enabled
comet_logger = None
if config.ENABLE_COMET_LOGGING:
    logger.info("Comet logging is enabled. Initializing CometLogger...")
    comet_logger = CometLogger(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=config.COMET_PROJECT_NAME,
        workspace=config.COMET_WORKSPACE,
    )
    logger.info("CometLogger initialized.")

    # Add Comet model checkpoint callback
    comet_model_callback = CometModelCheckpointCallback(comet_logger)
    callbacks = [checkpoint_callback, early_stopping_callback, comet_model_callback]
else:
    callbacks = [checkpoint_callback, early_stopping_callback]

logger.info("Callbacks defined: ModelCheckpoint, EarlyStopping, and CometModelCheckpoint (if enabled).")

# --- Initialize Trainer ---
logger.info("Initializing PyTorch Lightning Trainer...")

trainer = pl.Trainer(
    max_epochs=config.MAX_EPOCHS,
    accelerator="auto",  # Automatically chooses GPU, TPU, CPU etc.
    devices="auto",  # Automatically selects the devices
    precision="16-mixed",  # Enable 16-bit mixed-precision training
    deterministic=False,  # Set True for reproducibility, may impact performance
    enable_progress_bar=True,
    logger=comet_logger if comet_logger else False,  # Use CometLogger if enabled
    callbacks=callbacks,  # Add all callbacks
)

logger.info(f"Trainer initialized for {config.MAX_EPOCHS} epochs with mixed precision ('16-mixed').")

# --- Learning Rate Finder ---
if config.USE_LR_FINDER:
    logger.info("Running learning rate finder...")
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        flare_model,
        data_module,
        min_lr=config.LR_FINDER_MIN_LR,
        max_lr=config.LR_FINDER_MAX_LR,
        num_training=config.LR_FINDER_NUM_TRAINING,
    )

    # Plot and save results
    fig = lr_finder.plot(suggest=True)
    suggested_lr = lr_finder.suggestion()

    logger.info(f"Learning rate finder suggests: {suggested_lr}")

    # Save the plot
    os.makedirs("lr_finder_results", exist_ok=True)
    plot_path = "lr_finder_results/lr_finder_plot.png"
    fig.savefig(plot_path)

    # Log to Comet if enabled
    if config.ENABLE_COMET_LOGGING and comet_logger:
        logger.info("Logging learning rate finder plot to Comet...")
        comet_logger.experiment.log_figure(figure_name="Learning Rate Finder", figure=fig)
        # Also log as a metric
        comet_logger.experiment.log_metric("suggested_learning_rate", suggested_lr)

    plt.close(fig)

    # Update model's learning rate if auto-update is enabled
    if config.LR_FINDER_AUTO_UPDATE:
        logger.info(f"Updating learning rate from {flare_model.learning_rate} to {suggested_lr}")
        flare_model.learning_rate = suggested_lr
        flare_model.hparams.learning_rate = suggested_lr

# --- Run Training ---
logger.info("Starting training (trainer.fit)...")
trainer.fit(flare_model, data_module)
logger.info("Training finished.")

# --- Run Testing ---
logger.info("Starting testing (trainer.test) using the best checkpoint...")
# Load the best checkpoint automatically by trainer using ckpt_path='best'
test_results = trainer.test(model=flare_model, datamodule=data_module, ckpt_path="best")
logger.info("Testing finished.")
# Use logger to report results, converting the list/dict to string for logging
logger.info(f"Test Results: {test_results}")
