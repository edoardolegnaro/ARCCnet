import logging

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from arccnet.models.flares.binary_classification import config, dataset
from arccnet.models.flares.binary_classification import lighning_modules as lm
from arccnet.models.flares.binary_classification import model

torch.set_float32_matmul_precision("medium")

# Configure logging (already set up well)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- 1. Load and Split Data ---
logger.info("Loading and splitting data...")
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

# --- 2. Initialize DataModule ---
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

# --- 3. Initialize Model ---
logger.info(f"Initializing model '{config.MODEL_NAME}'...")
flare_model = model.FlareClassifier(
    model_name=config.MODEL_NAME, num_classes=1, in_chans=1, learning_rate=config.LEARNING_RATE, pretrained=False
)
logger.info(f"Model '{config.MODEL_NAME}' initialized.")

# --- 4. Define Callbacks ---
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
logger.info("Callbacks defined: ModelCheckpoint and EarlyStopping.")

# --- 5. Initialize Trainer ---
logger.info("Initializing PyTorch Lightning Trainer...")
trainer = pl.Trainer(
    max_epochs=config.MAX_EPOCHS,
    accelerator="auto",  # Automatically chooses GPU, TPU, CPU etc.
    devices="auto",  # Automatically selects the devices
    precision="16-mixed",  # Enable 16-bit mixed-precision training
    deterministic=False,  # Set True for reproducibility, may impact performance
    enable_progress_bar=True,
    logger=False,  # Uses TensorBoardLogger by default if installed
    callbacks=[checkpoint_callback, early_stopping_callback],  # Add callbacks
)
# Use logger instead of print
logger.info(f"Trainer initialized for {config.MAX_EPOCHS} epochs with mixed precision ('16-mixed').")

# --- 6. Run Training ---
logger.info("Starting training (trainer.fit)...")
trainer.fit(flare_model, data_module)
logger.info("Training finished.")

# --- 7. Run Testing ---
logger.info("Starting testing (trainer.test) using the best checkpoint...")
# Load the best checkpoint automatically by trainer using ckpt_path='best'
test_results = trainer.test(model=flare_model, datamodule=data_module, ckpt_path="best")
logger.info("Testing finished.")
# Use logger to report results, converting the list/dict to string for logging
logger.info(f"Test Results: {test_results}")
