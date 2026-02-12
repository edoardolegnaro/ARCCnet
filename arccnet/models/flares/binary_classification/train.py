import os
import logging
from typing import Any

# Import comet_ml before torch/pytorch-lightning for full automatic logging.
try:
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.tuner import Tuner

from arccnet.models.checkpoint_manager import BinaryClassificationCheckpointManager
from arccnet.models.flares import preprocessing
from arccnet.models.flares.binary_classification import config, dataset
from arccnet.models.flares.binary_classification import lighning_modules as lm
from arccnet.models.flares.binary_classification import model

torch.set_float32_matmul_precision("medium")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class CometModelCheckpointCallback(Callback):
    """Log a new best checkpoint artifact to Comet."""

    def __init__(self, comet_logger):
        super().__init__()
        self.comet_logger = comet_logger
        self._last_logged_best_path = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.checkpoint_callback and trainer.checkpoint_callback.best_model_path:
            best_path = trainer.checkpoint_callback.best_model_path
            if best_path == self._last_logged_best_path or not os.path.exists(best_path):
                return
            logger.info("Logging best model to Comet...")
            self.comet_logger.experiment.log_model("best_model", best_path, overwrite=True)
            self._last_logged_best_path = best_path
            logger.info("Best model logged to Comet successfully.")


def load_data():
    """Load, preprocess and split data in train, val and test sets."""
    logger.info("Loading and splitting data (one-time)...")
    train_df, val_df, test_df = dataset.load_and_split_data(
        data_folder=config.DATA_FOLDER,
        df_flares_name=config.FLARES_PARQ,
        dataset_folder=config.CUTOUT_DATASET_FOLDER,
        target_column=f"flares_above_{config.THRESHOLD_CLASS}",
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_SEED,
    )

    # Apply preprocessing (quality filtering, path filtering, longitude filtering)
    # Note: preprocessing is applied before splitting to ensure consistent filtering
    logger.info("Applying data preprocessing (cutouts-based filtering)...")

    # Combine all splits for preprocessing
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=False)

    combined_df_preprocessed = preprocessing.preprocess_flare_data(
        combined_df,
        apply_quality_filter=getattr(config, "APPLY_QUALITY_FILTER", True),
        apply_path_filter=getattr(config, "APPLY_PATH_FILTER", True),
        apply_longitude_filter=getattr(config, "APPLY_LONGITUDE_FILTER", True),
        apply_nan_filter=getattr(config, "APPLY_NAN_FILTER", False),
        max_longitude=getattr(config, "MAX_LONGITUDE", 65.0),
        data_folder=config.DATA_FOLDER,
        dataset_folder=config.CUTOUT_DATASET_FOLDER,
    )

    # Split again after preprocessing to maintain stratification
    preprocessed_train, preprocessed_val, preprocessed_test = dataset.split_preprocessed_data(
        combined_df_preprocessed,
        target_column=f"flares_above_{config.THRESHOLD_CLASS}",
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_SEED,
    )

    logger.info("Data loading, preprocessing and splitting complete.")
    return preprocessed_train, preprocessed_val, preprocessed_test


def resolve_trainer_runtime() -> dict[str, Any]:
    """Validate runtime device availability and return Trainer settings."""
    runtime = {
        "accelerator": config.ACCELERATOR,
        "devices": config.DEVICES,
        "precision": config.PRECISION,
        "force_disable_cuda": False,
    }

    accel = str(runtime["accelerator"]).lower()
    force_gpu = accel in {"gpu", "cuda"}

    if accel in {"auto", "gpu", "cuda"}:
        try:
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() returned False")
            if torch.cuda.device_count() < 1:
                raise RuntimeError("CUDA reports zero visible devices")
            # Force CUDA initialization now to avoid a delayed crash at trainer.fit().
            torch.cuda.get_device_capability(0)
            logger.info(
                "CUDA preflight passed. Visible devices: %d. Primary device: %s",
                torch.cuda.device_count(),
                torch.cuda.get_device_name(0),
            )
        except Exception as exc:
            if force_gpu or not config.FALLBACK_TO_CPU_ON_CUDA_ERROR:
                raise RuntimeError(f"CUDA initialization failed: {exc}") from exc
            logger.warning("CUDA preflight failed (%s). Falling back to CPU runtime.", exc)
            runtime["accelerator"] = "cpu"
            runtime["devices"] = 1
            runtime["precision"] = config.CPU_PRECISION
            runtime["force_disable_cuda"] = bool(getattr(config, "HARD_DISABLE_CUDA_ON_FALLBACK", True))

    if str(runtime["accelerator"]).lower() == "cpu":
        if runtime["devices"] == "auto":
            runtime["devices"] = 1
        if isinstance(runtime["precision"], str) and "16" in runtime["precision"]:
            runtime["precision"] = config.CPU_PRECISION

    return runtime


def hard_disable_cuda_for_process():
    """
    Force CPU-only execution for this process after a CUDA preflight failure.

    Some environments report CUDA as "available" during framework checks but still
    fail later at first real CUDA call; this prevents Lightning from touching CUDA.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    torch.cuda.is_available = lambda: False  # type: ignore[assignment]
    torch.cuda.device_count = lambda: 0  # type: ignore[assignment]
    torch.cuda.get_rng_state_all = lambda: []  # type: ignore[assignment]

    logger.info("CUDA has been disabled for this process after fallback.")


def init_comet_logger():
    if not config.ENABLE_COMET_LOGGING:
        return None

    logger.info("Comet logging is enabled. Initializing CometLogger...")

    common_kwargs = {
        "api_key": os.getenv("COMET_API_KEY"),
        "workspace": config.COMET_WORKSPACE,
    }

    try:
        comet_logger = CometLogger(project=config.COMET_PROJECT_NAME, **common_kwargs)
    except TypeError:
        comet_logger = CometLogger(project_name=config.COMET_PROJECT_NAME, **common_kwargs)

    logger.info("CometLogger initialized.")
    return comet_logger


def main():
    pl.seed_everything(config.RANDOM_SEED, workers=True)
    logger.info("Global seed set to %s.", config.RANDOM_SEED)

    runtime = resolve_trainer_runtime()
    if runtime.get("force_disable_cuda", False):
        hard_disable_cuda_for_process()

    use_pin_memory = bool(config.PIN_MEMORY and str(runtime["accelerator"]).lower() != "cpu")
    logger.info(
        "Trainer runtime config: accelerator=%s, devices=%s, precision=%s, pin_memory=%s",
        runtime["accelerator"],
        runtime["devices"],
        runtime["precision"],
        use_pin_memory,
    )

    train_df, val_df, test_df = load_data()

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
        pin_memory=use_pin_memory,
        persistent_workers=config.PERSISTENT_WORKERS,
        prefetch_factor=config.PREFETCH_FACTOR,
        multiprocessing_context=config.DATALOADER_MULTIPROCESSING_CONTEXT,
    )
    logger.info("DataModule initialized.")

    logger.info("Initializing model '%s'...", config.MODEL_NAME)

    positives = int(train_df[config.TARGET_COLUMN].sum())
    negatives = len(train_df) - positives
    if positives == 0:
        raise ValueError(f"No positive samples found in training set for target '{config.TARGET_COLUMN}'.")

    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32)
    logger.info("Calculated positive class weight: %.2f", pos_weight.item())
    logger.info("Using loss function: %s", config.LOSS_FUNCTION)

    use_pos_weight = pos_weight if config.LOSS_FUNCTION == "weighted_bce" and config.USE_CLASS_WEIGHTS else None

    flare_model = model.FlareClassifier(
        model_name=config.MODEL_NAME,
        num_classes=1,
        in_chans=1,
        learning_rate=config.LEARNING_RATE,
        pretrained=False,
        loss_function=config.LOSS_FUNCTION,
        pos_weight=use_pos_weight,
        focal_alpha=config.FOCAL_ALPHA,
        focal_gamma=config.FOCAL_GAMMA,
    )
    if config.LOSS_FUNCTION == "focal":
        logger.info(
            "Model '%s' initialized with Focal Loss (alpha=%s, gamma=%s).",
            config.MODEL_NAME,
            config.FOCAL_ALPHA,
            config.FOCAL_GAMMA,
        )
    elif config.LOSS_FUNCTION == "weighted_bce":
        if config.USE_CLASS_WEIGHTS:
            logger.info(
                "Model '%s' initialized with Weighted BCE Loss (pos_weight=%.2f).",
                config.MODEL_NAME,
                pos_weight.item(),
            )
        else:
            logger.info(
                "Model '%s' initialized with Weighted BCE Loss (no class weights).",
                config.MODEL_NAME,
            )
    else:
        logger.info("Model '%s' initialized with standard BCE loss.", config.MODEL_NAME)

    logger.info("Setting up callbacks...")
    # Initialize checkpoint manager
    checkpoint_manager = BinaryClassificationCheckpointManager(
        data_folder=config.DATA_FOLDER,
        model_name=config.MODEL_NAME,
        loss_function=config.LOSS_FUNCTION,
    )
    logger.info(f"Checkpoint directory: {checkpoint_manager.get_checkpoint_path()}")

    # Save configuration
    checkpoint_manager.save_config(vars(config))

    checkpoint_callback = checkpoint_manager.get_checkpoint_callback(
        monitor=config.CHECKPOINT_METRIC,
        mode="max",
    )
    early_stopping_callback = EarlyStopping(
        monitor=config.CHECKPOINT_METRIC,
        patience=config.PATIENCE,
        mode="max",
        verbose=True,
    )

    comet_logger = init_comet_logger()
    if comet_logger:
        comet_model_callback = CometModelCheckpointCallback(comet_logger)
        callbacks = [checkpoint_callback, early_stopping_callback, comet_model_callback]
    else:
        callbacks = [checkpoint_callback, early_stopping_callback]

    logger.info("Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=config.MAX_EPOCHS,
        accelerator=runtime["accelerator"],
        devices=runtime["devices"],
        precision=runtime["precision"],
        deterministic=True,
        enable_progress_bar=True,
        logger=comet_logger if comet_logger else False,
        callbacks=callbacks,
    )
    logger.info(
        "Trainer initialized for %d epochs (accelerator=%s, devices=%s, precision=%s).",
        config.MAX_EPOCHS,
        runtime["accelerator"],
        runtime["devices"],
        runtime["precision"],
    )

    if config.USE_LR_FINDER:
        logger.info("Running learning rate finder...")
        tuner = Tuner(trainer)
        lr_finder = tuner.lr_find(
            flare_model,
            datamodule=data_module,
            min_lr=config.LR_FINDER_MIN_LR,
            max_lr=config.LR_FINDER_MAX_LR,
            num_training=config.LR_FINDER_NUM_TRAINING,
            update_attr=False,
        )

        fig = lr_finder.plot(suggest=True)
        suggested_lr = lr_finder.suggestion()
        logger.info("Learning rate finder suggests: %s", suggested_lr)

        os.makedirs("lr_finder_results", exist_ok=True)
        plot_path = "lr_finder_results/lr_finder_plot.png"
        fig.savefig(plot_path)

        if config.ENABLE_COMET_LOGGING and comet_logger:
            logger.info("Logging learning rate finder plot to Comet...")
            comet_logger.experiment.log_figure(figure_name="Learning Rate Finder", figure=fig)
            comet_logger.experiment.log_metric("suggested_learning_rate", suggested_lr)

        plt.close(fig)

        if config.LR_FINDER_AUTO_UPDATE:
            if suggested_lr is not None:
                logger.info("Updating learning rate from %s to %s", flare_model.learning_rate, suggested_lr)
                flare_model.learning_rate = suggested_lr
                flare_model.hparams.learning_rate = suggested_lr
            else:
                logger.warning("LR finder did not return a valid suggestion; keeping current learning rate.")

    logger.info("Starting training (trainer.fit)...")
    trainer.fit(flare_model, data_module)
    logger.info("Training finished.")

    logger.info("Starting testing (trainer.test) using the best checkpoint...")
    test_results = trainer.test(model=flare_model, datamodule=data_module, ckpt_path="best")
    logger.info("Testing finished.")
    logger.info("Test Results: %s", test_results)

    # Save training summary and minimal logging
    training_metadata = {
        "best_epoch": trainer.current_epoch,
        f"best_{config.CHECKPOINT_METRIC}": test_results[0].get(f"test_{config.CHECKPOINT_METRIC}")
        if test_results
        else None,
        "num_epochs_trained": trainer.current_epoch + 1,
        "early_stopping_triggered": early_stopping_callback.wait_count > 0,
        "test_results": test_results[0] if test_results else {},
    }

    checkpoint_manager.save_training_metadata(training_metadata)
    checkpoint_manager.save_minimal_logging(
        best_epoch=trainer.current_epoch,
        best_metric_value=test_results[0].get(f"test_{config.CHECKPOINT_METRIC}") if test_results else None,
        best_metric_name=config.CHECKPOINT_METRIC,
        num_epochs_trained=trainer.current_epoch + 1,
        early_stopping_triggered=early_stopping_callback.wait_count > 0,
        additional_metrics=test_results[0] if test_results else {},
    )

    # Prepare and save classification report from test results
    if test_results:
        classification_report_data = {
            "test_metrics": test_results[0],
            "model_name": config.MODEL_NAME,
            "loss_function": config.LOSS_FUNCTION,
        }
        checkpoint_manager.save_classification_report(classification_report_data)
        logger.info("Classification report saved.")

    logger.info(f"All checkpoints and metadata saved to: {checkpoint_manager.get_checkpoint_path()}")


if __name__ == "__main__":
    main()
