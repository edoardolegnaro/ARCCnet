"""Training entrypoint for binary flare classification."""

import os
import re
import random
import logging
from typing import Any
from contextlib import contextmanager

# Import comet_ml before torch/pytorch-lightning for full automatic logging.
try:
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.tuner import Tuner

from arccnet.models.checkpoint_manager import BinaryClassificationCheckpointManager
from arccnet.models.flares.binary_classification import config, model
from arccnet.models.flares.binary_classification import threshold_tuning as tt
from arccnet.models.flares.common_datamodule import FlareDataModule

torch.set_float32_matmul_precision("medium")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _safe_comet_call(comet_logger: CometLogger | None, action: str, fn: str, *args: Any, **kwargs: Any) -> bool:
    """Call Comet logger methods safely so logging failures never crash a run."""
    if comet_logger is None:
        return False
    experiment = getattr(comet_logger, "experiment", None)
    if experiment is None:
        return False
    method = getattr(experiment, fn, None)
    if method is None:
        return False
    try:
        method(*args, **kwargs)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Comet %s failed: %s", action, exc)
        return False


def _extract_uppercase_config_values() -> dict[str, Any]:
    """Collect serializable, explicit module-level config values."""
    return {
        key: value
        for key, value in vars(config).items()
        if key.isupper() and not key.startswith("_") and not callable(value)
    }


def _build_split_cache_name() -> str:
    """Build deterministic split-cache identifier from explicit preprocessing config."""
    return (
        f"{str(config.FLARES_PARQ).replace('/', '_').replace('.', '_')}_{config.TARGET_COLUMN}"
        f"_seed{config.RANDOM_SEED}"
        f"_q{int(bool(getattr(config, 'APPLY_QUALITY_FILTER', True)))}"
        f"_p{int(bool(getattr(config, 'APPLY_PATH_FILTER', True)))}"
        f"_l{int(bool(getattr(config, 'APPLY_LONGITUDE_FILTER', True)))}"
        f"_n{int(bool(getattr(config, 'APPLY_NAN_FILTER', False)))}"
        f"_ml{str(getattr(config, 'MAX_LONGITUDE', 65.0)).replace('.', 'p')}"
        f"_nt{str(getattr(config, 'NAN_THRESHOLD', 0.05)).replace('.', 'p')}"
    )


def _set_deterministic_seed(seed: int) -> None:
    """Set deterministic random seeds across Python, NumPy, Torch, and Lightning."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    pl.seed_everything(seed, workers=True)
    logger.info("Global seed set to %d.", seed)


def _dataset_metadata(data_module: FlareDataModule) -> dict[str, Any]:
    """Collect dataset metadata from prepared splits for artifact logging."""
    metadata: dict[str, Any] = {
        "data_folder": data_module.data_folder,
        "dataset_folder": data_module.dataset_folder,
        "flares_parquet": data_module.df_flares_name,
        "target_column": data_module.target_column,
        "test_size": data_module.test_size,
        "val_size": data_module.val_size,
        "random_seed": data_module.random_state,
        "split_group_column": "number",
        "image_type": getattr(config, "IMAGE_TYPE", "magnetograms"),
        "preprocessing": {
            "apply_quality_filter": bool(data_module.apply_quality_filter),
            "apply_path_filter": bool(data_module.apply_path_filter),
            "apply_longitude_filter": bool(data_module.apply_longitude_filter),
            "apply_nan_filter": bool(data_module.apply_nan_filter),
            "max_longitude": float(data_module.max_longitude),
            "nan_threshold": float(data_module.nan_threshold),
        },
        "features": {
            "source": "HMI/MDI magnetogram cutouts",
            "input_shape": [1, int(data_module.img_target_height), int(data_module.img_target_width)],
            "hardtanh": {
                "divisor": float(data_module.img_divisor),
                "min_val": float(data_module.img_min_val),
                "max_val": float(data_module.img_max_val),
            },
        },
    }

    if data_module.train_df is not None:
        metadata["split_rows"] = {
            "train": int(len(data_module.train_df)),
            "val": int(len(data_module.val_df)),
            "test": int(len(data_module.test_df)),
        }
        target = data_module.target_column
        metadata["target_distribution"] = {
            "train_positive": int(data_module.train_df[target].sum()),
            "train_negative": int(len(data_module.train_df) - int(data_module.train_df[target].sum())),
            "val_positive": int(data_module.val_df[target].sum()),
            "val_negative": int(len(data_module.val_df) - int(data_module.val_df[target].sum())),
            "test_positive": int(data_module.test_df[target].sum()),
            "test_negative": int(len(data_module.test_df) - int(data_module.test_df[target].sum())),
        }
    return metadata


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries for parameter logging."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_dict(value, prefix=flat_key))
        elif isinstance(value, (list, tuple, set)):
            flat[flat_key] = ",".join(map(str, value))
        else:
            flat[flat_key] = value
    return flat


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
            logged = _safe_comet_call(
                self.comet_logger,
                "model logging",
                "log_model",
                "best_model",
                best_path,
                overwrite=True,
            )
            if logged:
                self._last_logged_best_path = best_path
                logger.info("Best model logged to Comet successfully.")


class CheckpointVerboseCallback(Callback):
    """Print checkpoint improvements on a new line."""

    def __init__(self):
        self.last_best = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.checkpoint_callback and hasattr(trainer.checkpoint_callback, "best_model_score"):
            current = trainer.checkpoint_callback.best_model_score
            if current is not None and current != self.last_best:
                self.last_best = current
                print(f"\n{trainer.checkpoint_callback.monitor} improved to {float(current):.3f}")


def _is_multi_device(devices: Any) -> bool:
    """Return True when trainer devices config resolves to multiple devices."""
    if devices == "auto":
        return False
    if isinstance(devices, int):
        return devices > 1
    if isinstance(devices, (list, tuple)):
        return len(devices) > 1
    return False


def _num_requested_devices(devices: Any) -> int:
    """Return the configured device count (minimum 1)."""
    if devices == "auto":
        return 1
    if isinstance(devices, int):
        return max(1, int(devices))
    if isinstance(devices, (list, tuple)):
        return max(1, len(devices))
    return 1


def _is_distributed_initialized() -> bool:
    """Return True when torch.distributed is available and initialized."""
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _single_device_for_eval(devices: Any) -> Any:
    """Select a single-device setting for rank-zero post-training evaluation."""
    if isinstance(devices, (list, tuple)) and len(devices) > 0:
        return [devices[0]]
    return 1


@contextmanager
def _single_process_env():
    """Temporarily remove distributed env vars for isolated single-process evaluation."""
    dist_keys = (
        "LOCAL_RANK",
        "RANK",
        "WORLD_SIZE",
        "NODE_RANK",
        "GROUP_RANK",
        "ROLE_RANK",
    )
    saved = {k: os.environ.get(k) for k in dist_keys}
    for key in dist_keys:
        os.environ.pop(key, None)
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


def _barrier_if_distributed(trainer: pl.Trainer) -> None:
    """Synchronize all ranks for distributed trainers."""
    if getattr(trainer, "world_size", 1) > 1:
        trainer.strategy.barrier()


def _extract_best_epoch(best_ckpt_path: str, checkpoint_payload: dict[str, Any]) -> int | None:
    """Resolve best epoch from checkpoint payload (fallback: parse checkpoint filename)."""
    epoch_value = checkpoint_payload.get("epoch")
    if isinstance(epoch_value, (int, float)):
        return int(epoch_value)

    match = re.search(r"best-(\d+)-", os.path.basename(best_ckpt_path))
    if match:
        return int(match.group(1))
    return None


def _num_completed_fit_epochs(trainer: pl.Trainer) -> int:
    """Return number of completed fit epochs using Lightning progress trackers."""
    fit_loop = getattr(trainer, "fit_loop", None)
    if fit_loop is not None:
        epoch_progress = getattr(fit_loop, "epoch_progress", None)
        if epoch_progress is not None:
            current = getattr(epoch_progress, "current", None)
            completed = getattr(current, "completed", None) if current is not None else None
            if isinstance(completed, int):
                return completed
            if isinstance(completed, float):
                return int(completed)

    current_epoch = getattr(trainer, "current_epoch", 0)
    if isinstance(current_epoch, int):
        return max(current_epoch, 0)
    if isinstance(current_epoch, float):
        return max(int(current_epoch), 0)
    return 0


def resolve_trainer_runtime() -> dict[str, Any]:
    """Validate runtime device availability and return Trainer settings."""
    runtime = {
        "accelerator": config.ACCELERATOR,
        "devices": config.DEVICES,
        "precision": config.PRECISION,
    }

    accel = str(runtime["accelerator"]).lower()

    if accel in {"gpu", "cuda"}:
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
            raise RuntimeError(f"CUDA initialization failed: {exc}") from exc

    if accel == "auto":
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            runtime["accelerator"] = "gpu"
            logger.info(
                "Auto accelerator selected GPU. Visible devices: %d. Primary device: %s",
                torch.cuda.device_count(),
                torch.cuda.get_device_name(0),
            )
        else:
            runtime["accelerator"] = "cpu"
            runtime["devices"] = 1
            runtime["precision"] = "32-true"
            logger.info("Auto accelerator selected CPU (CUDA unavailable).")

    if str(runtime["accelerator"]).lower() == "cpu":
        devices = runtime["devices"]
        if devices == "auto":
            runtime["devices"] = 1
        elif isinstance(devices, int):
            runtime["devices"] = max(1, devices)
        elif isinstance(devices, (list, tuple)):
            runtime["devices"] = max(1, len(devices))
        else:
            runtime["devices"] = 1
        if str(runtime["precision"]).endswith("16-mixed"):
            logger.warning("Mixed precision is not supported on CPU. Falling back to 32-true precision.")
            runtime["precision"] = "32-true"

    return runtime


def init_comet_logger() -> CometLogger | None:
    if not config.ENABLE_COMET_LOGGING:
        return None

    if comet_ml is None:
        logger.warning("Comet logging enabled but `comet_ml` is not installed. Continuing without Comet.")
        return None

    logger.info("Comet logging is enabled. Initializing CometLogger...")

    common_kwargs = {
        "workspace": config.COMET_WORKSPACE,
    }
    api_key = os.getenv("COMET_API_KEY")
    if api_key:
        common_kwargs["api_key"] = api_key

    try:
        comet_logger = CometLogger(project=config.COMET_PROJECT_NAME, **common_kwargs)
    except TypeError:
        comet_logger = CometLogger(project_name=config.COMET_PROJECT_NAME, **common_kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to initialize Comet logger: %s", exc)
        return None

    logger.info("CometLogger initialized.")
    return comet_logger


def main():
    _set_deterministic_seed(config.RANDOM_SEED)
    runtime = resolve_trainer_runtime()

    use_pin_memory = bool(config.PIN_MEMORY and str(runtime["accelerator"]).lower() != "cpu")
    logger.info(
        "Trainer runtime config: accelerator=%s, devices=%s, precision=%s, pin_memory=%s",
        runtime["accelerator"],
        runtime["devices"],
        runtime["precision"],
        use_pin_memory,
    )

    logger.info("Initializing DataModule...")
    data_module = FlareDataModule(
        data_folder=config.DATA_FOLDER,
        dataset_folder=config.CUTOUT_DATASET_FOLDER,
        target_column=config.TARGET_COLUMN,
        df_flares_name=config.FLARES_PARQ,
        test_size=config.TEST_SIZE,
        val_size=config.VAL_SIZE,
        random_state=config.RANDOM_SEED,
        apply_quality_filter=getattr(config, "APPLY_QUALITY_FILTER", True),
        apply_path_filter=getattr(config, "APPLY_PATH_FILTER", True),
        apply_longitude_filter=getattr(config, "APPLY_LONGITUDE_FILTER", True),
        apply_nan_filter=getattr(config, "APPLY_NAN_FILTER", False),
        max_longitude=getattr(config, "MAX_LONGITUDE", 65.0),
        nan_threshold=getattr(config, "NAN_THRESHOLD", 0.05),
        split_cache_name=_build_split_cache_name(),
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
    logger.info("Using loss function: %s", config.LOSS_FUNCTION)
    auto_compute_pos_weight = bool(config.LOSS_FUNCTION == "weighted_bce")
    if auto_compute_pos_weight:
        logger.info("Weighted BCE selected; pos_weight will be computed from training split at fit start.")

    flare_model = model.FlareClassifier(
        model_name=config.MODEL_NAME,
        num_classes=1,
        in_chans=1,
        learning_rate=config.LEARNING_RATE,
        pretrained=False,
        loss_function=config.LOSS_FUNCTION,
        pos_weight=None,
        auto_compute_pos_weight=auto_compute_pos_weight,
        focal_alpha=config.FOCAL_ALPHA,
        focal_gamma=config.FOCAL_GAMMA,
        decision_threshold=config.DEFAULT_DECISION_THRESHOLD,
    )

    checkpoint_manager = BinaryClassificationCheckpointManager(
        data_folder=config.DATA_FOLDER,
        model_name=config.MODEL_NAME,
        loss_function=config.LOSS_FUNCTION,
    )
    logger.info("Checkpoint directory: %s", checkpoint_manager.get_checkpoint_path())
    checkpoint_manager.save_config(_extract_uppercase_config_values())

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
    callbacks = [checkpoint_callback, early_stopping_callback, CheckpointVerboseCallback()]
    if comet_logger is not None:
        callbacks.append(CometModelCheckpointCallback(comet_logger))

    run_parameters = _extract_uppercase_config_values()
    run_parameters.update(_flatten_dict({"runtime": runtime, "dataset": _dataset_metadata(data_module)}))
    _safe_comet_call(comet_logger, "parameter logging", "log_parameters", run_parameters)

    if bool(config.USE_LR_FINDER):
        lr_finder_devices = _single_device_for_eval(runtime["devices"])
        lr_scale = 1.0
        if _is_multi_device(runtime["devices"]) and bool(getattr(config, "LR_FINDER_SCALE_WITH_WORLD_SIZE", True)):
            lr_scale = float(_num_requested_devices(runtime["devices"]))
            logger.info(
                "Running learning rate finder on %s for multi-device training (devices=%s). "
                "Suggested LR will be scaled by %.1f to approximate global batch size.",
                lr_finder_devices,
                runtime["devices"],
                lr_scale,
            )
        else:
            logger.info("Running learning rate finder on a single device.")

        lr_finder_trainer = pl.Trainer(
            accelerator=runtime["accelerator"],
            devices=lr_finder_devices,
            precision=runtime["precision"],
            max_epochs=1,
            deterministic=True,
            enable_progress_bar=True,
            logger=False,
            enable_checkpointing=False,
        )
        tuner = Tuner(lr_finder_trainer)
        lr_finder = tuner.lr_find(flare_model, datamodule=data_module)

        fig = lr_finder.plot(suggest=True)
        suggested_lr = lr_finder.suggestion()
        logger.info("Learning rate finder suggests: %s", suggested_lr)

        if suggested_lr is None:
            applied_lr = float(config.LEARNING_RATE)
            logger.warning(
                "LR finder did not return a suggestion. Falling back to configured LEARNING_RATE=%s.",
                config.LEARNING_RATE,
            )
        else:
            applied_lr = float(suggested_lr) * lr_scale

        flare_model.learning_rate = applied_lr
        flare_model.hparams.learning_rate = applied_lr
        logger.info("Using learning rate %.8g for training.", applied_lr)

        os.makedirs("lr_finder_results", exist_ok=True)
        fig.savefig("lr_finder_results/lr_finder_plot.png")
        _safe_comet_call(comet_logger, "figure logging", "log_figure", "Learning Rate Finder", fig)
        if suggested_lr is not None:
            _safe_comet_call(
                comet_logger, "metric logging", "log_metric", "suggested_learning_rate_raw", float(suggested_lr)
            )
        _safe_comet_call(comet_logger, "metric logging", "log_metric", "learning_rate_used_for_training", applied_lr)
        plt.close(fig)

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

    logger.info("Starting training (trainer.fit).")
    trainer.fit(flare_model, data_module)
    logger.info("Training finished.")

    _barrier_if_distributed(trainer)
    if _is_distributed_initialized():
        torch.distributed.destroy_process_group()

    if trainer.global_rank != 0:
        logger.info("Skipping post-training evaluation and checkpoint export on global rank %s.", trainer.global_rank)
        return

    best_ckpt_path = checkpoint_callback.best_model_path
    if not best_ckpt_path:
        raise RuntimeError("No best checkpoint path found after training.")

    logger.info("Loading best checkpoint weights from: %s", best_ckpt_path)
    best_checkpoint = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    state_dict = best_checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError(f"Missing 'state_dict' in checkpoint: {best_ckpt_path}")
    flare_model.load_state_dict(state_dict)
    best_epoch = _extract_best_epoch(best_ckpt_path, best_checkpoint)
    metadata = _dataset_metadata(data_module)

    default_threshold = float(getattr(config, "DEFAULT_DECISION_THRESHOLD", 0.5))
    selected_threshold = default_threshold
    flare_model.set_decision_threshold(selected_threshold)

    threshold_tuning_report: dict[str, Any] = {
        "enabled": bool(getattr(config, "ENABLE_THRESHOLD_TUNING", True)),
        "objective": getattr(config, "THRESHOLD_TUNING_OBJECTIVE", "f1"),
        "split": getattr(config, "THRESHOLD_TUNING_SPLIT", "val"),
        "search_method": getattr(config, "THRESHOLD_SEARCH_METHOD", "unique_probs_exact"),
        "default_threshold": default_threshold,
        "selected_threshold": selected_threshold,
        "status": "disabled",
        "candidate_count": 0,
        "objective_value": None,
        "baseline_val_metrics": None,
        "tuned_val_metrics": None,
        "tuned_test_metrics": None,
    }

    if threshold_tuning_report["enabled"]:
        split = str(threshold_tuning_report["split"]).lower()
        if split != "val":
            raise ValueError(f"Unsupported THRESHOLD_TUNING_SPLIT='{split}'. Only 'val' is currently supported.")

        logger.info("Collecting validation outputs for threshold tuning.")
        data_module.setup(stage="validate")
        val_loader = data_module.val_dataloader()
        flare_model.to(trainer.strategy.root_device)
        val_probs, val_labels = tt.collect_binary_outputs(flare_model, val_loader, device=trainer.strategy.root_device)

        baseline_val_metrics = tt.compute_binary_metrics(val_probs, val_labels, threshold=default_threshold)
        tuning_result = tt.tune_binary_threshold(
            val_probs,
            val_labels,
            default_threshold=default_threshold,
            objective=str(threshold_tuning_report["objective"]),
            search_method=str(threshold_tuning_report["search_method"]),
        )

        selected_threshold = float(tuning_result.threshold)
        tuned_val_metrics = tt.compute_binary_metrics(val_probs, val_labels, threshold=selected_threshold)
        threshold_tuning_report.update(
            {
                "selected_threshold": selected_threshold,
                "status": tuning_result.status,
                "candidate_count": tuning_result.candidate_count,
                "objective_value": tuning_result.objective_value,
                "baseline_val_metrics": baseline_val_metrics,
                "tuned_val_metrics": tuned_val_metrics,
            }
        )
        logger.info(
            "Threshold tuning complete: status=%s, selected_threshold=%.6f, val_%s=%.6f",
            tuning_result.status,
            selected_threshold,
            tuning_result.objective,
            tuning_result.objective_value,
        )
    else:
        logger.info("Threshold tuning disabled. Using default decision threshold %.6f.", selected_threshold)

    flare_model.set_decision_threshold(selected_threshold)
    objective_name = str(threshold_tuning_report["objective"])
    _safe_comet_call(comet_logger, "metric logging", "log_metric", "decision_threshold_selected", selected_threshold)
    if threshold_tuning_report["baseline_val_metrics"] is not None:
        baseline_metrics = threshold_tuning_report["baseline_val_metrics"]
        if "f1" in baseline_metrics:
            _safe_comet_call(
                comet_logger, "metric logging", "log_metric", "val_f1_default_threshold", baseline_metrics["f1"]
            )
        if "tss" in baseline_metrics:
            _safe_comet_call(
                comet_logger, "metric logging", "log_metric", "val_tss_default_threshold", baseline_metrics["tss"]
            )
        if objective_name in baseline_metrics:
            _safe_comet_call(
                comet_logger,
                "metric logging",
                "log_metric",
                f"val_{objective_name}_default_threshold",
                baseline_metrics[objective_name],
            )
    if threshold_tuning_report["tuned_val_metrics"] is not None:
        tuned_metrics = threshold_tuning_report["tuned_val_metrics"]
        if "f1" in tuned_metrics:
            _safe_comet_call(
                comet_logger, "metric logging", "log_metric", "val_f1_tuned_threshold", tuned_metrics["f1"]
            )
        if "tss" in tuned_metrics:
            _safe_comet_call(
                comet_logger, "metric logging", "log_metric", "val_tss_tuned_threshold", tuned_metrics["tss"]
            )
        if objective_name in tuned_metrics:
            _safe_comet_call(
                comet_logger,
                "metric logging",
                "log_metric",
                f"val_{objective_name}_tuned_threshold",
                tuned_metrics[objective_name],
            )

    logger.info(
        "Starting exact single-device testing with best checkpoint weights and decision threshold %.6f.",
        selected_threshold,
    )
    eval_devices = _single_device_for_eval(runtime["devices"])
    with _single_process_env():
        eval_trainer = pl.Trainer(
            accelerator=runtime["accelerator"],
            devices=eval_devices,
            precision=runtime["precision"],
            deterministic=True,
            enable_progress_bar=True,
            logger=comet_logger if comet_logger else False,
        )
        test_results = eval_trainer.test(model=flare_model, datamodule=data_module, ckpt_path=None)

    if test_results:
        threshold_tuning_report["tuned_test_metrics"] = test_results[0]
        for metric_name, metric_value in test_results[0].items():
            if isinstance(metric_value, (int, float)):
                _safe_comet_call(comet_logger, "metric logging", "log_metric", metric_name, float(metric_value))
    logger.info("Testing finished. Test Results: %s", test_results)

    best_metric_value = None
    if checkpoint_callback.best_model_score is not None:
        best_metric_value = float(checkpoint_callback.best_model_score.item())
    num_epochs_trained = _num_completed_fit_epochs(trainer)
    early_stopping_triggered = bool(getattr(early_stopping_callback, "stopped_epoch", 0) > 0)

    training_metadata = {
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_ckpt_path,
        f"best_{config.CHECKPOINT_METRIC}": best_metric_value,
        "num_epochs_trained": num_epochs_trained,
        "early_stopping_triggered": early_stopping_triggered,
        "decision_threshold": selected_threshold,
        "threshold_tuning_status": threshold_tuning_report["status"],
        "threshold_tuning_candidate_count": threshold_tuning_report["candidate_count"],
        "threshold_tuning_objective_value": threshold_tuning_report["objective_value"],
        "test_results": test_results[0] if test_results else {},
        "dataset_metadata": metadata,
    }
    checkpoint_manager.save_training_metadata(training_metadata)
    checkpoint_manager.save_minimal_logging(
        best_epoch=best_epoch,
        best_metric_value=best_metric_value,
        best_metric_name=config.CHECKPOINT_METRIC,
        num_epochs_trained=num_epochs_trained,
        early_stopping_triggered=early_stopping_triggered,
        additional_metrics={
            **(test_results[0] if test_results else {}),
            "decision_threshold": selected_threshold,
            "threshold_tuning_status": threshold_tuning_report["status"],
        },
    )
    checkpoint_manager.save_threshold_tuning_report(threshold_tuning_report)

    if test_results:
        checkpoint_manager.save_classification_report(
            {
                "test_metrics": test_results[0],
                "model_name": config.MODEL_NAME,
                "loss_function": config.LOSS_FUNCTION,
                "decision_threshold": selected_threshold,
                "threshold_tuning": threshold_tuning_report,
                "dataset_metadata": metadata,
            }
        )
        logger.info("Classification report saved.")

    logger.info("All checkpoints and metadata saved to: %s", checkpoint_manager.get_checkpoint_path())


if __name__ == "__main__":
    main()
