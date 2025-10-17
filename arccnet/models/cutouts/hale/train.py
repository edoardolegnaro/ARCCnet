import logging
import warnings
from pathlib import Path
from datetime import datetime

import arccnet.models.cutouts.hale.config as config
from arccnet.models.cutouts.hale.cross_validation import CrossValidationManager
from arccnet.models.cutouts.hale.trainer import HaleTrainer

# Suppress common warnings at the module level
warnings.filterwarnings("ignore", category=UserWarning, message=".*hipBLASLt.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Precision.*not supported by the model summary.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have many workers.*")


def log_section(title: str, width: int = 50) -> None:
    """Log a banner-style section header."""
    logging.info("=" * width)
    logging.info(title)
    logging.info("=" * width)


def log_metric_summary(label: str, stats: dict) -> None:
    """Log mean and std stats for a metric dict."""
    mean = stats.get("mean", 0)
    std = stats.get("std", 0)
    logging.info(f"{label}: {mean:.4f} ± {std:.4f}")


def train_single_fold_mode(df) -> None:
    """
    Train on a single fold for quick testing.

    Args:
        df: Processed dataset DataFrame
    """
    logging.info("Starting training on fold 1 (single fold mode)...")

    trainer = HaleTrainer()
    trainer_obj, _, test_results = trainer.train_single_fold(df, fold_num=1)

    logging.info("Single fold training completed!")

    # Log final results
    if test_results:
        test_metrics = test_results[0]
        log_section("SINGLE FOLD TRAINING COMPLETED")
        logging.info(f"Final Test Accuracy: {test_metrics.get('test_acc', 'N/A'):.4f}")
        logging.info(f"Final Test F1: {test_metrics.get('test_f1', 'N/A'):.4f}")
        logging.info(f"Final Test Loss: {test_metrics.get('test_loss', 'N/A'):.4f}")
        logging.info(f"Best checkpoint: {trainer_obj.checkpoint_callback.best_model_path}")
        logging.info("=" * 50)


def train_cross_validation_mode(df) -> None:
    """
    Train on all folds for comprehensive cross-validation.

    Args:
        df: Processed dataset DataFrame
    """
    logging.info("Starting cross-validation training on all folds...")

    cv_manager = CrossValidationManager()
    _, summary = cv_manager.run_cross_validation(df)

    logging.info("Cross-validation training completed!")

    # Log final summary
    if summary and "metrics_summary" in summary:
        metrics = summary["metrics_summary"]
        log_section("CROSS-VALIDATION COMPLETED")

        metric_labels = (
            ("Mean Test Accuracy", "test_accuracy"),
            ("Mean Test F1", "test_f1"),
            ("Mean Test Loss", "test_loss"),
        )
        for label, key in metric_labels:
            if key in metrics:
                log_metric_summary(label, metrics[key])

        exp_info = summary.get("experiment_info", {})
        total_time = exp_info.get("total_training_time", 0)
        logging.info(f"Total Training Time: {total_time:.1f}s")
        logging.info("=" * 50)


def main() -> None:
    """
    Main execution function.

    Prepares the dataset and runs training according to configuration.
    """
    log_dir = Path(getattr(config, "LOG_DIR", Path(__file__).parent / "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"hale_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        force=True,
    )

    # Log configuration info
    log_section("HALE CLASSIFICATION TRAINING", width=60)
    logging.info(f"Model: {config.MODEL_NAME}")
    logging.info(f"Classes: {config.classes}")
    logging.info(f"Number of folds: {config.N_FOLDS}")
    logging.info(f"Batch size: {config.BATCH_SIZE}")
    logging.info(f"Learning rate: {config.LEARNING_RATE}")
    logging.info(f"Max epochs: {config.MAX_EPOCHS}")
    logging.info(f"Train all folds: {config.TRAIN_ALL_FOLDS}")
    logging.info("=" * 60)

    try:
        # Prepare dataset once at runtime
        logging.info("Preparing dataset...")
        df_processed = HaleTrainer().prepare_dataset_once()
        logging.info(f"Dataset prepared successfully. Shape: {df_processed.shape}")

        # Run training based on configuration
        if config.TRAIN_ALL_FOLDS:
            train_cross_validation_mode(df_processed)
        else:
            train_single_fold_mode(df_processed)

    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise

    logging.info("Training script completed successfully!")


if __name__ == "__main__":
    main()
