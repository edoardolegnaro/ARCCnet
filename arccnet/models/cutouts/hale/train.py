"""
Main training script for Hale classification models.

This script provides a clean, modular interface for training Hale models
either on a single fold or with cross-validation across all folds.

The script has been refactored to use modular components:
- HaleTrainer: Handles single-fold training workflow
- CrossValidationManager: Manages cross-validation experiments
- Evaluation modules: Handle model evaluation and metrics
- Logging utilities: Manage experiment logging

Usage:
    python train.py  # Train according to config.TRAIN_ALL_FOLDS setting
"""

import logging
import warnings

import arccnet.models.cutouts.hale.config as config
from arccnet.models.cutouts.hale.cross_validation import CrossValidationManager
from arccnet.models.cutouts.hale.trainer import HaleTrainer

# Suppress common warnings at the module level
warnings.filterwarnings("ignore", category=UserWarning, message=".*hipBLASLt.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Precision.*not supported by the model summary.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*does not have many workers.*")


def setup_logging() -> None:
    """Set up basic logging configuration."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def prepare_and_encode_dataset():
    """
    Prepare and encode the dataset once.

    Returns:
        pd.DataFrame: Processed dataset with encoded labels
    """
    trainer = HaleTrainer()
    return trainer.prepare_dataset_once()


def train_single_fold_mode(df) -> None:
    """
    Train on a single fold for quick testing.

    Args:
        df: Processed dataset DataFrame
    """
    logging.info("Starting training on fold 1 (single fold mode)...")

    trainer = HaleTrainer()
    trainer_obj, model, test_results = trainer.train_single_fold(df, fold_num=1)

    logging.info("Single fold training completed!")

    # Log final results
    if test_results:
        test_metrics = test_results[0]
        logging.info("=" * 50)
        logging.info("SINGLE FOLD TRAINING COMPLETED")
        logging.info("=" * 50)
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
    results, summary = cv_manager.run_cross_validation(df)

    logging.info("Cross-validation training completed!")

    # Log final summary
    if summary and "metrics_summary" in summary:
        metrics = summary["metrics_summary"]
        logging.info("=" * 50)
        logging.info("CROSS-VALIDATION COMPLETED")
        logging.info("=" * 50)

        if "test_accuracy" in metrics:
            acc_mean = metrics["test_accuracy"].get("mean", 0)
            acc_std = metrics["test_accuracy"].get("std", 0)
            logging.info(f"Mean Test Accuracy: {acc_mean:.4f} ± {acc_std:.4f}")

        if "test_f1" in metrics:
            f1_mean = metrics["test_f1"].get("mean", 0)
            f1_std = metrics["test_f1"].get("std", 0)
            logging.info(f"Mean Test F1: {f1_mean:.4f} ± {f1_std:.4f}")

        if "test_loss" in metrics:
            loss_mean = metrics["test_loss"].get("mean", 0)
            loss_std = metrics["test_loss"].get("std", 0)
            logging.info(f"Mean Test Loss: {loss_mean:.4f} ± {loss_std:.4f}")

        exp_info = summary.get("experiment_info", {})
        total_time = exp_info.get("total_training_time", 0)
        logging.info(f"Total Training Time: {total_time:.1f}s")
        logging.info("=" * 50)


def main() -> None:
    """
    Main execution function.

    Prepares the dataset and runs training according to configuration.
    """
    # Set up logging
    setup_logging()

    # Log configuration info
    logging.info("=" * 60)
    logging.info("HALE CLASSIFICATION TRAINING")
    logging.info("=" * 60)
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
        df_processed = prepare_and_encode_dataset()
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
