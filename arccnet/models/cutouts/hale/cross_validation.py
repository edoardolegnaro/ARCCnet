"""
Cross-validation manager for Hale classification models.

This module contains the CrossValidationManager class that handles
cross-validation workflows, result aggregation, and summary generation.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import arccnet.models.cutouts.hale.config as config
from arccnet.models.cutouts.hale.logging_utils import (
    log_experiment_summary,
    setup_basic_logging,
    setup_loggers,
)
from arccnet.models.cutouts.hale.trainer import HaleTrainer


class CrossValidationManager:
    """
    Manager class for cross-validation experiments.

    Handles the execution of cross-validation across multiple folds,
    aggregates results, and generates comprehensive summaries.
    """

    def __init__(self, class_names: list[str] | None = None):
        """
        Initialize the CrossValidationManager.

        Args:
            class_names: List of class names (defaults to Alpha, Beta, Beta-Gamma)
        """
        self.class_names = class_names or ["Alpha", "Beta", "Beta-Gamma"]
        self.trainer = HaleTrainer(class_names=self.class_names)
        setup_basic_logging()

    def run_cross_validation(self, df: pd.DataFrame) -> tuple[dict, dict]:
        """
        Run cross-validation training across all folds.

        Args:
            df: Dataset DataFrame

        Returns:
            Tuple of (results dictionary, summary dictionary)
        """
        results = {}
        fold_metrics = []
        fold_confusion_matrices = []
        fold_classification_reports = []

        start_time = datetime.now()
        logging.info(f"Starting cross-validation training at {start_time}")
        logging.info(f"Training {config.N_FOLDS} folds with {config.classes} classes")

        # Create unified loggers for the entire cross-validation run
        main_loggers = self._setup_main_loggers(start_time)

        # Train each fold
        for fold_num in range(1, config.N_FOLDS + 1):
            logging.info(f"\n{'=' * 60}")
            logging.info(f"TRAINING FOLD {fold_num}/{config.N_FOLDS}")
            logging.info(f"{'=' * 60}")

            fold_start_time = datetime.now()

            try:
                fold_result = self._train_single_fold(df, fold_num, main_loggers, fold_start_time)
                results[fold_num] = fold_result

                if fold_result["status"] == "success":
                    fold_metrics.append(self._extract_fold_metrics(fold_result))

                    if fold_result.get("confusion_matrix"):
                        fold_confusion_matrices.append(np.array(fold_result["confusion_matrix"]))
                    if fold_result.get("classification_report"):
                        fold_classification_reports.append(fold_result["classification_report"])

                logging.info(f"Fold {fold_num} completed successfully in {fold_result.get('training_time', 0):.1f}s")

                if "test_accuracy" in fold_result:
                    logging.info(
                        f"Fold {fold_num} Test Accuracy: {fold_result['test_accuracy']:.4f}, "
                        f"F1: {fold_result['test_f1']:.4f}"
                    )

            except Exception as e:
                logging.error(f"Fold {fold_num} failed with error: {e}")
                results[fold_num] = {
                    "fold": fold_num,
                    "status": "error",
                    "error": str(e),
                    "training_time": (datetime.now() - fold_start_time).total_seconds(),
                }

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        # Generate summary
        summary = self._generate_summary(fold_metrics, fold_confusion_matrices, fold_classification_reports, total_time)

        # Log summary to main loggers
        self._log_summary_to_main_loggers(main_loggers, summary)

        # Save results and log summary
        self._save_and_log_results(summary, fold_metrics, results)

        return results, summary

    def _setup_main_loggers(self, start_time: datetime) -> list:
        """Set up unified loggers for the entire cross-validation run."""
        main_loggers = []
        experiment_name = f"hale_{config.classes}_cv_{start_time.strftime('%Y%m%d_%H%M%S')}"

        try:
            main_loggers = setup_loggers(experiment_name)

            # Log experiment configuration to main loggers
            experiment_config = {
                "n_folds": config.N_FOLDS,
                "classes": config.classes,
                "batch_size": config.BATCH_SIZE,
                "learning_rate": config.LEARNING_RATE,
                "max_epochs": config.MAX_EPOCHS,
                "model_name": config.MODEL_NAME,
                "patience": getattr(config, "PATIENCE", config.EARLY_STOPPING_PATIENCE),
                "image_size": getattr(config, "IMAGE_SIZE", "unknown"),
            }

            for logger in main_loggers:
                try:
                    if hasattr(logger, "log_hyperparams"):
                        logger.log_hyperparams(experiment_config)
                    elif hasattr(logger, "experiment") and hasattr(logger.experiment, "log_parameters"):
                        logger.experiment.log_parameters(experiment_config)
                except Exception as e:
                    logging.warning(f"Could not log hyperparameters to {type(logger).__name__}: {e}")

            logging.info(f"Unified experiment loggers created: {[type(logger).__name__ for logger in main_loggers]}")

        except Exception as e:
            logging.warning(f"Could not set up unified loggers: {e}")
            main_loggers = []

        return main_loggers

    def _train_single_fold(
        self, df: pd.DataFrame, fold_num: int, main_loggers: list, fold_start_time: datetime
    ) -> dict:
        """Train a single fold and return results."""
        trainer, model, test_results = self.trainer.train_single_fold(df, fold_num, parent_loggers=main_loggers)

        if test_results and len(test_results) > 0:
            test_metrics = test_results[0]

            # Get confusion matrix and classification report from model
            cm, class_report = model.get_confusion_matrix_and_classification_report(self.class_names)

            fold_result = {
                "fold": fold_num,
                "status": "success",
                "trainer": trainer,
                "model": model,
                "test_loss": test_metrics.get("test_loss", float("nan")),
                "test_accuracy": test_metrics.get("test_acc", float("nan")),
                "test_f1": test_metrics.get("test_f1", float("nan")),
                "best_checkpoint": trainer.checkpoint_callback.best_model_path,
                "training_time": (datetime.now() - fold_start_time).total_seconds(),
                "confusion_matrix": cm.tolist() if cm is not None else None,
                "classification_report": class_report,
            }
        else:
            fold_result = {
                "fold": fold_num,
                "status": "no_test_results",
                "trainer": trainer,
                "model": model,
                "error": "No test results available",
                "training_time": (datetime.now() - fold_start_time).total_seconds(),
            }

        return fold_result

    def _extract_fold_metrics(self, fold_result: dict) -> dict:
        """Extract metrics from fold result for summary."""
        return {
            "fold": fold_result["fold"],
            "test_loss": fold_result["test_loss"],
            "test_accuracy": fold_result["test_accuracy"],
            "test_f1": fold_result["test_f1"],
            "training_time": fold_result["training_time"],
        }

    def _generate_summary(
        self,
        fold_metrics: list[dict],
        fold_confusion_matrices: list[np.ndarray],
        fold_classification_reports: list[dict],
        total_time: float,
    ) -> dict:
        """Generate comprehensive cross-validation summary statistics."""
        summary = {
            "experiment_info": {
                "timestamp": datetime.now().isoformat(),
                "model_name": config.MODEL_NAME,
                "classes": config.classes,
                "n_folds": config.N_FOLDS,
                "total_training_time": total_time,
                "dataset_config": {
                    "batch_size": config.BATCH_SIZE,
                    "learning_rate": config.LEARNING_RATE,
                    "max_epochs": config.MAX_EPOCHS,
                    "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
                    "data_type": config.DATA_TYPE,
                },
            }
        }

        # Calculate metric statistics
        if fold_metrics:
            summary["metrics_summary"] = self._calculate_metric_statistics(fold_metrics)

        # Calculate average confusion matrix
        if fold_confusion_matrices:
            avg_cm = np.mean(fold_confusion_matrices, axis=0)
            summary["average_confusion_matrix"] = avg_cm.tolist()

        # Calculate average per-class metrics
        if fold_classification_reports:
            summary.update(self._calculate_class_metrics(fold_classification_reports))

        return summary

    def _calculate_metric_statistics(self, fold_metrics: list[dict]) -> dict:
        """Calculate statistics for main metrics across folds."""
        valid_metrics = [m for m in fold_metrics if not np.isnan(m.get("test_accuracy", float("nan")))]

        if not valid_metrics:
            return {}

        accuracies = [m["test_accuracy"] for m in valid_metrics]
        f1_scores = [m["test_f1"] for m in valid_metrics]
        losses = [m["test_loss"] for m in valid_metrics]
        times = [m["training_time"] for m in valid_metrics]

        return {
            "test_accuracy": {
                "mean": np.mean(accuracies),
                "std": np.std(accuracies),
                "min": np.min(accuracies),
                "max": np.max(accuracies),
                "values": accuracies,
            },
            "test_f1": {
                "mean": np.mean(f1_scores),
                "std": np.std(f1_scores),
                "min": np.min(f1_scores),
                "max": np.max(f1_scores),
                "values": f1_scores,
            },
            "test_loss": {
                "mean": np.mean(losses),
                "std": np.std(losses),
                "min": np.min(losses),
                "max": np.max(losses),
                "values": losses,
            },
            "training_time": {
                "mean": np.mean(times),
                "std": np.std(times),
                "total": np.sum(times),
                "values": times,
            },
        }

    def _calculate_class_metrics(self, fold_classification_reports: list[dict]) -> dict:
        """Calculate per-class metrics across folds."""
        per_class_metrics = {}

        for class_name in self.class_names:
            precisions = []
            recalls = []
            f1s = []
            supports = []

            for report in fold_classification_reports:
                if class_name in report:
                    precisions.append(report[class_name].get("precision", 0))
                    recalls.append(report[class_name].get("recall", 0))
                    f1s.append(report[class_name].get("f1-score", 0))
                    supports.append(report[class_name].get("support", 0))

            if precisions:  # Only add if we have data
                per_class_metrics[class_name] = {
                    "precision": {
                        "mean": np.mean(precisions),
                        "std": np.std(precisions),
                        "min": np.min(precisions),
                        "max": np.max(precisions),
                    },
                    "recall": {
                        "mean": np.mean(recalls),
                        "std": np.std(recalls),
                        "min": np.min(recalls),
                        "max": np.max(recalls),
                    },
                    "f1-score": {"mean": np.mean(f1s), "std": np.std(f1s), "min": np.min(f1s), "max": np.max(f1s)},
                    "support": {
                        "mean": np.mean(supports),
                        "std": np.std(supports),
                        "min": np.min(supports),
                        "max": np.max(supports),
                    },
                }

        result = {"per_class_metrics": per_class_metrics}

        # Overall macro/weighted averages
        macro_metrics = self._calculate_averaged_metrics(fold_classification_reports, "macro avg")
        weighted_metrics = self._calculate_averaged_metrics(fold_classification_reports, "weighted avg")

        if macro_metrics:
            result["macro_averaged_metrics"] = macro_metrics
        if weighted_metrics:
            result["weighted_averaged_metrics"] = weighted_metrics

        return result

    def _calculate_averaged_metrics(self, fold_classification_reports: list[dict], avg_type: str) -> dict | None:
        """Calculate macro or weighted averaged metrics."""
        precisions = []
        recalls = []
        f1s = []

        for report in fold_classification_reports:
            if avg_type in report:
                precisions.append(report[avg_type].get("precision", 0))
                recalls.append(report[avg_type].get("recall", 0))
                f1s.append(report[avg_type].get("f1-score", 0))

        if not precisions:
            return None

        return {
            "precision": {
                "mean": np.mean(precisions),
                "std": np.std(precisions),
                "min": np.min(precisions),
                "max": np.max(precisions),
            },
            "recall": {
                "mean": np.mean(recalls),
                "std": np.std(recalls),
                "min": np.min(recalls),
                "max": np.max(recalls),
            },
            "f1-score": {"mean": np.mean(f1s), "std": np.std(f1s), "min": np.min(f1s), "max": np.max(f1s)},
        }

    def _log_summary_to_main_loggers(self, main_loggers: list, summary: dict) -> None:
        """Log summary metrics to main loggers."""
        if not main_loggers or not summary:
            return

        try:
            summary_metrics = {
                "cv_mean_accuracy": summary.get("metrics_summary", {}).get("test_accuracy", {}).get("mean", 0),
                "cv_std_accuracy": summary.get("metrics_summary", {}).get("test_accuracy", {}).get("std", 0),
                "cv_mean_f1": summary.get("metrics_summary", {}).get("test_f1", {}).get("mean", 0),
                "cv_std_f1": summary.get("metrics_summary", {}).get("test_f1", {}).get("std", 0),
                "cv_mean_loss": summary.get("metrics_summary", {}).get("test_loss", {}).get("mean", 0),
                "cv_std_loss": summary.get("metrics_summary", {}).get("test_loss", {}).get("std", 0),
                "cv_total_time": summary.get("experiment_info", {}).get("total_training_time", 0),
                "cv_successful_folds": len(
                    summary.get("metrics_summary", {}).get("test_accuracy", {}).get("values", [])
                ),
            }

            log_experiment_summary(main_loggers, summary_metrics)

        except Exception as e:
            logging.warning(f"Could not log summary metrics: {e}")

    def _save_and_log_results(self, summary: dict, fold_metrics: list[dict], results: dict) -> None:
        """Save cross-validation results and log summary."""
        # Save summary to files
        results_dir = self._save_results_to_files(summary, fold_metrics, results)

        # Log summary to console
        self._log_summary_to_console(summary)

        return results_dir

    def _save_results_to_files(self, summary: dict, fold_metrics: list[dict], results: dict) -> Path:
        """Save cross-validation results to files."""
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path(f"cv_results_{config.classes}_{config.MODEL_NAME}_{timestamp}")
        results_dir.mkdir(exist_ok=True)

        # Save summary as JSON
        summary_path = results_dir / "cross_validation_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        # Save fold metrics as CSV
        if fold_metrics:
            metrics_df = pd.DataFrame(fold_metrics)
            metrics_path = results_dir / "fold_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)

        # Save detailed results as JSON (excluding non-serializable objects)
        serializable_results = {}
        for fold_num, fold_result in results.items():
            serializable_results[fold_num] = {
                k: v
                for k, v in fold_result.items()
                if k not in ["trainer", "model"]  # Exclude non-serializable objects
            }

        detailed_path = results_dir / "detailed_results.json"
        with open(detailed_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        # Save configuration
        config_dict = {
            attr: getattr(config, attr)
            for attr in dir(config)
            if not attr.startswith("_") and not callable(getattr(config, attr))
        }
        config_path = results_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        logging.info(f"Cross-validation results saved to: {results_dir}")
        return results_dir

    def _log_summary_to_console(self, summary: dict) -> None:
        """Log cross-validation summary to console."""
        logging.info("\n" + "=" * 80)
        logging.info("CROSS-VALIDATION SUMMARY")
        logging.info("=" * 80)

        # Experiment info
        exp_info = summary.get("experiment_info", {})
        logging.info(f"Model: {exp_info.get('model_name', 'N/A')}")
        logging.info(f"Classes: {exp_info.get('classes', 'N/A')}")
        logging.info(f"Folds: {exp_info.get('n_folds', 'N/A')}")
        logging.info(f"Total Training Time: {exp_info.get('total_training_time', 0):.1f}s")

        # Metrics summary
        if "metrics_summary" in summary:
            self._log_metrics_summary(summary["metrics_summary"])

        # Per-class metrics
        if "per_class_metrics" in summary:
            self._log_per_class_metrics(summary["per_class_metrics"])

        # Macro/weighted averages
        if "macro_averaged_metrics" in summary:
            self._log_averaged_metrics("MACRO AVERAGED METRICS", summary["macro_averaged_metrics"])

        if "weighted_averaged_metrics" in summary:
            self._log_averaged_metrics("WEIGHTED AVERAGED METRICS", summary["weighted_averaged_metrics"])

        logging.info("=" * 80)

    def _log_metrics_summary(self, metrics: dict) -> None:
        """Log metrics summary to console."""
        logging.info("\nMETRICS ACROSS FOLDS:")
        logging.info("-" * 40)

        for metric_name, metric_data in metrics.items():
            if metric_name != "training_time":
                mean_val = metric_data.get("mean", 0)
                min_val = metric_data.get("min", 0)
                max_val = metric_data.get("max", 0)
                range_val = (max_val - min_val) / 2  # Half range for ± notation
                logging.info(f"{metric_name:15}: {mean_val:.4f} ± {range_val:.4f} (range: {min_val:.4f}-{max_val:.4f})")

    def _log_per_class_metrics(self, per_class: dict) -> None:
        """Log per-class metrics to console."""
        logging.info("\nPER-CLASS METRICS (Mean ± Range/2):")
        logging.info("-" * 40)

        for class_name, metrics in per_class.items():
            logging.info(f"{class_name}:")
            for metric_name, metric_data in metrics.items():
                if metric_name != "support":
                    mean_val = metric_data.get("mean", 0)
                    min_val = metric_data.get("min", 0)
                    max_val = metric_data.get("max", 0)
                    range_val = (max_val - min_val) / 2  # Half range for ± notation
                    logging.info(
                        f"  {metric_name:10}: {mean_val:.4f} ± {range_val:.4f} (range: {min_val:.4f}-{max_val:.4f})"
                    )

    def _log_averaged_metrics(self, title: str, averaged_metrics: dict) -> None:
        """Log macro/weighted averaged metrics to console."""
        logging.info(f"\n{title}:")
        logging.info("-" * 40)
        for metric_name, metric_data in averaged_metrics.items():
            mean_val = metric_data.get("mean", 0)
            min_val = metric_data.get("min", 0)
            max_val = metric_data.get("max", 0)
            range_val = (max_val - min_val) / 2  # Half range for ± notation
            logging.info(f"{metric_name:15}: {mean_val:.4f} ± {range_val:.4f} (range: {min_val:.4f}-{max_val:.4f})")
