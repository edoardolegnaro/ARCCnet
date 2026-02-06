import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from sklearn.metrics import classification_report, confusion_matrix

from arccnet.models import train_utils as ut_t
from arccnet.models.flares.multiclass import config


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for multiclass classification.

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the true class.

    Args:
        alpha (float or Tensor): Weighting factor for classes. Default: 1.0
        gamma (float): Focusing parameter. Higher gamma puts more focus on hard examples. Default: 2.0
        weight (Tensor, optional): Manual rescaling weight for each class.
        reduction (str): Specifies the reduction to apply to the output. Default: 'mean'
    """

    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (Tensor): Predictions of shape (N, C) where N is batch size and C is number of classes
            targets (Tensor): Ground truth labels of shape (N,)
        """
        # Ensure weight tensor is on the same device as inputs
        weight = self.weight
        if weight is not None:
            weight = weight.to(inputs.device)

        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, weight=weight, reduction="none")

        # Compute probabilities
        pt = torch.exp(-ce_loss)

        # Compute focal loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FlareClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module for multi-class flare classification.
    """

    def __init__(self, num_classes, class_names, class_weights=None):
        super().__init__()
        self.save_hyperparameters()

        self.model = timm.create_model(
            config.MODEL_NAME, pretrained=True, num_classes=self.hparams.num_classes, in_chans=1
        )

        # Replace all ReLU activations with LeakyReLU (negative_slope=0.01)
        ut_t.replace_activations(self.model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01)

        # Initialize loss function based on config
        self._setup_loss_function()

        # Define metrics for evaluation
        metric_args = {"task": "multiclass", "num_classes": self.hparams.num_classes, "average": "macro"}
        self.accuracy = torchmetrics.Accuracy(**metric_args)
        self.f1_score = torchmetrics.F1Score(**metric_args)
        self.precision_metric = torchmetrics.Precision(**metric_args)
        self.recall = torchmetrics.Recall(**metric_args)

        # Storage for test predictions and labels
        self.test_predictions = []
        self.test_labels = []
        self.test_images = []
        self.test_probs = []

    def _setup_loss_function(self):
        """Setup the loss function based on configuration."""
        if config.LOSS_TYPE == "focal":
            self.loss_fn = FocalLoss(
                alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA, weight=self.hparams.class_weights
            )
        elif config.LOSS_TYPE == "weighted_focal":
            # Use focal loss with stronger class weighting
            focal_weights = self.hparams.class_weights
            if focal_weights is not None:
                # Amplify weights for focal loss
                focal_weights = focal_weights * config.FOCAL_ALPHA
            self.loss_fn = FocalLoss(
                alpha=1.0,
                gamma=config.FOCAL_GAMMA,
                weight=focal_weights,  # Alpha handled by weights
            )
        elif config.LOSS_TYPE == "weighted_ce":
            self.loss_fn = nn.CrossEntropyLoss(weight=self.hparams.class_weights)
        else:  # default cross_entropy
            self.loss_fn = nn.CrossEntropyLoss()

        # Define metrics for evaluation
        metric_args = {"task": "multiclass", "num_classes": self.hparams.num_classes, "average": "macro"}
        self.accuracy = torchmetrics.Accuracy(**metric_args)
        self.f1_score = torchmetrics.F1Score(**metric_args)
        self.precision_metric = torchmetrics.Precision(**metric_args)
        self.recall = torchmetrics.Recall(**metric_args)

        # Storage for test predictions and labels
        self.test_predictions = []
        self.test_labels = []
        self.test_images = []
        self.test_probs = []

    def to(self, device):
        """Override to method to ensure class weights are moved with the model."""
        self = super().to(device)
        # Move class weights to the same device
        if hasattr(self, "loss_fn") and hasattr(self.loss_fn, "weight") and self.loss_fn.weight is not None:
            self.loss_fn.weight = self.loss_fn.weight.to(device)
        return self

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", self.f1_score(preds, labels), on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._shared_step(batch)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.f1_score(preds, labels), on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss_fn(logits, labels)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)

        # Store predictions and labels for confusion matrix
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_labels.extend(labels.cpu().numpy())
        self.test_probs.extend(probs.cpu().numpy())

        # Store a few images for misclassification analysis (limit to avoid memory issues)
        if len(self.test_images) < 100:  # Store max 100 images
            self.test_images.extend(images.cpu().numpy())

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_f1", self.f1_score(preds, labels), on_epoch=True)
        self.log("test_precision", self.precision_metric(preds, labels), on_epoch=True)
        self.log("test_recall", self.recall(preds, labels), on_epoch=True)

    def on_test_epoch_end(self):
        """Called at the end of test epoch to log confusion matrix and misclassified examples."""
        if len(self.test_predictions) > 0 and self.logger and config.ENABLE_COMET_LOGGING:
            # Create confusion matrix
            if config.LOG_CONFUSION_MATRIX:
                self._log_confusion_matrix()

            # Log classification report
            if config.LOG_CLASSIFICATION_REPORT:
                self._log_classification_report()

            # Log misclassified examples
            if config.LOG_MISCLASSIFIED_EXAMPLES:
                self._log_misclassified_examples(max_examples=config.MAX_MISCLASSIFIED_EXAMPLES)

            # Clear stored data
            self.test_predictions.clear()
            self.test_labels.clear()
            self.test_images.clear()
            self.test_probs.clear()

    def _log_confusion_matrix(self):
        """Create and log confusion matrix to Comet using both native and matplotlib methods."""
        y_true = np.array(self.test_labels)
        y_pred = np.array(self.test_predictions)

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Log to Comet using native confusion matrix feature
        if hasattr(self.logger, "experiment"):
            # Log using Comet's native confusion matrix
            self.logger.experiment.log_confusion_matrix(
                y_true=y_true,
                y_predicted=y_pred,
                labels=self.hparams.class_names,
                title="Test Set Confusion Matrix",
                file_name="confusion_matrix.json",
            )

            # Also log as a matplotlib figure for visualization
            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=self.hparams.class_names,
                yticklabels=self.hparams.class_names,
                cbar_kws={"label": "Count"},
            )
            plt.title("Confusion Matrix - Test Set", fontsize=14, fontweight="bold")
            plt.ylabel("True Label", fontsize=12)
            plt.xlabel("Predicted Label", fontsize=12)
            plt.tight_layout()

            # Log the matplotlib figure
            self.logger.experiment.log_figure("confusion_matrix_heatmap", plt.gcf())
            plt.close()

            # Log normalized confusion matrix
            cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                cm_normalized,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                xticklabels=self.hparams.class_names,
                yticklabels=self.hparams.class_names,
                cbar_kws={"label": "Normalized Count"},
            )
            plt.title("Normalized Confusion Matrix - Test Set", fontsize=14, fontweight="bold")
            plt.ylabel("True Label", fontsize=12)
            plt.xlabel("Predicted Label", fontsize=12)
            plt.tight_layout()

            # Log the normalized confusion matrix
            self.logger.experiment.log_figure("confusion_matrix_normalized", plt.gcf())
            plt.close()

    def _log_classification_report(self):
        """Generate and log detailed classification report with per-class metrics."""
        y_true = np.array(self.test_labels)
        y_pred = np.array(self.test_predictions)

        # Generate classification report as dictionary
        report_dict = classification_report(
            y_true, y_pred, target_names=self.hparams.class_names, output_dict=True, zero_division=0
        )

        # Log individual class metrics
        for class_name in self.hparams.class_names:
            if class_name in report_dict:
                class_metrics = report_dict[class_name]

                # Log per-class metrics to Comet
                if hasattr(self.logger, "experiment"):
                    self.logger.experiment.log_metric(f"test_precision_{class_name}", class_metrics["precision"])
                    self.logger.experiment.log_metric(f"test_recall_{class_name}", class_metrics["recall"])
                    self.logger.experiment.log_metric(f"test_f1_{class_name}", class_metrics["f1-score"])
                    self.logger.experiment.log_metric(f"test_support_{class_name}", class_metrics["support"])

        # Log overall metrics
        if "macro avg" in report_dict:
            macro_metrics = report_dict["macro avg"]
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.log_metric("test_macro_precision", macro_metrics["precision"])
                self.logger.experiment.log_metric("test_macro_recall", macro_metrics["recall"])
                self.logger.experiment.log_metric("test_macro_f1", macro_metrics["f1-score"])

        if "weighted avg" in report_dict:
            weighted_metrics = report_dict["weighted avg"]
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.log_metric("test_weighted_precision", weighted_metrics["precision"])
                self.logger.experiment.log_metric("test_weighted_recall", weighted_metrics["recall"])
                self.logger.experiment.log_metric("test_weighted_f1", weighted_metrics["f1-score"])

        # Log accuracy
        if "accuracy" in report_dict:
            if hasattr(self.logger, "experiment"):
                self.logger.experiment.log_metric("test_accuracy", report_dict["accuracy"])

        # Create and log a formatted classification report as text
        report_str = classification_report(y_true, y_pred, target_names=self.hparams.class_names, zero_division=0)

        if hasattr(self.logger, "experiment"):
            self.logger.experiment.log_text("classification_report", report_str)

        # Create a visual representation of the classification report
        self._log_classification_report_figure(report_dict)

    def _log_classification_report_figure(self, report_dict):
        """Create a visual representation of the classification report."""
        # Extract metrics for visualization
        classes = self.hparams.class_names
        metrics = ["precision", "recall", "f1-score"]

        # Create data matrix for heatmap
        data = []
        for class_name in classes:
            if class_name in report_dict:
                row = [report_dict[class_name][metric] for metric in metrics]
                data.append(row)
            else:
                data.append([0.0, 0.0, 0.0])  # Default values if class not found

        data = np.array(data)

        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            data,
            annot=True,
            fmt=".3f",
            cmap="RdYlBu_r",
            xticklabels=metrics,
            yticklabels=classes,
            vmin=0,
            vmax=1,
            cbar_kws={"label": "Score"},
        )
        plt.title("Per-Class Classification Metrics")
        plt.xlabel("Metrics")
        plt.ylabel("Classes")
        plt.tight_layout()

        # Log to Comet
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.log_figure("classification_report_heatmap", plt.gcf())

        plt.close()

    def _log_misclassified_examples(self, max_examples=20):
        """Log misclassified examples to Comet."""
        if len(self.test_images) == 0:
            return

        predictions = np.array(self.test_predictions[: len(self.test_images)])
        true_labels = np.array(self.test_labels[: len(self.test_images)])
        images = np.array(self.test_images)
        probs = np.array(self.test_probs[: len(self.test_images)])

        # Find misclassified examples
        misclassified_mask = predictions != true_labels
        misclassified_indices = np.where(misclassified_mask)[0]

        if len(misclassified_indices) == 0:
            return

        # Limit number of examples
        num_examples = min(max_examples, len(misclassified_indices))
        selected_indices = misclassified_indices[:num_examples]

        # Create figure with misclassified examples
        cols = 4
        rows = (num_examples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))

        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, idx in enumerate(selected_indices):
            row = i // cols
            col = i % cols

            # Get image (remove channel dimension for display)
            img = images[idx].squeeze()

            # Plot image
            axes[row, col].imshow(img, cmap="gray")
            axes[row, col].set_title(
                f"True: {self.hparams.class_names[true_labels[idx]]}\n"
                f"Pred: {self.hparams.class_names[predictions[idx]]}\n"
                f"Conf: {probs[idx][predictions[idx]]:.3f}",
                fontsize=10,
            )
            axes[row, col].axis("off")

        # Hide empty subplots
        for i in range(num_examples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis("off")

        plt.tight_layout()

        # Log to Comet
        if hasattr(self.logger, "experiment"):
            self.logger.experiment.log_figure("misclassified_examples", fig)

        plt.close()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=config.LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": config.CHECKPOINT_METRIC},
        }
