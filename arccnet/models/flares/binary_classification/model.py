# model.py (Modified Content)
"""
PyTorch Lightning model definition using timm library.
"""

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from arccnet.models import train_utils as ut_t


class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.

    Focal Loss is designed to address class imbalance by down-weighting
    easy examples and focusing on hard examples.

    Args:
        alpha (float): Weight for rare class (positive class). Default: 0.25
        gamma (float): Focusing parameter. Higher values focus more on hard examples. Default: 2.0
        reduction (str): Specifies the reduction to apply to the output. Default: 'mean'
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model (before sigmoid)
            targets (torch.Tensor): Ground truth binary labels (0 or 1)
        """
        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

        # Compute probabilities
        p_t = torch.sigmoid(inputs)

        # For binary classification, we need to handle both classes
        # When target is 1 (positive class), use p_t as is
        # When target is 0 (negative class), use (1 - p_t)
        p_t = targets * p_t + (1 - targets) * (1 - p_t)

        # Compute alpha weight
        # When target is 1 (positive class), use alpha
        # When target is 0 (negative class), use (1 - alpha)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # Compute focal weight
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Apply focal weight to BCE loss
        focal_loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class FlareClassifier(pl.LightningModule):
    """
    PyTorch Lightning Module wrapping a generic image classification model
    from the 'timm' library (e.g., ViT, ResNet, EfficientNet) for binary
    flare classification. Includes loss, metrics, optimizer configuration,
    and training/validation/test steps.
    """

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        num_classes: int = 1,
        in_chans: int = 1,
        pretrained: bool = False,
        learning_rate: float = 1e-4,
        loss_function: str = "bce",
        pos_weight: torch.Tensor = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            model_name: Name of the model architecture from timm (e.g., 'resnet50', 'efficientnet_b0', 'vit_base_patch16_224').
            num_classes: Number of output classes (1 for binary classification with BCEWithLogitsLoss).
            in_chans: Number of input image channels.
            pretrained: Whether to load pretrained weights (if available for the model and compatible with in_chans).
            learning_rate: Learning rate for the optimizer.
            loss_function: Type of loss function to use. Options: "bce", "focal", "weighted_bce".
            pos_weight: Weight for positive samples in binary cross entropy loss.
            focal_alpha: Alpha parameter for focal loss (weight for rare class).
            focal_gamma: Gamma parameter for focal loss (focusing parameter).
        """
        super().__init__()
        # Save hyperparameters like learning rate, model name etc. automatically
        # Makes them accessible via self.hparams
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.loss_function = loss_function

        # Register pos_weight as a buffer to ensure it moves with the model
        # This is needed for Distributed Training on multiple GPUs
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.register_buffer("pos_weight", torch.tensor([1.0], dtype=torch.float32))

        # Initialize loss function based on configuration
        if loss_function == "focal":
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif loss_function == "weighted_bce":
            # Will use F.binary_cross_entropy_with_logits with pos_weight
            self.criterion = None
        elif loss_function == "bce":
            # Standard BCE without weights
            self.criterion = None
        else:
            raise ValueError(f"Unknown loss function: {loss_function}. Options are: 'bce', 'focal', 'weighted_bce'")

        # Create the specified model using timm
        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=self.hparams.pretrained,
            num_classes=self.hparams.num_classes,
            in_chans=self.hparams.in_chans,
        )
        ut_t.replace_activations(self.model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01)

        metrics = MetricCollection(
            {
                "acc": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Create separate confusion matrix metrics
        self.train_confusion_matrix = BinaryConfusionMatrix()
        self.val_confusion_matrix = BinaryConfusionMatrix()
        self.test_confusion_matrix = BinaryConfusionMatrix()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss based on the configured loss function."""
        logits = logits.squeeze(1)  # Convert from [batch_size, 1] to [batch_size]

        if self.loss_function == "focal":
            return self.criterion(logits, targets.float())
        elif self.loss_function == "weighted_bce":
            return F.binary_cross_entropy_with_logits(
                logits,
                targets.float(),
                pos_weight=self.pos_weight,
            )
        elif self.loss_function == "bce":
            return F.binary_cross_entropy_with_logits(
                logits,
                targets.float(),
            )
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function}")

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute training step."""
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        # Calculate metrics with reshaped predictions
        preds = torch.sigmoid(logits.squeeze(1))  # Convert to probabilities
        self.train_metrics(preds, y)
        self.train_confusion_matrix(preds, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, sync_dist=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute validation step."""
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        # Calculate metrics with reshaped predictions
        preds = torch.sigmoid(logits.squeeze(1))  # Convert to probabilities
        self.val_metrics(preds, y)
        self.val_confusion_matrix(preds, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True, sync_dist=True)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute test step."""
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        # Calculate metrics with reshaped predictions
        preds = torch.sigmoid(logits.squeeze(1))  # Convert to probabilities
        self.test_metrics(preds, y)
        self.test_confusion_matrix(preds, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True, sync_dist=True)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        # Use learning rate from saved hyperparameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        self.train_confusion_matrix.reset()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        self.val_confusion_matrix.reset()

    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        if self.logger is not None:
            cm = self.test_confusion_matrix.compute().cpu().numpy()
            self.logger.experiment.log_confusion_matrix(
                matrix=cm,
                labels=["No Flare", "Flare"],
                title="Test Set Confusion Matrix",
                row_label="Actual",
                column_label="Predicted",
            )
        self.test_confusion_matrix.reset()
