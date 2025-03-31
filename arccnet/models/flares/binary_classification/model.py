# model.py (Modified Content)
"""
PyTorch Lightning model definition using timm library.
"""

import pytorch_lightning as pl
import timm
import torch
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
        pos_weight: torch.Tensor = None,
    ):
        """
        Args:
            model_name: Name of the model architecture from timm (e.g., 'resnet50', 'efficientnet_b0', 'vit_base_patch16_224').
            num_classes: Number of output classes (1 for binary classification with BCEWithLogitsLoss).
            in_chans: Number of input image channels.
            pretrained: Whether to load pretrained weights (if available for the model and compatible with in_chans).
            learning_rate: Learning rate for the optimizer.
            pos_weight: Weight for positive samples in binary cross entropy loss.
        """
        super().__init__()
        # Save hyperparameters like learning rate, model name etc. automatically
        # Makes them accessible via self.hparams
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        # Register pos_weight as a buffer to ensure it moves with the model
        # This is needed for Distributed Training on multiple GPUs
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.register_buffer("pos_weight", torch.tensor([1.0], dtype=torch.float32))

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
                "confusion_matrix": BinaryConfusionMatrix(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

        # Store confusion matrix for logging
        self.test_confusion_matrix = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute binary cross entropy loss with optional positive weight."""
        logits = logits.squeeze(1)  # Convert from [batch_size, 1] to [batch_size]
        return F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            pos_weight=self.pos_weight,
        )

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute training step."""
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        # Calculate metrics with reshaped predictions
        preds = torch.sigmoid(logits.squeeze(1))  # Convert to probabilities
        self.train_metrics(preds, y)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute validation step."""
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        # Calculate metrics with reshaped predictions
        preds = torch.sigmoid(logits.squeeze(1))  # Convert to probabilities
        self.val_metrics(preds, y)
        self.log_dict(self.val_metrics, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute test step."""
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        # Calculate metrics with reshaped predictions
        preds = torch.sigmoid(logits.squeeze(1))  # Convert to probabilities
        self.test_metrics(preds, y)
        self.log_dict(self.test_metrics, on_step=False, on_epoch=True)
        self.log("test_loss", loss, prog_bar=True)

        # Store confusion matrix
        if batch_idx == 0:  # Initialize on first batch
            self.test_confusion_matrix = self.test_metrics.confusion_matrix(preds, y)
        else:  # Update for subsequent batches
            self.test_confusion_matrix += self.test_metrics.confusion_matrix(preds, y)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        # Use learning rate from saved hyperparameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

    def on_test_end(self):
        """Called when test ends."""
        # Convert confusion matrix to numpy for logging
        if self.test_confusion_matrix is not None:
            cm = self.test_confusion_matrix.cpu().numpy()
            # Log confusion matrix to Comet if logger exists
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.experiment.log_confusion_matrix(
                    matrix=cm,
                    labels=["No Flare", "Flare"],
                    title="Test Set Confusion Matrix",
                    row_label="Actual",
                    column_label="Predicted",
                )
