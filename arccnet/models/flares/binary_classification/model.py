# model.py (Modified Content)
"""
PyTorch Lightning model definition using timm library.
"""

import pytorch_lightning as pl
import timm
import torch
from torch import nn
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

from arccnet.models import train_utils as ut_t


# Renamed class for generality
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
        pretrained: bool = False,  # Load pretrained weights?
        learning_rate: float = 1e-4,
    ):
        """
        Args:
            model_name: Name of the model architecture from timm (e.g., 'resnet50', 'efficientnet_b0', 'vit_base_patch16_224').
            num_classes: Number of output classes (1 for binary classification with BCEWithLogitsLoss).
            in_chans: Number of input image channels.
            pretrained: Whether to load pretrained weights (if available for the model and compatible with in_chans).
            learning_rate: Learning rate for the optimizer.
        """
        super().__init__()
        # Save hyperparameters like learning rate, model name etc. automatically
        # Makes them accessible via self.hparams
        self.save_hyperparameters()

        # Create the specified model using timm
        # This function handles various architectures based on model_name
        self.model = timm.create_model(
            self.hparams.model_name,
            pretrained=self.hparams.pretrained,
            num_classes=self.hparams.num_classes,
            in_chans=self.hparams.in_chans,
        )
        ut_t.replace_activations(self.model, torch.nn.ReLU, torch.nn.LeakyReLU, negative_slope=0.01)

        self.loss_fn = nn.BCEWithLogitsLoss()

        metrics = MetricCollection(
            {"acc": BinaryAccuracy(), "precision": BinaryPrecision(), "recall": BinaryRecall(), "f1": BinaryF1Score()}
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def _shared_step(self, batch: tuple, metrics: MetricCollection) -> torch.Tensor:
        """Common logic for training, validation, and test steps."""
        x, y = batch
        # Get raw logits from the model
        y_logits = self(x)
        # Remove trailing dimension if num_classes is 1
        if self.hparams.num_classes == 1:
            y_logits = y_logits.squeeze(-1)

        # Calculate loss (ensure target is float for BCEWithLogitsLoss)
        loss = self.loss_fn(y_logits, y.float())

        # Calculate predictions (apply sigmoid and threshold for binary case)
        if self.hparams.num_classes == 1:
            preds = (torch.sigmoid(y_logits) > 0.5).long()
        else:
            preds = torch.argmax(y_logits, dim=1)

        # Update metrics state (ensure target `y` has correct type for metrics)
        metric_output = metrics(preds, y.int())

        # Log metrics and loss for monitoring
        self.log(f"{metrics.prefix}loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log_dict(metric_output, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute training step."""
        return self._shared_step(batch, self.train_metrics)

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute validation step."""
        return self._shared_step(batch, self.val_metrics)

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Execute test step."""
        return self._shared_step(batch, self.test_metrics)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        # Use learning rate from saved hyperparameters
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # You could add learning rate schedulers here as well
        return optimizer
