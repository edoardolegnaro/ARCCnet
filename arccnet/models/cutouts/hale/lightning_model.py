"""
PyTorch Lightning model for Hale classification.
"""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics import Accuracy, F1Score

from . import config


class HaleLightningModel(pl.LightningModule):
    """
    Simple Lightning model for Hale classification (Alpha, Beta, Beta-Gamma).
    Supports magnetogram, continuum, or both as input.
    """

    def __init__(
        self,
        num_classes: int = 3,
        learning_rate: float = 1e-3,
        model_name: str = "resnet18",
        class_weights: torch.Tensor = None,
        data_type: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.data_type = data_type or config.DATA_TYPE

        # Determine input channels based on data type
        if self.data_type == "both":
            input_channels = 2  # magnetogram + continuum
        else:
            input_channels = 1  # magnetogram or continuum only

        # Load pretrained ResNet and modify for our input channels
        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        elif model_name == "resnet34":
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        # Modify first conv layer for our input channels
        original_conv1 = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            input_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False,
        )

        # If using pretrained weights and input_channels=1, copy the RGB weights
        if input_channels == 1:
            self.backbone.conv1.weight.data = original_conv1.weight.data.mean(dim=1, keepdim=True)
        elif input_channels == 2:
            # For 2 channels, use first 2 channels of RGB weights
            self.backbone.conv1.weight.data = original_conv1.weight.data[:, :2, :, :]

        # Modify final layer for our number of classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        # Loss function with optional class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, labels)
        f1 = self.train_f1(preds, labels)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        self.log("train_f1", f1)

    def on_train_epoch_end(self):
        """Log training epoch summary."""
        train_acc = self.train_accuracy.compute()
        train_f1 = self.train_f1.compute()
        print(f"Training Epoch {self.current_epoch} completed - Acc: {train_acc:.4f}, F1: {train_f1:.4f}")

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, labels)
        f1 = self.val_f1(preds, labels)

        # Log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_f1", f1)

        return loss

    def on_validation_epoch_end(self):
        """Log validation epoch summary."""
        val_acc = self.val_accuracy.compute()
        val_f1 = self.val_f1.compute()
        print(f"Validation Epoch {self.current_epoch} completed - Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    def test_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.criterion(logits, labels)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(preds, labels)
        f1 = self.test_f1(preds, labels)

        # Log metrics
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_f1", f1)

        return {"test_loss": loss, "preds": preds, "targets": labels}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets with proper label handling.

    Args:
        labels: Array of original encoded label indices (e.g., [2, 3, 4])
        num_classes: Number of unique classes (should be 3 for Hale)

    Returns:
        Class weights as torch tensor for model classes [0, 1, 2]
    """
    # Map original labels to contiguous indices [0, 1, 2, ...]
    unique_labels = np.unique(labels)
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}

    # Convert original labels to contiguous indices
    mapped_labels = np.array([label_to_idx[label] for label in labels])

    # Compute balanced class weights for the mapped labels
    class_weights = compute_class_weight("balanced", classes=np.arange(len(unique_labels)), y=mapped_labels)

    return torch.FloatTensor(class_weights)
