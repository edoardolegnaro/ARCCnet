"""
PyTorch Lightning model for Hale classification.
"""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import classification_report, confusion_matrix
from torchmetrics import Accuracy, F1Score

import arccnet.models.cutouts.hale.config as config
from arccnet.models.train_utils import replace_activations


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

        # Replace ReLU with LeakyReLU activations
        replace_activations(self.backbone, nn.ReLU, nn.LeakyReLU, negative_slope=config.LEAKY_RELU_NEGATIVE_SLOPE)

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

        # Modify final layer for our number of classes with dropout
        self.backbone.fc = nn.Sequential(
            nn.Dropout(config.DROPOUT_RATE), nn.Linear(self.backbone.fc.in_features, num_classes)
        )

        # Loss function with optional class weights
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

        # Store for confusion matrix and classification report
        self.test_predictions = []
        self.test_targets = []
        self.test_logits = []
        self.test_misclassified_samples = []

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

        return loss

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

        # Store predictions and targets for confusion matrix and classification report
        self.test_predictions.extend(preds.cpu().numpy())
        self.test_targets.extend(labels.cpu().numpy())
        self.test_logits.append(logits.detach().cpu())
        self._update_top_misclassified_samples(images, labels, preds, logits, batch_idx)

        # Log metrics
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        self.log("test_f1", f1)

        return {"test_loss": loss, "preds": preds, "targets": labels}

    def get_confusion_matrix_and_classification_report(self, class_names=None):
        """
        Compute confusion matrix and classification report from collected test predictions.

        Args:
            class_names: List of class names for the classification report

        Returns:
            tuple: (confusion_matrix, classification_report_dict)
        """
        if not self.test_predictions or not self.test_targets:
            return None, None

        # Convert to numpy arrays
        y_true = np.array(self.test_targets)
        y_pred = np.array(self.test_predictions)

        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Compute classification report
        if class_names is None:
            class_names = [f"Class_{i}" for i in range(self.num_classes)]

        # Get classification report as dictionary for better logging
        class_report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        return cm, class_report

    def _update_top_misclassified_samples(self, images, labels, preds, logits, batch_idx: int) -> None:
        """Track top-confidence misclassifications during the existing test pass."""
        misclassified_mask = preds != labels
        if not misclassified_mask.any():
            return

        probabilities = torch.softmax(logits.detach(), dim=1)
        wrong_confidences = probabilities[misclassified_mask].max(dim=1)[0]
        misclassified_images = images[misclassified_mask].detach().cpu()
        misclassified_true = labels[misclassified_mask].detach().cpu()
        misclassified_pred = preds[misclassified_mask].detach().cpu()

        for i in range(len(misclassified_images)):
            self.test_misclassified_samples.append(
                {
                    "image": misclassified_images[i],
                    "true_label": misclassified_true[i].item(),
                    "pred_label": misclassified_pred[i].item(),
                    "confidence": wrong_confidences[i].detach().cpu().item(),
                    "batch_idx": batch_idx,
                    "sample_idx": i,
                }
            )

        max_samples = max(1, int(getattr(config, "MISCLASSIFIED_SAMPLES_TO_LOG", 10)))
        self.test_misclassified_samples.sort(key=lambda x: x["confidence"], reverse=True)
        if len(self.test_misclassified_samples) > max_samples:
            self.test_misclassified_samples = self.test_misclassified_samples[:max_samples]

    def get_top_misclassified_samples(self, num_samples: int = 10) -> list[dict]:
        """Return cached misclassified samples sorted by confidence descending."""
        return self.test_misclassified_samples[:num_samples]

    def reset_test_collections(self):
        """Reset the test predictions and targets collections."""
        self.test_predictions = []
        self.test_targets = []
        self.test_logits = []
        self.test_misclassified_samples = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=config.WEIGHT_DECAY,  # Add L2 regularization
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.LR_SCHEDULER_MODE,
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": config.LR_SCHEDULER_MONITOR,
            },
        }
