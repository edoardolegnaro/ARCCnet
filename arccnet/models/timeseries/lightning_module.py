"""PyTorch Lightning module for timeseries flare forecasting."""

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from .flare_forecaster import FlareForecaster
from .focal_loss import FocalLoss


class FlareForecasterLightning(pl.LightningModule):
    """PyTorch Lightning module for flare forecasting."""

    def __init__(
        self,
        task_type="multiclass",
        num_channels=10,
        output_dim=3,
        learning_rate=1e-4,
        weight_decay=1e-5,
        class_weights=None,
        loss_function="cross_entropy",
        focal_alpha=0.25,
        focal_gamma=2.0,
        **kwargs,
    ):
        """
        Initialize the Lightning module.

        Parameters
        ----------
        task_type : str
            'multiclass' or 'regression'
        num_channels : int
            Number of input channels
        output_dim : int
            Output dimension (3 for both tasks)
        learning_rate : float
            Learning rate for optimizer
        weight_decay : float
            Weight decay for optimizer
        class_weights : list or None
            Class weights for multiclass classification
        loss_function : str
            Loss function to use: 'cross_entropy' or 'focal'
        focal_alpha : float
            Alpha parameter for focal loss
        focal_gamma : float
            Gamma parameter for focal loss
        **kwargs
            Additional parameters for FlareForecaster
        """
        super().__init__()
        self.save_hyperparameters()

        self.task_type = task_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_function = loss_function

        # Create model
        self.model = FlareForecaster(task_type=task_type, num_channels=num_channels, output_dim=output_dim, **kwargs)

        # Loss function
        if task_type == "multiclass":
            if loss_function == "focal":
                # For focal loss, convert class weights to alpha if provided
                if class_weights is not None:
                    weight = torch.tensor(class_weights, dtype=torch.float32)
                else:
                    weight = None
                self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, weight=weight)
            else:  # cross_entropy
                if class_weights is not None:
                    weight = torch.tensor(class_weights, dtype=torch.float32)
                    self.criterion = nn.CrossEntropyLoss(weight=weight)
                else:
                    self.criterion = nn.CrossEntropyLoss()
        else:  # regression
            self.criterion = nn.MSELoss()

        # For collecting predictions
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x, mask=None):
        """Forward pass."""
        return self.model(x, mask=mask)

    def training_step(self, batch, batch_idx):
        """Training step."""
        x = batch["x"]
        y = batch["y"]
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(x.device)
        output = self(x, mask=mask)
        loss = self.criterion(output, y)

        # Log training loss
        batch_size = x.size(0)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch["x"]
        y = batch["y"]
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(x.device)
        output = self(x, mask=mask)
        loss = self.criterion(output, y)

        # Store predictions for metric computation
        self.validation_step_outputs.append({"loss": loss, "preds": output.detach(), "targets": y.detach()})

        return loss

    def on_validation_epoch_end(self):
        """Compute validation metrics at the end of epoch."""
        if not self.validation_step_outputs:
            return

        # Aggregate outputs
        avg_loss = torch.stack([x["loss"] for x in self.validation_step_outputs]).mean()
        all_preds = torch.cat([x["preds"] for x in self.validation_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.validation_step_outputs])

        # Move to CPU for sklearn metrics
        all_preds = all_preds.cpu().numpy()
        all_targets = all_targets.cpu().numpy()

        # Compute task-specific metrics
        if self.task_type == "multiclass":
            metrics = self._compute_multiclass_metrics(all_targets, all_preds)
            primary_metric = metrics.get("m_plus_tss", metrics["balanced_accuracy"])
        else:  # regression
            metrics = self._compute_regression_metrics(all_targets, all_preds)
            primary_metric = -metrics["rmse"]  # negative RMSE for maximization

        # Log metrics
        batch_size = len(all_targets)
        self.log("val/loss", avg_loss, prog_bar=True, batch_size=batch_size)
        self.log("val/primary_metric", primary_metric, prog_bar=True, batch_size=batch_size)

        for metric_name, metric_value in metrics.items():
            self.log(f"val/{metric_name}", metric_value, batch_size=batch_size)

        # Clear outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        """Test step."""
        x = batch["x"]
        y = batch["y"]
        mask = batch.get("mask")
        if mask is not None:
            mask = mask.to(x.device)
        output = self(x, mask=mask)
        loss = self.criterion(output, y)

        # Store predictions for metric computation
        self.test_step_outputs.append({"loss": loss, "preds": output.detach(), "targets": y.detach()})

        return loss

    def on_test_epoch_end(self):
        """Compute test metrics at the end of testing."""
        if not self.test_step_outputs:
            return

        # Aggregate outputs
        avg_loss = torch.stack([x["loss"] for x in self.test_step_outputs]).mean()
        all_preds = torch.cat([x["preds"] for x in self.test_step_outputs])
        all_targets = torch.cat([x["targets"] for x in self.test_step_outputs])

        # Move to CPU for sklearn metrics
        all_preds = all_preds.cpu().numpy()
        all_targets = all_targets.cpu().numpy()

        # Compute task-specific metrics
        if self.task_type == "multiclass":
            metrics = self._compute_multiclass_metrics(all_targets, all_preds)
        else:  # regression
            metrics = self._compute_regression_metrics(all_targets, all_preds)

        # Log metrics
        batch_size = len(all_targets)
        self.log("test/loss", avg_loss, batch_size=batch_size)
        for metric_name, metric_value in metrics.items():
            self.log(f"test/{metric_name}", metric_value, batch_size=batch_size)

        # Clear outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/primary_metric",
            },
        }

    def _compute_multiclass_metrics(self, y_true, y_pred):
        """Compute metrics for multiclass classification."""
        y_true = np.asarray(y_true, dtype=np.int64)
        logits = np.asarray(y_pred, dtype=np.float32)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        y_pred_class = np.argmax(probs, axis=1)

        # Get unique classes present in data
        all_classes = [0, 1, 2]

        acc = accuracy_score(y_true, y_pred_class)
        bal_acc = balanced_accuracy_score(y_true, y_pred_class)
        cm = confusion_matrix(y_true, y_pred_class, labels=all_classes)

        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

        metrics = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "class_0_acc": float(per_class_acc[0]) if len(per_class_acc) > 0 else 0.0,
            "class_1_acc": float(per_class_acc[1]) if len(per_class_acc) > 1 else 0.0,
            "class_2_acc": float(per_class_acc[2]) if len(per_class_acc) > 2 else 0.0,
        }

        m_plus_true = (y_true >= 1).astype(int)
        m_plus_score = probs[:, 1] + probs[:, 2]
        x_plus_true = (y_true == 2).astype(int)
        x_plus_score = probs[:, 2]

        for prefix, y_bin, y_score in [
            ("m_plus", m_plus_true, m_plus_score),
            ("x_plus", x_plus_true, x_plus_score),
        ]:
            skill = self._compute_binary_skill(y_bin, y_score, threshold=0.5)
            metrics[f"{prefix}_tss"] = skill["tss"]
            metrics[f"{prefix}_hss"] = skill["hss"]
            metrics[f"{prefix}_tpr"] = skill["tpr"]
            metrics[f"{prefix}_fpr"] = skill["fpr"]
            metrics[f"{prefix}_pr_auc"] = skill["pr_auc"] if skill["pr_auc"] is not None else 0.0
            metrics[f"{prefix}_roc_auc"] = skill["roc_auc"] if skill["roc_auc"] is not None else 0.0

        return metrics

    @staticmethod
    def _compute_binary_skill(y_true, y_score, threshold=0.5):
        """Compute operational binary skill metrics from score outputs."""
        y_true = np.asarray(y_true).astype(int)
        y_pred = (np.asarray(y_score) >= threshold).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        tss = tpr - fpr

        hss_num = 2.0 * (tp * tn - fp * fn)
        hss_den = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn)) + 1e-8
        hss = hss_num / hss_den

        if len(np.unique(y_true)) > 1:
            roc_auc = float(roc_auc_score(y_true, y_score))
            pr_auc = float(average_precision_score(y_true, y_score))
        else:
            roc_auc = None
            pr_auc = None

        return {
            "tss": float(tss),
            "hss": float(hss),
            "tpr": float(tpr),
            "fpr": float(fpr),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }

    def _compute_regression_metrics(self, y_true, y_pred):
        """Compute metrics for regression."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # Per-target metrics
        mse_per_target = ((y_true - y_pred) ** 2).mean(axis=0)
        mae_per_target = np.abs(y_true - y_pred).mean(axis=0)

        # R2 per target
        r2_scores = []
        for i in range(y_true.shape[1]):
            try:
                r2 = r2_score(y_true[:, i], y_pred[:, i])
                r2_scores.append(float(r2))
            except Exception:
                r2_scores.append(0.0)

        return {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "target_0_mse": float(mse_per_target[0]),
            "target_1_mse": float(mse_per_target[1]),
            "target_2_mse": float(mse_per_target[2]),
            "target_0_mae": float(mae_per_target[0]),
            "target_1_mae": float(mae_per_target[1]),
            "target_2_mae": float(mae_per_target[2]),
            "target_0_r2": r2_scores[0],
            "target_1_r2": r2_scores[1],
            "target_2_r2": r2_scores[2],
        }
