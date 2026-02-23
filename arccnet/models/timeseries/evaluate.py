"""Evaluation script for timeseries flare forecasting model."""

import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

warnings.filterwarnings("ignore")

from .config import *
from .dataset import SDOTimeseriesDataset
from .flare_forecaster import FlareForecaster
from .splitters import get_split


def compute_tss(y_true, y_pred, threshold=0.5):
    """Compute True Skill Statistic (TSS)."""
    y_pred_binary = (y_pred >= threshold).astype(int)

    tp = ((y_true == 1) & (y_pred_binary == 1)).sum()
    tn = ((y_true == 0) & (y_pred_binary == 0)).sum()
    fp = ((y_true == 0) & (y_pred_binary == 1)).sum()
    fn = ((y_true == 1) & (y_pred_binary == 0)).sum()

    tpr = tp / (tp + fn + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    tss = tpr - fpr

    return {
        "tss": float(tss),
        "tpr": float(tpr),
        "fpr": float(fpr),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def compute_metrics(y_true, y_pred, threshold=0.5):
    """Compute comprehensive metrics for multi-label classification."""
    num_classes = y_true.shape[1]
    class_names = ["C+", "M+", "X+"]

    metrics = {}

    for i in range(num_classes):
        y_true_i = y_true[:, i]
        y_pred_i = y_pred[:, i]

        class_name = class_names[i]

        # TSS
        tss_metrics = compute_tss(y_true_i, y_pred_i, threshold)
        metrics[f"{class_name}_tss"] = tss_metrics["tss"]
        metrics[f"{class_name}_tpr"] = tss_metrics["tpr"]
        metrics[f"{class_name}_fpr"] = tss_metrics["fpr"]

        # ROC-AUC
        try:
            roc_auc = roc_auc_score(y_true_i, y_pred_i)
            metrics[f"{class_name}_roc_auc"] = float(roc_auc)
        except:
            metrics[f"{class_name}_roc_auc"] = None

        # PR-AUC
        try:
            pr_auc = average_precision_score(y_true_i, y_pred_i)
            metrics[f"{class_name}_pr_auc"] = float(pr_auc)
        except:
            metrics[f"{class_name}_pr_auc"] = None

        # Confusion matrix
        cm = confusion_matrix(y_true_i, (y_pred_i >= threshold).astype(int))
        metrics[f"{class_name}_confusion_matrix"] = cm.tolist()

    # Aggregate metrics
    metrics["avg_tss"] = np.mean([metrics[f"{c}_tss"] for c in class_names])
    valid_roc = [metrics[f"{c}_roc_auc"] for c in class_names if metrics[f"{c}_roc_auc"] is not None]
    metrics["avg_roc_auc"] = float(np.mean(valid_roc)) if valid_roc else None
    valid_pr = [metrics[f"{c}_pr_auc"] for c in class_names if metrics[f"{c}_pr_auc"] is not None]
    metrics["avg_pr_auc"] = float(np.mean(valid_pr)) if valid_pr else None

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Run model inference and collect predictions."""
    model.eval()

    all_preds = []
    all_labels = []
    all_meta = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        x = batch["x"].to(device)
        y = batch["y"].cpu().numpy()
        meta = batch["meta"]

        # Forward
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()

        all_preds.append(probs)
        all_labels.append(y)
        all_meta.append(meta)

    # Concatenate results
    predictions = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    # Merge metadata
    metadata = {
        "sample_id": [],
        "noaa_ar": [],
        "date": [],
    }
    for meta_batch in all_meta:
        for key in metadata:
            if isinstance(meta_batch[key], list):
                metadata[key].extend(meta_batch[key])
            else:
                metadata[key].extend(
                    meta_batch[key].tolist() if hasattr(meta_batch[key], "tolist") else [meta_batch[key]]
                )

    return predictions, labels, metadata


def main(args):
    """Main evaluation function."""

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest
    print(f"Loading manifest from {args.manifest_path}")
    manifest = pd.read_parquet(args.manifest_path)

    # Get split
    splits = get_split(manifest, strategy=SPLIT_STRATEGY, seed=SEED)

    # Select split to evaluate
    if args.split == "test":
        eval_df = splits["test"]
    elif args.split == "val":
        eval_df = splits["val"]
    elif args.split == "train":
        eval_df = splits["train"]
    else:
        raise ValueError(f"Unknown split: {args.split}")

    print(f"Evaluating on {args.split} split: {len(eval_df)} samples")

    # Load normalization stats
    norm_stats_path = Path(args.checkpoint_dir).parent / "norm_stats.json"
    if norm_stats_path.exists():
        print(f"Loading normalization stats from {norm_stats_path}")
        with open(norm_stats_path) as f:
            norm_stats = json.load(f)
    else:
        print("Warning: Normalization stats not found, computing from data")
        norm_stats = None

    # Create dataset
    dataset = SDOTimeseriesDataset(
        eval_df,
        split=args.split,
        resize=RESIZE,
        augment=False,
        norm_stats=norm_stats,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Load model
    print(f"Loading model from {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    model = FlareForecaster(
        num_channels=NUM_CHANNELS,
        spatial_feature_dim=SPATIAL_FEATURE_DIM,
        temporal_num_layers=TEMPORAL_NUM_LAYERS,
        temporal_num_heads=TEMPORAL_NUM_HEADS,
        temporal_dim_feedforward=TEMPORAL_DIM_FEEDFORWARD,
        temporal_dropout=TEMPORAL_DROPOUT,
        temporal_pooling=TEMPORAL_POOLING,
        num_classes=NUM_CLASSES,
        pretrained_spatial=PRETRAINED_SPATIAL,
        freeze_spatial=FREEZE_SPATIAL,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Run evaluation
    print("\nRunning evaluation...")
    predictions, labels, metadata = evaluate(model, dataloader, device)

    print(f"Predictions shape: {predictions.shape}")
    print(f"Labels shape: {labels.shape}")

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(labels, predictions, threshold=args.threshold)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    class_names = ["C+", "M+", "X+"]
    for class_name in class_names:
        print(f"\n{class_name}:")
        print(f"  TSS:     {metrics[f'{class_name}_tss']:.4f}")
        print(
            f"  ROC-AUC: {metrics[f'{class_name}_roc_auc']:.4f}"
            if metrics[f"{class_name}_roc_auc"]
            else "  ROC-AUC: N/A"
        )
        print(
            f"  PR-AUC:  {metrics[f'{class_name}_pr_auc']:.4f}" if metrics[f"{class_name}_pr_auc"] else "  PR-AUC:  N/A"
        )
        print(f"  TPR:     {metrics[f'{class_name}_tpr']:.4f}")
        print(f"  FPR:     {metrics[f'{class_name}_fpr']:.4f}")

    print("\nAverage:")
    print(f"  TSS:     {metrics['avg_tss']:.4f}")
    print(f"  ROC-AUC: {metrics['avg_roc_auc']:.4f}" if metrics["avg_roc_auc"] else "  ROC-AUC: N/A")
    print(f"  PR-AUC:  {metrics['avg_pr_auc']:.4f}" if metrics["avg_pr_auc"] else "  PR-AUC:  N/A")

    # Save metrics
    metrics_path = output_dir / f"metrics_{args.split}.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Save predictions
    if args.save_predictions:
        predictions_df = pd.DataFrame(
            {
                "sample_id": metadata["sample_id"],
                "noaa_ar": metadata["noaa_ar"],
                "date": metadata["date"],
                "pred_c_plus": predictions[:, 0],
                "pred_m_plus": predictions[:, 1],
                "pred_x_plus": predictions[:, 2],
                "label_c_plus": labels[:, 0],
                "label_m_plus": labels[:, 1],
                "label_x_plus": labels[:, 2],
            }
        )

        predictions_path = output_dir / f"predictions_{args.split}.csv"
        predictions_df.to_csv(predictions_path, index=False)
        print(f"Predictions saved to {predictions_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate timeseries flare forecasting model")

    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory containing checkpoint (used to find norm_stats.json)",
    )
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to manifest file")
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "val", "test"], help="Which split to evaluate"
    )
    parser.add_argument(
        "--output_dir", type=str, default="/ARCAFF/ARCCnet/outputs/timeseries/eval", help="Directory for outputs"
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--save_predictions", action="store_true", help="Save predictions to CSV")

    args = parser.parse_args()

    # Infer checkpoint_dir from checkpoint_path if not provided
    if args.checkpoint_dir is None:
        args.checkpoint_dir = Path(args.checkpoint_path).parent

    main(args)
