"""Tests for evaluation metric contracts."""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from arccnet.models.timeseries.evaluate import compute_multiclass_metrics


def test_multiclass_metric_contract():
    """compute_multiclass_metrics should accept 1D integer labels and return operational keys."""
    y_true = np.array([0, 1, 2, 1, 0, 2], dtype=np.int64)
    probs = np.array(
        [
            [0.9, 0.08, 0.02],
            [0.1, 0.8, 0.1],
            [0.05, 0.2, 0.75],
            [0.15, 0.7, 0.15],
            [0.75, 0.2, 0.05],
            [0.1, 0.3, 0.6],
        ],
        dtype=np.float32,
    )

    metrics = compute_multiclass_metrics(y_true, probs, threshold=0.5)
    required = {"accuracy", "balanced_accuracy", "m_plus_tss", "m_plus_hss", "x_plus_pr_auc", "x_plus_tss"}
    assert required.issubset(metrics.keys()), f"Missing expected metric keys: {required - set(metrics.keys())}"


if __name__ == "__main__":
    test_multiclass_metric_contract()
    print("\n✅ Evaluation metric tests passed!")
