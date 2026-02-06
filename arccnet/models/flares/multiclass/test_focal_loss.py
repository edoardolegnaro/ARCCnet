#!/usr/bin/env python3
"""
Test script to verify the focal loss implementation works correctly.
"""

import os
import sys

import torch
import torch.nn.functional as F

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))

from arccnet.models.flares.multiclass.model import FocalLoss


def test_focal_loss():
    """Test the focal loss implementation."""
    print("Testing Focal Loss implementation...")

    # Test parameters
    batch_size = 8
    num_classes = 3
    alpha = 1.0
    gamma = 2.0

    # Create test data
    logits = torch.randn(batch_size, num_classes, requires_grad=True)
    targets = torch.randint(0, num_classes, (batch_size,))

    # Initialize focal loss
    focal_loss = FocalLoss(alpha=alpha, gamma=gamma)

    # Compute focal loss
    loss = focal_loss(logits, targets)

    print(f"✅ Focal loss computed successfully: {loss.item():.4f}")

    # Test with class weights
    class_weights = torch.tensor([0.5, 1.0, 2.0])  # Example weights for imbalanced classes
    weighted_focal_loss = FocalLoss(alpha=alpha, gamma=gamma, weight=class_weights)
    weighted_loss = weighted_focal_loss(logits, targets)

    print(f"✅ Weighted focal loss computed successfully: {weighted_loss.item():.4f}")

    # Compare with standard cross entropy
    ce_loss = F.cross_entropy(logits, targets)
    print(f"📊 Standard CE loss: {ce_loss.item():.4f}")
    print(f"📊 Focal loss: {loss.item():.4f}")
    print(f"📊 Weighted focal loss: {weighted_loss.item():.4f}")

    # Test gradient computation
    loss.backward()
    print("✅ Gradients computed successfully")

    # Test with easy vs hard examples
    print("\n🔍 Testing focal loss behavior on easy vs hard examples:")

    # Easy example (high confidence correct prediction)
    easy_logits = torch.tensor([[10.0, -5.0, -5.0]])  # Very confident about class 0
    easy_targets = torch.tensor([0])

    # Hard example (low confidence correct prediction)
    hard_logits = torch.tensor([[0.1, -0.05, -0.05]])  # Less confident about class 0
    hard_targets = torch.tensor([0])

    easy_loss = focal_loss(easy_logits, easy_targets)
    hard_loss = focal_loss(hard_logits, hard_targets)

    print(f"Easy example loss: {easy_loss.item():.6f}")
    print(f"Hard example loss: {hard_loss.item():.6f}")
    print(f"Ratio (hard/easy): {(hard_loss / easy_loss).item():.2f}")

    if hard_loss > easy_loss:
        print("✅ Focal loss correctly focuses more on hard examples!")
    else:
        print("❌ Focal loss behavior unexpected")

    return True


def test_different_gamma_values():
    """Test focal loss with different gamma values."""
    print("\n🔍 Testing different gamma values:")

    logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 2.0, 0.5], [0.5, 0.1, 2.0], [1.0, 1.0, 1.0]])
    targets = torch.tensor([0, 1, 2, 0])

    gamma_values = [0.0, 0.5, 1.0, 2.0, 5.0]

    for gamma in gamma_values:
        focal_loss = FocalLoss(alpha=1.0, gamma=gamma)
        loss = focal_loss(logits, targets)
        print(f"Gamma {gamma}: Loss = {loss.item():.4f}")

    print("✅ Different gamma values tested successfully")


if __name__ == "__main__":
    print("🚀 Starting Focal Loss Tests...")

    try:
        test_focal_loss()
        test_different_gamma_values()
        print("\n🎉 All focal loss tests passed!")
    except Exception as e:
        print(f"\n❌ Focal loss test failed: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
