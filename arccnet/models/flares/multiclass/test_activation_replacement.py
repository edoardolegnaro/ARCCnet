# test_activation_replacement.py
"""
Simple test script to verify that ReLU activations are properly replaced with LeakyReLU
in the multiclass model.
"""

import torch.nn as nn


def count_activation_types(model):
    """Count the number of different activation functions in a model."""
    relu_count = 0
    leaky_relu_count = 0
    other_activations = {}

    def count_activations(module):
        nonlocal relu_count, leaky_relu_count, other_activations

        for child in module.children():
            if isinstance(child, nn.ReLU):
                relu_count += 1
            elif isinstance(child, nn.LeakyReLU):
                leaky_relu_count += 1
            elif isinstance(child, (nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.Softmax)):
                act_type = type(child).__name__
                other_activations[act_type] = other_activations.get(act_type, 0) + 1
            else:
                count_activations(child)

    count_activations(model)
    return relu_count, leaky_relu_count, other_activations


def test_activation_replacement():
    """Test the activation replacement functionality."""
    # Import here to avoid circular imports
    from arccnet.models.flares.multiclass.model import FlareClassifier

    # Create a simple test model
    num_classes = 5
    class_names = ["A", "B", "C", "M", "X"]

    print("Creating FlareClassifier model...")
    model = FlareClassifier(num_classes=num_classes, class_names=class_names, class_weights=None)

    print("Counting activation functions...")
    relu_count, leaky_relu_count, other_activations = count_activation_types(model.model)

    print("\nActivation function counts in the model:")
    print(f"ReLU activations: {relu_count}")
    print(f"LeakyReLU activations: {leaky_relu_count}")
    if other_activations:
        print("Other activation functions:")
        for act_type, count in other_activations.items():
            print(f"  {act_type}: {count}")

    if relu_count == 0 and leaky_relu_count > 0:
        print("\n✅ SUCCESS: All ReLU activations have been replaced with LeakyReLU!")
    elif relu_count > 0:
        print(f"\n❌ WARNING: Found {relu_count} ReLU activations that were not replaced.")
    else:
        print(
            "\n⚠️  INFO: No ReLU or LeakyReLU activations found. This might be normal depending on the model architecture."
        )

    # Test that LeakyReLU has the correct negative_slope
    for name, module in model.model.named_modules():
        if isinstance(module, nn.LeakyReLU):
            if abs(module.negative_slope - 0.01) < 1e-6:
                print(f"✅ LeakyReLU at {name} has correct negative_slope: {module.negative_slope}")
            else:
                print(f"❌ LeakyReLU at {name} has incorrect negative_slope: {module.negative_slope} (expected: 0.01)")
            break  # Just check the first one as an example


if __name__ == "__main__":
    test_activation_replacement()
