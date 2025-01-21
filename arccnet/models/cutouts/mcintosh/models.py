import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class HierarchicalResNet18(nn.Module):
    def __init__(self, num_classes_Z: int, num_classes_P: int, num_classes_C: int):
        """
        Hierarchical ResNet18 model for multi-level classification.

        Args:
            num_classes_Z (int): Number of classes for the Z component.
            num_classes_P (int): Number of classes for the P component.
            num_classes_C (int): Number of classes for the C component.
        """
        super().__init__()

        # Load the pre-trained ResNet18 model
        weights = ResNet18_Weights.DEFAULT
        self.resnet = models.resnet18(weights=weights)

        # 1) Modify the first convolutional layer to accept single-channel (monochrome) images
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # 2) Remove the original fully connected layer
        self.resnet.fc = nn.Identity()

        # 3) Additional fully connected layer to project ResNet features into a lower dimension
        self.fc_features = nn.Linear(512, 224)

        # 4) Hierarchical classification heads
        #    Z -> P -> C
        self.fc_Z = nn.Linear(224, num_classes_Z)
        self.fc_P = nn.Linear(224 + num_classes_Z, num_classes_P)
        self.fc_C = nn.Linear(224 + num_classes_Z + num_classes_P, num_classes_C)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the HierarchicalResNet18 model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 100, 200).

        Returns:
            tuple: Contains raw logits for Z, P, and C classification heads.
                   (Z_logits, P_logits, C_logits)
        """
        # Pass through ResNet18 backbone
        features = self.resnet(x)  # shape: (B, 512)

        # Project features to 224 dimensions with ReLU activation
        features_224 = F.relu(self.fc_features(features))  # shape: (B, 224)

        # Z-component prediction
        Z_logits = self.fc_Z(features_224)  # shape: (B, num_classes_Z)
        Z_probs = F.softmax(Z_logits, dim=1)

        # P-component prediction (depends on Z logits)
        P_input = torch.cat([features_224, Z_probs], dim=1)  # shape: (B, 224 + num_classes_Z)
        P_logits = self.fc_P(P_input)  # shape: (B, num_classes_P)
        P_probs = F.softmax(P_logits, dim=1)

        # C-component prediction (depends on Z and P logits)
        C_input = torch.cat([features_224, Z_probs, P_probs], dim=1)  # shape: (B, 224 + num_classes_Z + num_classes_P)
        C_logits = self.fc_C(C_input)  # shape: (B, num_classes_C)
        C_probs = F.softmax(C_logits, dim=1)

        return Z_probs, P_probs, C_probs
