import random

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

        # Modify the first convolutional layer to accept single-channel (monochrome) images
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the original fully connected layer
        self.resnet.fc = nn.Identity()

        # Additional fully connected layer to project ResNet features into a lower dimension
        self.fc_features = nn.Linear(512, 224)

        # Hierarchical classification heads
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


class TeacherForcingResNet18(nn.Module):
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

        # Modify the first convolutional layer to accept single-channel (monochrome) images
        self.resnet.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

        # Remove the original fully connected layer
        self.resnet.fc = nn.Identity()

        # Additional fully connected layer to project ResNet features into a lower dimension
        self.fc_features = nn.Linear(512, 224)

        # Hierarchical classification heads
        #    Z -> P -> C
        self.fc_Z = nn.Linear(224, num_classes_Z)
        self.fc_P = nn.Linear(224 + num_classes_Z, num_classes_P)
        self.fc_C = nn.Linear(224 + num_classes_Z + num_classes_P, num_classes_C)

    def forward(
        self,
        x: torch.Tensor,
        Z_true: torch.Tensor = None,
        P_true: torch.Tensor = None,
        teacher_forcing_ratio: float = 1.0,
    ) -> tuple:
        """
        Forward pass of the HierarchicalResNet18 model with Teacher Forcing and Scheduled Sampling.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 100, 200).
            Z_true (torch.Tensor, optional): Ground truth labels for Z component. Shape: (B,)
            P_true (torch.Tensor, optional): Ground truth labels for P component. Shape: (B,)
            teacher_forcing_ratio (float, optional): Probability of using ground truth labels. Defaults to 1.0.

        Returns:
            tuple: Contains probabilities for Z, P, and C classification heads.
                   (Z_probs, P_probs, C_probs)
        """
        # Pass through ResNet18 backbone
        features = self.resnet(x)  # shape: (B, 512)

        # Project features to 224 dimensions with ReLU activation
        features_224 = F.relu(self.fc_features(features))  # shape: (B, 224)

        # Z-component prediction
        Z_logits = self.fc_Z(features_224)  # shape: (B, num_classes_Z)
        Z_probs = F.softmax(Z_logits, dim=1)  # shape: (B, num_classes_Z)

        # Decide whether to use ground truth or model prediction for Z
        if self.training and Z_true is not None and random.random() < teacher_forcing_ratio:
            # Use ground truth labels (one-hot encoded)
            Z_input = F.one_hot(Z_true, num_classes=Z_probs.size(1)).float()  # shape: (B, num_classes_Z)
        else:
            # Use model's own predictions
            Z_input = Z_probs  # shape: (B, num_classes_Z)

        # P-component prediction (depends on Z)
        P_input = torch.cat([features_224, Z_input], dim=1)  # shape: (B, 224 + num_classes_Z)
        P_logits = self.fc_P(P_input)  # shape: (B, num_classes_P)
        P_probs = F.softmax(P_logits, dim=1)  # shape: (B, num_classes_P)

        # Decide whether to use ground truth or model prediction for P
        if self.training and P_true is not None and random.random() < teacher_forcing_ratio:
            # Use ground truth labels (one-hot encoded)
            P_input_final = F.one_hot(P_true, num_classes=P_probs.size(1)).float()  # shape: (B, num_classes_P)
        else:
            # Use model's own predictions
            P_input_final = P_probs  # shape: (B, num_classes_P)

        # C-component prediction (depends on Z and P)
        C_input = torch.cat(
            [features_224, Z_input, P_input_final], dim=1
        )  # shape: (B, 224 + num_classes_Z + num_classes_P)
        C_logits = self.fc_C(C_input)  # shape: (B, num_classes_C)
        C_probs = F.softmax(C_logits, dim=1)  # shape: (B, num_classes_C)

        return Z_probs, P_probs, C_probs
