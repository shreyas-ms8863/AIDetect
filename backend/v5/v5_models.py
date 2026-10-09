"""
AIDetect V5 Models Module
=========================
Defines the authoritative model architectures for V5:
- GatedResidualResNet50: 3-branch gated fusion of Spatial, Frequency, and Noise Residual features.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50


class GatedResidualResNet50(nn.Module):
    """
    V5-D: Gated Spatial + Frequency + Noise Residual ResNet-50.
    Triple-backbone architecture fusing:
      1. Spatial ResNet-50 features (2048-d)
      2. Frequency ResNet-50 features (2048-d)
      3. Noise Residual ResNet-50 features (2048-d)
    Via learned feature gating layer (6144 -> 6144 -> 6144 -> Sigmoid).
    Final classification head maps 6144-d -> 2 classes (REAL=0, AI=1).
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes

        # Spatial stream
        self.spatial_encoder = resnet50(weights=None)
        spatial_dim = self.spatial_encoder.fc.in_features
        self.spatial_encoder.fc = nn.Identity()

        # Frequency stream
        self.frequency_encoder = resnet50(weights=None)
        frequency_dim = self.frequency_encoder.fc.in_features
        self.frequency_encoder.fc = nn.Identity()

        # Noise residual stream
        self.residual_encoder = resnet50(weights=None)
        residual_dim = self.residual_encoder.fc.in_features
        self.residual_encoder.fc = nn.Identity()

        self.fused_dim = spatial_dim + frequency_dim + residual_dim

        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.fused_dim, self.fused_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.fused_dim, self.fused_dim),
            nn.Sigmoid(),
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.30),
            nn.Linear(self.fused_dim, num_classes),
        )

    def forward(
        self, spatial: torch.Tensor, frequency: torch.Tensor, residual: torch.Tensor
    ) -> torch.Tensor:
        spatial_features = self.spatial_encoder(spatial)
        frequency_features = self.frequency_encoder(frequency)
        residual_features = self.residual_encoder(residual)

        combined = torch.cat(
            [spatial_features, frequency_features, residual_features],
            dim=1,
        )

        gate = self.gate(combined)
        gated_features = combined * gate

        return self.classifier(gated_features)

    def forward_with_gates(
        self, spatial: torch.Tensor, frequency: torch.Tensor, residual: torch.Tensor
    ):
        spatial_features = self.spatial_encoder(spatial)
        frequency_features = self.frequency_encoder(frequency)
        residual_features = self.residual_encoder(residual)

        combined = torch.cat(
            [spatial_features, frequency_features, residual_features],
            dim=1,
        )

        gate = self.gate(combined)
        gated_features = combined * gate
        logits = self.classifier(gated_features)

        return logits, gate

