"""
AIDetect Phase 25: Multi-Generator Manipulation Detector Architecture (V2)
===========================================================================
Dedicated frequency-domain binary classifier trained across multi-generator
manipulation data, distinguishing:
  0 = ORIGINAL_REAL   (Untouched camera photograph)
  1 = AI_MANIPULATED  (Real photograph altered with AI editing/inpainting)

Architecture:
  Native RGB image
        ↓
  Authoritative FFT preprocessing (native-resolution 2D FFT, log1p, min-max norm, resize to 224)
        ↓
  Frequency Tensor [3, 224, 224] (ImageNet-normalized)
        ↓
  ImageNet-pretrained ResNet-50 (ResNet50_Weights.DEFAULT)
        ↓
  2048-dimensional frequency feature representation (Global Average Pooling)
        ↓
  Dropout(p=0.3)
        ↓
  Linear(2048, 2)
        ↓
  Logits [B, 2] (0=ORIGINAL_REAL, 1=AI_MANIPULATED)

Strict research model for Phase 25. Isolated from production V1 models.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

CLASS_MAPPING = {"0": "ORIGINAL_REAL", "1": "AI_MANIPULATED"}
NUM_CLASSES   = 2
FEATURE_DIM   = 2048


class ManipulationFrequencyResNet50V2(nn.Module):
    """
    Multi-Generator Frequency-Domain AI-Manipulation Detector (V2).
    Operates strictly on the 224x224 FFT magnitude representation.
    """
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        in_features = backbone.fc.in_features  # 2048
        
        # Replace classification head with Dropout + Linear(2048, 2)
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, NUM_CLASSES)
        )
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input frequency tensor [B, 3, 224, 224]
        Returns:
            Logits tensor [B, 2]
        """
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dimensional pre-classifier frequency features."""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

