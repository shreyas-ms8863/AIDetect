"""
AIDetect Phase 6: Dedicated AI-Manipulation Detector Architecture
==================================================================
Binary classifier distinguishing:
  0 = ORIGINAL_REAL   (Untouched camera photograph)
  1 = AI_MANIPULATED  (Real photograph altered with AI editing/inpainting)

Architecture:
  Image [3, 224, 224] (ImageNet-normalized)
    ↓
  ImageNet-pretrained ResNet-50 (ResNet50_Weights.DEFAULT)
    ↓
  2048-dimensional feature representation (Global Average Pooling)
    ↓
  Dropout(p=0.3)
    ↓
  Linear(2048, 2)
    ↓
  Logits [B, 2] (0=ORIGINAL_REAL, 1=AI_MANIPULATED)

Note: Completely separate from the V4 full-generation detection models.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

CLASS_MAPPING = {"0": "ORIGINAL_REAL", "1": "AI_MANIPULATED"}
NUM_CLASSES   = 2
FEATURE_DIM   = 2048

class ManipulationResNet50V1(nn.Module):
    """
    Dedicated AI-Manipulation detector backbone based on pretrained ResNet-50.
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
            x: Input tensor [B, 3, 224, 224]
        Returns:
            Logits tensor [B, 2]
        """
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-dimensional pre-classifier features."""
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        return torch.flatten(x, 1)

ARCHITECTURE_DOC = {
    "model_name": "ManipulationResNet50V1",
    "backbone": "ResNet-50 (ResNet50_Weights.DEFAULT)",
    "feature_dim": FEATURE_DIM,
    "head": "Sequential(Dropout(p=0.3), Linear(2048, 2))",
    "classes": CLASS_MAPPING,
    "task": "Binary detection: ORIGINAL_REAL vs AI_MANIPULATED",
    "input_resolution": [3, 224, 224]
}

if __name__ == "__main__":
    m = ManipulationResNet50V1()
    m.eval()
    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = m(dummy)
        feat = m.extract_features(dummy)
    print("ManipulationResNet50V1 Forward Check:")
    print("  Input shape   :", tuple(dummy.shape))
    print("  Output logits :", tuple(out.shape))
    print("  Features shape:", tuple(feat.shape))
    assert out.shape == (2, 2)
    assert feat.shape == (2, 2048)
    print("All architecture checks passed.")

