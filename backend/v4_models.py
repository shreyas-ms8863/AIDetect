"""
AIDetect V4 Model Architectures
=================================
Three binary classifiers for REAL vs AI-GENERATED detection:

A) SpatialV4   -- pretrained ResNet-50 spatial feature extractor + linear head
B) FrequencyV4 -- pretrained ResNet-50 on FFT frequency spectrum + linear head
C) HybridV4    -- pretrained ResNet-50 spatial branch (2048-d) +
                  frequency ConvNet branch (256-d) +
                  feature concatenation (2304-d) +
                  linear classifier

Label convention:  0 = REAL,  1 = AI

Architecture notes:
  - ResNet50_Weights.DEFAULT used for all pretrained branches.
  - Hybrid frequency branch: 4-layer ConvNet (consistent with V3) for comparability.
  - Hybrid fusion: explicit FEATURE CONCATENATION (not gated or attention-based).
  - All models output raw logits [batch, 2]; softmax applied externally.

V3 preservation:
  - V4 HybridV4 shares the same architecture signature as V3's HybridResNet50FFT
    in main.py, so checkpoint loading code remains compatible.
  - V3 checkpoints are NOT loaded or modified by this file.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

SPATIAL_FEATURE_DIM   = 2048   # ResNet-50 avgpool output
FREQUENCY_FEATURE_DIM = 256    # Frequency ConvNet output
FUSED_DIM             = SPATIAL_FEATURE_DIM + FREQUENCY_FEATURE_DIM  # 2304
NUM_CLASSES           = 2      # 0=REAL, 1=AI

# ---------------------------------------------------------------------------
# A) SPATIAL V4
# ---------------------------------------------------------------------------

class SpatialV4(nn.Module):
    """
    Spatial V4 model.

    Architecture:
        Pretrained ResNet-50 (ResNet50_Weights.DEFAULT)
        -> Replace fc: Linear(2048, 2)

    Input:  spatial image tensor [B, 3, 224, 224] (ImageNet-normalised)
    Output: logits [B, 2]  (0=REAL, 1=AI)
    """

    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)
        self.model = backbone

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# B) FREQUENCY V4
# ---------------------------------------------------------------------------

class FrequencyV4(nn.Module):
    """
    Frequency V4 model.

    Architecture:
        Pretrained ResNet-50 (ResNet50_Weights.DEFAULT)
        -> Replace fc: Linear(2048, 2)

    Input:  FFT magnitude tensor [B, 3, 224, 224] (via AuthoritativeFFTTransform)
    Output: logits [B, 2]  (0=REAL, 1=AI)
    """

    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, NUM_CLASSES)
        self.model = backbone

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# C) HYBRID V4
# ---------------------------------------------------------------------------

class HybridV4(nn.Module):
    """
    Hybrid V4 model.

    Architecture:
        Spatial branch:
            Pretrained ResNet-50 backbone (ResNet50_Weights.DEFAULT)
            -> GlobalAvgPool -> 2048-d spatial feature vector

        Frequency branch:
            4-layer ConvNet on FFT magnitude spectrum
            -> AdaptiveAvgPool -> 256-d frequency feature vector

        Fusion:
            FEATURE CONCATENATION: [2048, 256] -> 2304-d combined vector
            (Explicit concatenation, NOT gated or attention fusion.)

        Classifier:
            Linear(2304, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 2)

    Spatial input:   [B, 3, 224, 224] (ImageNet-normalised spatial image)
    Frequency input: [B, 3, 224, 224] (AuthoritativeFFTTransform output)
    Output:          logits [B, 2]  (0=REAL, 1=AI)
    """

    def __init__(self):
        super().__init__()

        # SPATIAL BRANCH: pretrained ResNet-50 (strip final FC layer)
        spatial_backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.spatial_features = nn.Sequential(
            *list(spatial_backbone.children())[:-1]
        )
        # Output: [B, 2048, 1, 1] -> flatten to [B, 2048]

        # FREQUENCY BRANCH: 4-layer ConvNet (consistent with V3)
        self.frequency_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 224->112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 112->56

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 56->28

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),        # -> [B, 256, 1, 1]
        )
        # Output: [B, 256, 1, 1] -> flatten to [B, 256]

        # FUSION + CLASSIFIER
        # Fusion: explicit feature concatenation (2048 + 256 = 2304)
        self.classifier = nn.Sequential(
            nn.Linear(FUSED_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, NUM_CLASSES),
        )

    def forward(self, spatial_input, freq_input):
        """
        Args:
            spatial_input: [B, 3, 224, 224] -- spatial image tensor
            freq_input:    [B, 3, 224, 224] -- FFT magnitude tensor
        Returns:
            logits [B, 2]
        """
        s_feat   = self.spatial_features(spatial_input)
        s_feat   = torch.flatten(s_feat, 1)                # [B, 2048]

        f_feat   = self.frequency_features(freq_input)
        f_feat   = torch.flatten(f_feat, 1)                # [B, 256]

        combined = torch.cat([s_feat, f_feat], dim=1)      # [B, 2304]
        return self.classifier(combined)                   # [B, 2]


# ---------------------------------------------------------------------------
# ARCHITECTURE DOCUMENTATION (for checkpoint embedding)
# ---------------------------------------------------------------------------

ARCHITECTURE_DOC = {
    "SpatialV4": {
        "backbone": "ResNet-50 (ResNet50_Weights.DEFAULT)",
        "head": "Linear(2048, 2)",
        "input": "[B, 3, 224, 224] spatial image (ImageNet-normalised)",
        "output": "logits [B, 2]",
    },
    "FrequencyV4": {
        "backbone": "ResNet-50 (ResNet50_Weights.DEFAULT)",
        "head": "Linear(2048, 2)",
        "input": "[B, 3, 224, 224] FFT magnitude (AuthoritativeFFTTransform)",
        "output": "logits [B, 2]",
    },
    "HybridV4": {
        "spatial_branch": "ResNet-50 backbone (ResNet50_Weights.DEFAULT) -> 2048-d",
        "frequency_branch": "4-layer ConvNet (32->64->128->256, AdaptiveAvgPool) -> 256-d",
        "fusion": "Feature concatenation: 2048 + 256 = 2304-d (NOT gated/attention)",
        "classifier": "Linear(2304, 512) -> ReLU -> Dropout(0.3) -> Linear(512, 2)",
        "spatial_input": "[B, 3, 224, 224] spatial image (ImageNet-normalised)",
        "freq_input": "[B, 3, 224, 224] FFT magnitude (AuthoritativeFFTTransform)",
        "output": "logits [B, 2]",
        "fusion_type": "explicit_feature_concatenation",
    },
}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dummy_spatial = torch.randn(2, 3, 224, 224).to(device)
    dummy_freq    = torch.randn(2, 3, 224, 224).to(device)

    for ModelClass, name, takes_both in [
        (SpatialV4,   "SpatialV4",   False),
        (FrequencyV4, "FrequencyV4", False),
        (HybridV4,    "HybridV4",    True),
    ]:
        model = ModelClass().to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy_spatial, dummy_freq) if takes_both else model(dummy_spatial)
        print("{}: input {} -> output {}".format(name, tuple(dummy_spatial.shape), tuple(out.shape)))
        assert out.shape == (2, 2), "Expected (2,2), got {}".format(out.shape)

    assert FUSED_DIM == 2304, "Expected 2304, got {}".format(FUSED_DIM)
    print("\nHybrid fused dim: {} + {} = {}  OK".format(
        SPATIAL_FEATURE_DIM, FREQUENCY_FEATURE_DIM, FUSED_DIM))
    print("All architecture checks passed.")
