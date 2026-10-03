"""
AIDetect Phase 8: Hybrid AI-Manipulation Detector Architecture
==============================================================
Dual-branch deep architecture combining:
  1. Spatial Branch: ResNet-50 on RGB image -> 2048 features
  2. Frequency Branch: ResNet-50 on native-resolution 2D FFT spectrum -> 2048 features

Concatenation:
  [Spatial (2048) || Frequency (2048)] = 4096-dimensional joint representation

Fusion Classification Head:
  Linear(4096, 512) -> ReLU -> Dropout(p=0.3) -> Linear(512, 2)

Classes:
  0 = ORIGINAL_REAL   (Untouched camera photograph)
  1 = AI_MANIPULATED  (Real photograph altered with AI editing/inpainting)

Fairness & Independence:
  - Both ResNet-50 backbones are initialized independently from ResNet50_Weights.DEFAULT.
  - No weights or checkpoints are copied from Phase 6 or Phase 7.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

CLASS_MAPPING = {"0": "ORIGINAL_REAL", "1": "AI_MANIPULATED"}
NUM_CLASSES   = 2
BRANCH_DIM    = 2048
JOINT_DIM     = 4096
FUSION_HIDDEN = 512

class ManipulationHybridResNet50V1(nn.Module):
    """
    Dedicated Two-Stream Hybrid AI-Manipulation Detector.
    Processes Spatial RGB and Native-FFT Frequency tensors concurrently.
    """
    def __init__(self, dropout_p: float = 0.3):
        super().__init__()
        
        # 1. Spatial stream backbone (independent ImageNet pretrained ResNet-50)
        spatial_bb = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.spatial_backbone = nn.Sequential(*list(spatial_bb.children())[:-1])  # Output: [B, 2048, 1, 1]
        
        # 2. Frequency stream backbone (independent ImageNet pretrained ResNet-50)
        freq_bb = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.freq_backbone = nn.Sequential(*list(freq_bb.children())[:-1])        # Output: [B, 2048, 1, 1]
        
        # 3. Concatenation & Fusion head: 4096 -> 512 -> ReLU -> Dropout -> 2
        self.fusion_head = nn.Sequential(
            nn.Linear(JOINT_DIM, FUSION_HIDDEN),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Linear(FUSION_HIDDEN, NUM_CLASSES),
        )

    def extract_branch_features(self, x_spatial: torch.Tensor, x_freq: torch.Tensor):
        """
        Extract spatial (2048) and frequency (2048) features along with concatenated (4096).
        """
        f_spatial = torch.flatten(self.spatial_backbone(x_spatial), 1)
        f_freq    = torch.flatten(self.freq_backbone(x_freq), 1)
        f_joint   = torch.cat([f_spatial, f_freq], dim=1)
        return f_spatial, f_freq, f_joint

    def forward(self, x_spatial: torch.Tensor, x_freq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_spatial: [B, 3, 224, 224] Spatial RGB tensor
            x_freq:    [B, 3, 224, 224] Frequency FFT spectrum tensor
        Returns:
            Logits:    [B, 2] (0=ORIGINAL_REAL, 1=AI_MANIPULATED)
        """
        _, _, f_joint = self.extract_branch_features(x_spatial, x_freq)
        logits = self.fusion_head(f_joint)
        return logits

ARCHITECTURE_DOC = {
    "model_name": "ManipulationHybridResNet50V1",
    "spatial_backbone": "ResNet-50 (ResNet50_Weights.DEFAULT, output 2048)",
    "frequency_backbone": "ResNet-50 (ResNet50_Weights.DEFAULT, output 2048)",
    "concatenation_dim": JOINT_DIM,
    "fusion_head": "Sequential(Linear(4096, 512), ReLU(), Dropout(0.3), Linear(512, 2))",
    "classes": CLASS_MAPPING,
    "task": "Binary detection: ORIGINAL_REAL vs AI_MANIPULATED (Spatial + Frequency Fusion)",
    "input_resolutions": {
        "spatial": [3, 224, 224],
        "frequency": [3, 224, 224]
    }
}

if __name__ == "__main__":
    m = ManipulationHybridResNet50V1()
    m.eval()
    dummy_s = torch.randn(2, 3, 224, 224)
    dummy_f = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        fs, ff, fj = m.extract_branch_features(dummy_s, dummy_f)
        logits = m(dummy_s, dummy_f)
    print("ManipulationHybridResNet50V1 Forward Check:")
    print("  Spatial features shape :", tuple(fs.shape))
    print("  Frequency feat shape   :", tuple(ff.shape))
    print("  Joint features shape   :", tuple(fj.shape))
    print("  Output logits shape    :", tuple(logits.shape))
    assert fs.shape == (2, 2048)
    assert ff.shape == (2, 2048)
    assert fj.shape == (2, 4096)
    assert logits.shape == (2, 2)
    print("All hybrid architecture checks passed.")

