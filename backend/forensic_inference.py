"""
AIDetect Phase 10: Unified Forensic Inference Pipeline
======================================================
Orchestrates dual independent frozen models:
  1. Generation Detector: V4 Frequency/Hybrid ResNet-50
  2. Manipulation Detector: Phase 7 Frequency ResNet-50
  3. Integrated Forensic Decision Engine

Deterministic Preprocessing:
  - Both models execute native-resolution FFT magnitude preprocessing,
    preserving high-frequency forensic artifacts before any downsampling.
  - Softmax probabilities are extracted independently without cross-model tuning.
  - The final interpretation is produced by ForensicDecisionEngine.
"""

import os
import sys
import io
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Union, Optional

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v4_models import FrequencyV4, SpatialV4, HybridV4
from v4_dataset import FFT_TRANSFORM as V4_FFT_TRANSFORM, SPATIAL_VAL_XFORM as V4_SPATIAL_VAL_XFORM
from manipulation_freq_v1_models import ManipulationFrequencyResNet50V1
from manipulation_freq_v1_dataset import AuthoritativeFrequencyTransform as ManipFrequencyTransform
from forensic_decision_engine import ForensicDecisionEngine

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_DIR  = BASE_DIR.parent
MODEL_DIR    = PROJECT_DIR / "models"

# ---------------------------------------------------------------------------
# DEFAULT FROZEN CHECKPOINT PATHS
# ---------------------------------------------------------------------------

DEFAULT_GEN_CKPT   = MODEL_DIR / "hybrid_resnet50_fft_v4.pth"
DEFAULT_MANIP_CKPT = MODEL_DIR / "manipulation_frequency_resnet50_v1.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ForensicInferencePipeline:
    """
    Unified forensic pipeline hosting both frozen detectors and the decision engine.
    """
    def __init__(
        self,
        generation_ckpt: Optional[Union[str, Path]] = None,
        manipulation_ckpt: Union[str, Path] = DEFAULT_MANIP_CKPT,
        generation_model_type: str = "hybrid",
        device: Optional[torch.device] = None,
        min_confidence: float = 0.55,
        strategy: str = "calibrated",
    ):
        self.device = device or DEVICE
        self.gen_model_type = generation_model_type.lower()
        self.strategy = strategy

        if generation_ckpt is None:
            if self.gen_model_type == "hybrid":
                self.gen_ckpt_path = MODEL_DIR / "hybrid_resnet50_fft_v4.pth"
            elif self.gen_model_type == "spatial":
                self.gen_ckpt_path = MODEL_DIR / "spatial_resnet50_v4.pth"
            else:
                self.gen_ckpt_path = MODEL_DIR / "frequency_resnet50_v4.pth"
        else:
            self.gen_ckpt_path = Path(generation_ckpt)

        self.manip_ckpt_path = Path(manipulation_ckpt)
        
        # Verify checkpoints exist
        if not self.gen_ckpt_path.exists():
            raise FileNotFoundError(f"Generation checkpoint not found: {self.gen_ckpt_path}")
        if not self.manip_ckpt_path.exists():
            raise FileNotFoundError(f"Manipulation checkpoint not found: {self.manip_ckpt_path}")
            
        # 1. Load V4 Generation Model
        self.generation_model = self._load_generation_model()
        
        # 2. Load Phase 7 Manipulation Model
        self.manipulation_model = self._load_manipulation_model()
        
        # Preprocessors
        self.v4_fft_xform = V4_FFT_TRANSFORM
        self.v4_spatial_xform = V4_SPATIAL_VAL_XFORM
        self.manip_fft_xform = ManipFrequencyTransform()
        
        # Decision Engine (supports both 'v1_baseline' and 'v2_calibrated')
        self.engine = ForensicDecisionEngine(min_confidence=min_confidence, strategy=strategy)

    def _load_generation_model(self) -> torch.nn.Module:
        if self.gen_model_type == "frequency":
            model = FrequencyV4()
        elif self.gen_model_type == "spatial":
            model = SpatialV4()
        elif self.gen_model_type == "hybrid":
            model = HybridV4()
        else:
            raise ValueError(f"Unknown generation model type: {self.gen_model_type}")

        ckpt = torch.load(self.gen_ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        cleaned_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(cleaned_sd, strict=True)
        model.to(self.device).eval()
        return model

    def _load_manipulation_model(self) -> torch.nn.Module:
        model = ManipulationFrequencyResNet50V1(dropout_p=0.3)
        ckpt = torch.load(self.manip_ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        cleaned_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(cleaned_sd, strict=True)
        model.to(self.device).eval()
        return model

    def predict(self, image_input: Union[Image.Image, str, Path, bytes], strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute independent inference across both detectors and evaluate decision rules.

        Args:
            image_input: PIL Image, path to image file, or raw bytes.
            strategy: Optional decision strategy override ('v1_baseline' or 'v2_calibrated').

        Returns:
            Dict containing:
                - generation: output from Generation detector
                - manipulation: output from Manipulation detector
                - final: integrated decision (label, confidence, reason, decision_case, strategy)
        """
        # Load PIL image
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as raw:
                img = raw.convert("RGB")
        elif isinstance(image_input, bytes):
            with Image.open(io.BytesIO(image_input)) as raw:
                img = raw.convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        t0 = time.time()
        
        # -------------------------------------------------------------------
        # 1. GENERATION INFERENCE
        # -------------------------------------------------------------------
        with torch.no_grad():
            if self.gen_model_type == "frequency":
                tensor_gen = self.v4_fft_xform(img).unsqueeze(0).to(self.device)
                logits_gen = self.generation_model(tensor_gen)
            elif self.gen_model_type == "spatial":
                tensor_gen = self.v4_spatial_xform(img).unsqueeze(0).to(self.device)
                logits_gen = self.generation_model(tensor_gen)
            else:  # hybrid
                t_s = self.v4_spatial_xform(img).unsqueeze(0).to(self.device)
                t_f = self.v4_fft_xform(img).unsqueeze(0).to(self.device)
                logits_gen = self.generation_model(t_s, t_f)

            probs_gen = torch.softmax(logits_gen, dim=1).cpu()[0]
            p_gen_real = float(probs_gen[0].item())
            p_gen_ai   = float(probs_gen[1].item())

        gen_dict = {
            "model_name": f"V4_{self.gen_model_type.capitalize()}_Generation_Detector",
            "probability_real": round(p_gen_real, 4),
            "probability_ai_generated": round(p_gen_ai, 4),
            "label": "AI_GENERATED" if p_gen_ai >= 0.50 else "REAL",
            "confidence": round(max(p_gen_real, p_gen_ai), 4),
        }

        # -------------------------------------------------------------------
        # 2. MANIPULATION INFERENCE (Phase 7 Frequency Detector)
        # -------------------------------------------------------------------
        with torch.no_grad():
            tensor_manip = self.manip_fft_xform(img).unsqueeze(0).to(self.device)
            logits_manip = self.manipulation_model(tensor_manip)
            probs_manip = torch.softmax(logits_manip, dim=1).cpu()[0]
            p_manip_orig = float(probs_manip[0].item())
            p_manip_ai   = float(probs_manip[1].item())

        manip_dict = {
            "model_name": "Phase7_Frequency_Manipulation_Detector",
            "probability_original": round(p_manip_orig, 4),
            "probability_ai_manipulated": round(p_manip_ai, 4),
            "label": "AI_MANIPULATED" if p_manip_ai >= 0.50 else "ORIGINAL_REAL",
            "confidence": round(max(p_manip_orig, p_manip_ai), 4),
        }

        # -------------------------------------------------------------------
        # 3. FORENSIC DECISION ENGINE INTEGRATION
        # -------------------------------------------------------------------
        decision = self.engine.decide(gen_dict, manip_dict, strategy=strategy)
        decision["elapsed_seconds"] = round(time.time() - t0, 3)
        return decision

if __name__ == "__main__":
    print("=" * 70)
    print("TESTING FORENSIC INFERENCE PIPELINE")
    print("=" * 70)
    pipeline = ForensicInferencePipeline()
    print("Pipeline initialized successfully.")
    
    # Run test on a synthetic dummy image
    test_img = Image.new("RGB", (300, 300), color=(128, 128, 128))
    result = pipeline.predict(test_img)
    print("Dummy image prediction result:")
    print("  Generation  :", result["generation"])
    print("  Manipulation:", result["manipulation"])
    print("  Final       :", result["final"])
    assert result["final"]["label"] in {"REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED", "UNCERTAIN"}
    print("Inference check passed.")
