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
from domain_analyzer import analyze_image_domain

BASE_DIR     = Path(__file__).resolve().parent
PROJECT_DIR  = BASE_DIR.parent
MODEL_DIR    = PROJECT_DIR / "models"

# ---------------------------------------------------------------------------
# DEFAULT FROZEN CHECKPOINT PATHS
# ---------------------------------------------------------------------------

DEFAULT_GEN_CKPT     = MODEL_DIR / "frequency_resnet50_v4.pth"
DEFAULT_FREQ_CKPT    = MODEL_DIR / "frequency_resnet50_v4.pth"
DEFAULT_SPATIAL_CKPT = MODEL_DIR / "spatial_resnet50_v4.pth"
DEFAULT_HYBRID_CKPT  = MODEL_DIR / "hybrid_resnet50_fft_v4.pth"
DEFAULT_MANIP_CKPT   = MODEL_DIR / "manipulation_frequency_resnet50_v1.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ForensicInferencePipeline:
    """
    Unified forensic pipeline hosting frozen generation and manipulation detectors
    and the integrated decision engine with multi-stream consensus.
    """
    def __init__(
        self,
        generation_ckpt: Optional[Union[str, Path]] = None,
        manipulation_ckpt: Union[str, Path] = DEFAULT_MANIP_CKPT,
        generation_model_type: str = "frequency",
        device: Optional[torch.device] = None,
        min_confidence: float = 0.55,
        strategy: str = "calibrated",
    ):
        self.device = device or DEVICE
        self.strategy = strategy
        self.gen_model_type = generation_model_type.lower()

        # Checkpoint paths
        self.freq_ckpt_path = Path(generation_ckpt) if (generation_ckpt and self.gen_model_type == "frequency") else DEFAULT_FREQ_CKPT
        self.spatial_ckpt_path = Path(generation_ckpt) if (generation_ckpt and self.gen_model_type == "spatial") else DEFAULT_SPATIAL_CKPT
        self.hybrid_ckpt_path = Path(generation_ckpt) if (generation_ckpt and self.gen_model_type == "hybrid") else DEFAULT_HYBRID_CKPT
        self.manip_ckpt_path = Path(manipulation_ckpt)

        # Verify checkpoints exist
        for name, p in [
            ("Frequency V4", self.freq_ckpt_path),
            ("Spatial V4", self.spatial_ckpt_path),
            ("Hybrid V4", self.hybrid_ckpt_path),
            ("Manipulation V1", self.manip_ckpt_path),
        ]:
            if not p.exists():
                raise FileNotFoundError(f"{name} checkpoint not found: {p}")

        # 1. Load V4 Generation Models
        self.freq_model = self._load_model(FrequencyV4(), self.freq_ckpt_path)
        self.spatial_model = self._load_model(SpatialV4(), self.spatial_ckpt_path)
        self.hybrid_model = self._load_model(HybridV4(), self.hybrid_ckpt_path)
        self.generation_model = self.freq_model  # Primary detector for Strategy E

        # 2. Load Phase 7 Manipulation Model
        self.manipulation_model = self._load_model(
            ManipulationFrequencyResNet50V1(dropout_p=0.3), self.manip_ckpt_path
        )

        # Preprocessors
        self.v4_fft_xform = V4_FFT_TRANSFORM
        self.v4_spatial_xform = V4_SPATIAL_VAL_XFORM
        self.manip_fft_xform = ManipFrequencyTransform()

        # Decision Engine (supports both 'v1_baseline' and 'v2_calibrated')
        self.engine = ForensicDecisionEngine(min_confidence=min_confidence, strategy=strategy)

    def _load_model(self, model: torch.nn.Module, ckpt_path: Path) -> torch.nn.Module:
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        cleaned_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
        model.load_state_dict(cleaned_sd, strict=True)
        model.to(self.device).eval()
        return model

    def predict(self, image_input: Union[Image.Image, str, Path, bytes], strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute independent multi-stream inference across generation and manipulation
        detectors and evaluate decision engine rules.

        Args:
            image_input: PIL Image, path to image file, or raw bytes.
            strategy: Optional decision strategy override ('v1_baseline' or 'v2_calibrated').

        Returns:
            Dict containing:
                - generation: output from authoritative Frequency Generation detector
                - manipulation: output from Manipulation detector
                - auxiliary: multi-stream probabilities (spatial, freq, hybrid, manip, orig)
                - final: integrated decision (label, confidence, reason, decision_case, strategy, decision_source)
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
        # 1. MULTI-STREAM FORWARD PASSES
        # -------------------------------------------------------------------
        with torch.no_grad():
            tensor_spatial = self.v4_spatial_xform(img).unsqueeze(0).to(self.device)
            tensor_fft = self.v4_fft_xform(img).unsqueeze(0).to(self.device)
            tensor_manip = self.manip_fft_xform(img).unsqueeze(0).to(self.device)

            # A. Frequency V4 (Authoritative generation feature for Strategy E)
            logits_freq = self.freq_model(tensor_fft)
            probs_freq = torch.softmax(logits_freq, dim=1).cpu()[0]
            p_freq_real = float(probs_freq[0].item())
            p_freq_ai   = float(probs_freq[1].item())

            # B. Spatial V4
            logits_spatial = self.spatial_model(tensor_spatial)
            probs_spatial = torch.softmax(logits_spatial, dim=1).cpu()[0]
            p_spatial_real = float(probs_spatial[0].item())
            p_spatial_ai   = float(probs_spatial[1].item())

            # C. Hybrid V4
            logits_hybrid = self.hybrid_model(tensor_spatial, tensor_fft)
            probs_hybrid = torch.softmax(logits_hybrid, dim=1).cpu()[0]
            p_hybrid_real = float(probs_hybrid[0].item())
            p_hybrid_ai   = float(probs_hybrid[1].item())

            # D. Manipulation V1 (Phase 7 Frequency Detector)
            logits_manip = self.manipulation_model(tensor_manip)
            probs_manip = torch.softmax(logits_manip, dim=1).cpu()[0]
            p_manip_orig = float(probs_manip[0].item())
            p_manip_ai   = float(probs_manip[1].item())

        # -------------------------------------------------------------------
        # 1B. LIGHTWEIGHT GENERIC VISUAL-DOMAIN ANALYSIS
        # -------------------------------------------------------------------
        domain_info = analyze_image_domain(img)

        gen_dict = {
            "model_name": "V4_Frequency_Generation_Detector",
            "probability_real": round(p_freq_real, 4),
            "probability_ai_generated": round(p_freq_ai, 4),
            "label": "AI_GENERATED" if p_freq_ai >= 0.50 else "REAL",
            "confidence": round(max(p_freq_real, p_freq_ai), 4),
        }

        manip_dict = {
            "model_name": "Phase7_Frequency_Manipulation_Detector",
            "probability_original": round(p_manip_orig, 4),
            "probability_ai_manipulated": round(p_manip_ai, 4),
            "label": "AI_MANIPULATED" if p_manip_ai >= 0.50 else "ORIGINAL_REAL",
            "confidence": round(max(p_manip_orig, p_manip_ai), 4),
        }

        auxiliary_evidence = {
            "probability_spatial_ai": round(p_spatial_ai, 4),
            "probability_spatial_real": round(p_spatial_real, 4),
            "probability_freq_ai": round(p_freq_ai, 4),
            "probability_freq_real": round(p_freq_real, 4),
            "probability_hybrid_ai": round(p_hybrid_ai, 4),
            "probability_hybrid_real": round(p_hybrid_real, 4),
            "probability_manipulated": round(p_manip_ai, 4),
            "probability_original": round(p_manip_orig, 4),
            "domain_type": domain_info["domain_type"],
            "domain_risk": domain_info["domain_risk"],
            "domain_risk_score": domain_info["domain_risk_score"],
            "domain_signals": domain_info["domain_signals"],
        }

        # -------------------------------------------------------------------
        # 2. FORENSIC DECISION ENGINE INTEGRATION
        # -------------------------------------------------------------------
        decision = self.engine.decide(
            gen_dict,
            manip_dict,
            strategy=strategy,
            auxiliary_evidence=auxiliary_evidence,
        )
        decision["auxiliary"] = auxiliary_evidence
        decision["domain"] = domain_info
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
