"""
AIDetect Phase 14: Production Forensic Decision Engine (v2)
===========================================================
Deterministic dual-strategy decision engine integrating independent frozen models:
  1. Generation Detector: REAL (0) vs AI-GENERATED (1)
  2. Manipulation Detector: ORIGINAL_REAL (0) vs AI_MANIPULATED (1)

Produces exactly four final forensic states:
  1. REAL_ORIGINAL   - Real photograph with no AI generation or manipulation
  2. AI_GENERATED    - Synthesized directly by a generative AI model
  3. AI_MANIPULATED   - Authentic real photograph altered with AI inpainting/editing
  4. UNCERTAIN       - Insufficient confidence or contradictory evidence

Supported Decision Strategies:
-----------------------------
1. "v1_baseline" (or "baseline"):
   The original Phase 10/11 deterministic rule-based hierarchy.
   Prioritizes CASE 1 (Generation AI) over CASE 3 (Manipulation AI).
   Preserved 100% intact for scientific benchmarking and backward compatibility.

2. "v2_calibrated" (or "calibrated"):
   The Phase 12/13 validated 2D probability fusion strategy (Strategy E).
   Uses frozen multinomial logistic calibration over (P_GEN_AI, P_MANIPULATED).
   Resolves the dual-excitation conflict region, boosting AI_MANIPULATED test recall
   from 12.99% to 96.10% while maintaining 89.74% REAL and 86.12% AI_GENERATED test recall.
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import numpy as np

# ---------------------------------------------------------------------------
# PATHS & CONSTANTS
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_CALIBRATION_PATH = BASE_DIR / "phase12_results" / "strategy_e_calibration.json"

DEFAULT_DECISION_THRESHOLD     = 0.50
DEFAULT_MIN_CONFIDENCE         = 0.55
DEFAULT_CONFLICT_ORIGINAL_TH   = 0.85
DEFAULT_COMPETING_MARGIN       = 0.05

VALID_FINAL_STATES = {
    "REAL_ORIGINAL",
    "AI_GENERATED",
    "AI_MANIPULATED",
    "UNCERTAIN"
}

VALID_STRATEGIES = {
    "baseline",
    "v1_baseline",
    "phase10_baseline",
    "calibrated",
    "v2_calibrated",
    "strategy_e",
}

# Frozen calibration fallback constants (Phase 12 Development Calibration)
FROZEN_CALIBRATION_FEATURE_NAMES = ["p_gen_ai", "p_manip_manipulated"]
FROZEN_CALIBRATION_CLASSES       = ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"]
FROZEN_UNCERTAINTY_THRESHOLD     = 0.50

FROZEN_CALIBRATION_WEIGHTS = [
    [-1.7879907846450807, -4.235018396377564],
    [4.0133381366729735, -5.3892566680908205],
    [-2.2253125667572022, 9.624278831481934],
]
FROZEN_CALIBRATION_BIASES = [
    3.055479621887207,
    0.35649876594543456,
    -3.411933946609497,
]

# ---------------------------------------------------------------------------
# FORENSIC DECISION ENGINE CLASS
# ---------------------------------------------------------------------------

class ForensicDecisionEngine:
    """
    Production decision engine supporting both 'v1_baseline' and 'v2_calibrated' strategies.
    """
    def __init__(
        self,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        conflict_strong_original: float = DEFAULT_CONFLICT_ORIGINAL_TH,
        competing_margin: float = DEFAULT_COMPETING_MARGIN,
        strategy: str = "baseline",
        calibration_path: Optional[Union[str, Path]] = None,
    ):
        self.min_confidence = min_confidence
        self.conflict_strong_original = conflict_strong_original
        self.competing_margin = competing_margin
        
        self.default_strategy = self._normalize_strategy_name(strategy)
        self.calibration_path = Path(calibration_path) if calibration_path else DEFAULT_CALIBRATION_PATH
        
        # Load and verify Strategy E calibration parameters
        self._load_calibration()

    def _normalize_strategy_name(self, strategy: str) -> str:
        s = strategy.lower().strip()
        if s in ("baseline", "v1_baseline", "phase10_baseline"):
            return "v1_baseline"
        elif s in ("calibrated", "v2_calibrated", "strategy_e"):
            return "v2_calibrated"
        else:
            raise ValueError(f"Unknown decision strategy: '{strategy}'. Valid options: {VALID_STRATEGIES}")

    def _load_calibration(self):
        """
        Load frozen calibration parameters for Strategy E.
        Verifies feature order, class names, and tensor dimensions.
        """
        if self.calibration_path.exists():
            with open(self.calibration_path, "r", encoding="utf-8") as f:
                calib = json.load(f)
            
            # Verify feature names and order
            feat_names = calib.get("feature_names", [])
            if feat_names != FROZEN_CALIBRATION_FEATURE_NAMES:
                raise ValueError(
                    f"Invalid calibration feature order: expected {FROZEN_CALIBRATION_FEATURE_NAMES}, got {feat_names}"
                )
            
            # Verify class names
            class_names = calib.get("class_names", [])
            if class_names != FROZEN_CALIBRATION_CLASSES:
                raise ValueError(
                    f"Invalid calibration class order: expected {FROZEN_CALIBRATION_CLASSES}, got {class_names}"
                )
            
            self.calib_weights = np.array(calib["average_weights"], dtype=np.float64)
            self.calib_biases = np.array(calib["average_biases"], dtype=np.float64)
            self.calib_classes = class_names
            self.calib_uncertainty_threshold = float(calib.get("uncertainty_threshold", FROZEN_UNCERTAINTY_THRESHOLD))
            self.calib_metadata = {
                "source": calib.get("source"),
                "created_at": calib.get("created_at"),
                "path": str(self.calibration_path),
            }
        else:
            # Fallback to frozen embedded constants
            self.calib_weights = np.array(FROZEN_CALIBRATION_WEIGHTS, dtype=np.float64)
            self.calib_biases = np.array(FROZEN_CALIBRATION_BIASES, dtype=np.float64)
            self.calib_classes = list(FROZEN_CALIBRATION_CLASSES)
            self.calib_uncertainty_threshold = FROZEN_UNCERTAINTY_THRESHOLD
            self.calib_metadata = {
                "source": "Embedded Frozen Constants (Phase 12)",
                "created_at": "2026-10-02T23:09:05.308430",
                "path": "embedded",
            }

        # Assert shapes
        assert self.calib_weights.shape == (3, 2), f"Invalid weights shape: {self.calib_weights.shape}"
        assert self.calib_biases.shape == (3,), f"Invalid biases shape: {self.calib_biases.shape}"

    def decide(
        self,
        generation_output: Optional[Dict[str, Any]],
        manipulation_output: Optional[Dict[str, Any]],
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate model predictions and return the structured forensic decision.

        Args:
            generation_output: Dict containing:
                - 'probability_real': float in [0, 1]
                - 'probability_ai_generated': float in [0, 1]
            manipulation_output: Dict containing:
                - 'probability_original': float in [0, 1]
                - 'probability_ai_manipulated': float in [0, 1]
            strategy: Optional override ('v1_baseline' or 'v2_calibrated').
                     If None, uses self.default_strategy.

        Returns:
            Dict containing:
                - 'generation': Dict of normalized generation metrics
                - 'manipulation': Dict of normalized manipulation metrics
                - 'final': Dict with 'label', 'confidence', 'reason', 'decision_case', 'strategy'
        """
        active_strategy = self._normalize_strategy_name(strategy) if strategy else self.default_strategy

        # -------------------------------------------------------------------
        # 1. VALIDATION OF INPUTS
        # -------------------------------------------------------------------
        gen_valid, gen_err = self._validate_model_output(
            generation_output, ("probability_real", "probability_ai_generated")
        )
        manip_valid, manip_err = self._validate_model_output(
            manipulation_output, ("probability_original", "probability_ai_manipulated")
        )

        if not gen_valid or not manip_valid:
            err_msg = "; ".join(filter(None, [gen_err, manip_err]))
            return {
                "generation": generation_output or {},
                "manipulation": manipulation_output or {},
                "final": {
                    "label": "UNCERTAIN",
                    "confidence": 0.0,
                    "reason": f"Invalid or missing model output: {err_msg}",
                    "decision_case": "CASE_4_INVALID",
                    "strategy": active_strategy,
                }
            }

        # Normalize generation values
        p_real = float(generation_output["probability_real"])
        p_gen_ai = float(generation_output["probability_ai_generated"])
        gen_label = "AI_GENERATED" if p_gen_ai >= DEFAULT_DECISION_THRESHOLD else "REAL"
        gen_conf = max(p_real, p_gen_ai)

        norm_gen = {
            "label": gen_label,
            "raw_label": 1 if gen_label == "AI_GENERATED" else 0,
            "probability_real": round(p_real, 4),
            "probability_ai_generated": round(p_gen_ai, 4),
            "confidence": round(gen_conf, 4),
            "model_name": generation_output.get("model_name", "V4_Generation_Detector"),
        }

        # Normalize manipulation values
        p_orig = float(manipulation_output["probability_original"])
        p_manip_ai = float(manipulation_output["probability_ai_manipulated"])
        manip_label = "AI_MANIPULATED" if p_manip_ai >= DEFAULT_DECISION_THRESHOLD else "ORIGINAL_REAL"
        manip_conf = max(p_orig, p_manip_ai)

        norm_manip = {
            "label": manip_label,
            "raw_label": 1 if manip_label == "AI_MANIPULATED" else 0,
            "probability_original": round(p_orig, 4),
            "probability_ai_manipulated": round(p_manip_ai, 4),
            "confidence": round(manip_conf, 4),
            "model_name": manipulation_output.get("model_name", "Phase7_Frequency_Manipulation_Detector"),
        }

        # -------------------------------------------------------------------
        # 2. DISPATCH TO STRATEGY
        # -------------------------------------------------------------------
        if active_strategy == "v1_baseline":
            final_dict = self._decide_baseline(norm_gen, norm_manip)
        else:
            final_dict = self._decide_calibrated(norm_gen, norm_manip)

        return {
            "generation": norm_gen,
            "manipulation": norm_manip,
            "final": final_dict,
        }

    # -----------------------------------------------------------------------
    # STRATEGY 1: BASELINE HIERARCHICAL RULE (PHASE 10/11)
    # -----------------------------------------------------------------------
    def _decide_baseline(self, norm_gen: Dict[str, Any], norm_manip: Dict[str, Any]) -> Dict[str, Any]:
        p_real = norm_gen["probability_real"]
        p_gen_ai = norm_gen["probability_ai_generated"]
        gen_label = norm_gen["label"]
        gen_conf = norm_gen["confidence"]

        p_orig = norm_manip["probability_original"]
        p_manip_ai = norm_manip["probability_ai_manipulated"]
        manip_label = norm_manip["label"]
        manip_conf = norm_manip["confidence"]

        # Case 4A: Low confidence
        if gen_conf < self.min_confidence:
            return {
                "label": "UNCERTAIN",
                "confidence": round(gen_conf, 4),
                "reason": (
                    f"Model evidence is insufficient: Generation detector confidence "
                    f"({gen_conf:.2%}) is below threshold ({self.min_confidence:.2%})."
                ),
                "decision_case": "CASE_4_LOW_CONFIDENCE",
                "strategy": "v1_baseline",
            }

        if manip_conf < self.min_confidence:
            return {
                "label": "UNCERTAIN",
                "confidence": round(manip_conf, 4),
                "reason": (
                    f"Model evidence is insufficient: Manipulation detector confidence "
                    f"({manip_conf:.2%}) is below threshold ({self.min_confidence:.2%})."
                ),
                "decision_case": "CASE_4_LOW_CONFIDENCE",
                "strategy": "v1_baseline",
            }

        # Case 4B: Contradictory evidence
        if gen_label == "AI_GENERATED" and p_orig >= self.conflict_strong_original and p_gen_ai < 0.90:
            return {
                "label": "UNCERTAIN",
                "confidence": round(min(gen_conf, manip_conf), 4),
                "reason": (
                    f"Model evidence is conflicting: Generation detector indicates AI-generated "
                    f"content ({p_gen_ai:.2%}), but manipulation detector strongly indicates a "
                    f"pristine original photograph ({p_orig:.2%})."
                ),
                "decision_case": "CASE_4_CONTRADICTORY",
                "strategy": "v1_baseline",
            }

        if gen_label == "AI_GENERATED" and manip_label == "AI_MANIPULATED":
            if abs(p_gen_ai - p_manip_ai) <= self.competing_margin and p_gen_ai >= 0.70 and p_manip_ai >= 0.70:
                return {
                    "label": "UNCERTAIN",
                    "confidence": round(min(gen_conf, manip_conf), 4),
                    "reason": (
                        f"Model evidence is conflicting: Both full-generation ({p_gen_ai:.2%}) and "
                        f"localized-manipulation ({p_manip_ai:.2%}) hypotheses have competing high-confidence "
                        f"evidence within margin ({self.competing_margin:.2%})."
                    ),
                    "decision_case": "CASE_4_CONTRADICTORY",
                    "strategy": "v1_baseline",
                }

        # Case 1: Generation detector identifies AI-generated
        if gen_label == "AI_GENERATED":
            return {
                "label": "AI_GENERATED",
                "confidence": round(p_gen_ai, 4),
                "reason": "Generation detector identifies the image as AI-generated.",
                "decision_case": "CASE_1_AI_GENERATED",
                "strategy": "v1_baseline",
            }

        # Generation = REAL
        if gen_label == "REAL":
            # Case 2: Real photograph + Original (no manipulation)
            if manip_label == "ORIGINAL_REAL":
                final_conf = min(p_real, p_orig)
                return {
                    "label": "REAL_ORIGINAL",
                    "confidence": round(final_conf, 4),
                    "reason": (
                        "Generation detector identifies the image as real and "
                        "manipulation detector identifies no AI manipulation."
                    ),
                    "decision_case": "CASE_2_REAL_ORIGINAL",
                    "strategy": "v1_baseline",
                }

            # Case 3: Real origin + AI manipulation detected
            if manip_label == "AI_MANIPULATED":
                return {
                    "label": "AI_MANIPULATED",
                    "confidence": round(p_manip_ai, 4),
                    "reason": (
                        "Image is classified as real-origin content with evidence "
                        "of AI-based manipulation."
                    ),
                    "decision_case": "CASE_3_AI_MANIPULATED",
                    "strategy": "v1_baseline",
                }

        # Fallback
        return {
            "label": "UNCERTAIN",
            "confidence": 0.0,
            "reason": "Indeterminate state.",
            "decision_case": "CASE_4_FALLBACK",
            "strategy": "v1_baseline",
        }

    # -----------------------------------------------------------------------
    # STRATEGY 2: CALIBRATED 2D FUSION (PHASE 12/13 STRATEGY E)
    # -----------------------------------------------------------------------
    def _decide_calibrated(self, norm_gen: Dict[str, Any], norm_manip: Dict[str, Any]) -> Dict[str, Any]:
        p_gen_ai = norm_gen["probability_ai_generated"]
        p_manip_ai = norm_manip["probability_ai_manipulated"]

        # Feature vector: [P_GEN_AI, P_MANIPULATED]
        x = np.array([p_gen_ai, p_manip_ai], dtype=np.float64)

        # Logits: W @ x + b
        logits = np.dot(self.calib_weights, x) + self.calib_biases

        # Softmax probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Max class selection
        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])
        pred_label = self.calib_classes[max_idx]

        # Conflict safeguard: if calibration chooses AI_GENERATED, but generation detector has
        # weak/borderline AI excitation (< 0.60) while manipulation detector strongly confirms
        # an authentic original camera photograph (>= conflict_strong_original = 0.85):
        # Prevent borderline generation noise from falsely convicting genuine photos.
        p_orig = norm_manip["probability_original"]
        if pred_label == "AI_GENERATED" and p_orig >= self.conflict_strong_original:
            if p_gen_ai < 0.50:
                pred_label = "REAL_ORIGINAL"
                max_prob = p_orig
            elif p_gen_ai < 0.60:
                pred_label = "UNCERTAIN"
                max_prob = max(p_gen_ai, p_orig)

        fusion_probs = {
            cls: round(float(probs[i]), 4) for i, cls in enumerate(self.calib_classes)
        }

        # Check uncertainty threshold
        if pred_label == "UNCERTAIN" or max_prob < self.calib_uncertainty_threshold:
            return {
                "label": "UNCERTAIN",
                "confidence": round(max_prob, 4),
                "fusion_confidence": round(max_prob, 4),
                "fusion_probabilities": fusion_probs,
                "reason": (
                    f"Calibrated forensic evidence is ambiguous or contradictory: generation detector "
                    f"excitation ({p_gen_ai:.2%}) is in conflict with authentic pristine capture ({p_orig:.2%})."
                ) if pred_label == "UNCERTAIN" else (
                    f"Calibrated probability fusion evidence is ambiguous: dominant class confidence "
                    f"({max_prob:.2%}) is below threshold ({self.calib_uncertainty_threshold:.2%})."
                ),
                "decision_case": "V2_CALIBRATED_UNCERTAIN",
                "strategy": "v2_calibrated",
            }

        # Format descriptive reason
        if pred_label == "REAL_ORIGINAL":
            reason = (
                f"Calibrated forensic fusion confirms an unaltered genuine camera photograph "
                f"(confidence: {max_prob:.2%})."
            )
        elif pred_label == "AI_GENERATED":
            reason = (
                f"Calibrated forensic fusion identifies end-to-end AI synthesis artifacts "
                f"(confidence: {max_prob:.2%})."
            )
        else:  # AI_MANIPULATED
            reason = (
                f"Calibrated forensic fusion identifies authentic origin with localized AI manipulation traces "
                f"(confidence: {max_prob:.2%})."
            )

        return {
            "label": pred_label,
            "confidence": round(max_prob, 4),
            "fusion_confidence": round(max_prob, 4),
            "fusion_probabilities": fusion_probs,
            "reason": reason,
            "decision_case": f"V2_CALIBRATED_{pred_label}",
            "strategy": "v2_calibrated",
        }

    # -----------------------------------------------------------------------
    # INPUT VALIDATION
    # -----------------------------------------------------------------------
    @staticmethod
    def _validate_model_output(output: Optional[Dict[str, Any]], expected_keys: tuple) -> tuple:
        if not output or not isinstance(output, dict):
            return False, "Model output is null or not a dictionary"
        for k in expected_keys:
            if k not in output:
                return False, f"Missing required key: {k}"
            val = output[k]
            if not isinstance(val, (int, float)):
                return False, f"Key '{k}' must be numeric, got {type(val)}"
            if math.isnan(val) or math.isinf(val):
                return False, f"Key '{k}' has NaN or Inf value"
            if val < 0.0 or val > 1.0:
                return False, f"Key '{k}' must be in range [0, 1], got {val}"
        
        # Check sum approximately 1.0
        k1, k2 = expected_keys
        s = output[k1] + output[k2]
        if abs(s - 1.0) > 0.02:
            return False, f"Probabilities ({k1}={output[k1]}, {k2}={output[k2]}) do not sum to 1.0 (sum={s:.4f})"
        return True, ""
