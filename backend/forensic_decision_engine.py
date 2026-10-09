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

# ---------------------------------------------------------------------------
# RELIABILITY / ABSTENTION THRESHOLDS
# ---------------------------------------------------------------------------

# Strong AI consensus: both Spatial and Frequency independently detect AI evidence
STRONG_AI_SPATIAL_THRESHOLD  = 0.75
STRONG_AI_FREQ_THRESHOLD     = 0.75

# Genuine photograph consensus: all generation streams agree on REAL + manipulation confirms original
STRONG_REAL_SPATIAL_MAX      = 0.40
STRONG_REAL_FREQ_MAX         = 0.60
STRONG_REAL_HYBRID_MAX       = 0.20
STRONG_REAL_ORIG_MIN         = 0.85

# Disagreement thresholds for reliability classification
#   HIGH  → UNCERTAIN (no consensus formed, models strongly contradict)
#   MODERATE → UNCERTAIN unless Strategy E is unambiguously decisive
HIGH_DISAGREEMENT_THRESHOLD     = 0.45   # max_gen - min_gen >= 0.45
MODERATE_DISAGREEMENT_THRESHOLD = 0.30   # max_gen - min_gen >= 0.30

# Spatial-Frequency raw disagreement cap: if S and F are themselves very far apart
# this alone can flag a CONFLICT image
SF_HIGH_DISAGREEMENT_THRESHOLD  = 0.50   # abs(P_SPATIAL_AI - P_FREQ_AI) >= 0.50

# Strategy E must be strongly decisive (away from 50/50 ambiguity)
# to issue a final verdict under moderate disagreement
STRATEGY_E_DECISIVE_THRESHOLD   = 0.65   # calib_prob must be >= this to accept in moderate zone

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
        auxiliary_evidence: Optional[Dict[str, float]] = None,
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
            auxiliary_evidence: Optional Dict of multi-stream model probabilities:
                - 'probability_spatial_ai'
                - 'probability_spatial_real'
                - 'probability_freq_ai'
                - 'probability_freq_real'
                - 'probability_hybrid_ai'
                - 'probability_hybrid_real'
                - 'probability_manipulated'
                - 'probability_original'

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
            final_dict = self._decide_calibrated(norm_gen, norm_manip, auxiliary_evidence=auxiliary_evidence)

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
    # STRATEGY 2: CALIBRATED 2D FUSION WITH RELIABILITY-AWARE ABSTENTION
    #   Phase 12/13 Strategy E + multi-stream consensus + disagreement checks
    #
    # Decision Hierarchy (STEPS 1-9):
    #   STEP 1 – Validate probabilities (done in decide() before dispatch)
    #   STEP 2 – Strong AI Consensus (Spatial>=0.75 AND Freq>=0.75) → AI_GENERATED
    #   STEP 3 – Strong Real Consensus (S<0.40 AND F<0.60 AND H<0.20 AND Orig>=0.85) → REAL_ORIGINAL
    #   STEP 4 – High Disagreement (gen_disagree>=0.45) → UNCERTAIN
    #   STEP 5 – Frozen Strategy E on [P_FREQ_AI, P_MANIPULATED]
    #   STEP 6 – Moderate Disagreement (gen_disagree>=0.30) + Strategy E not decisive → UNCERTAIN
    #   STEP 7 – Strategy E result (below-threshold confidence) → UNCERTAIN
    #   STEP 8 – Return Strategy E result
    # -----------------------------------------------------------------------
    def _decide_calibrated(
        self,
        norm_gen: Dict[str, Any],
        norm_manip: Dict[str, Any],
        auxiliary_evidence: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        p_gen_ai = norm_gen["probability_ai_generated"]
        p_manip_ai = norm_manip["probability_ai_manipulated"]

        # ---------------------------------------------------------------
        # Determine if we have full multi-stream evidence
        # ---------------------------------------------------------------
        has_auxiliary = (
            auxiliary_evidence is not None
            and "probability_spatial_ai" in auxiliary_evidence
            and "probability_freq_ai" in auxiliary_evidence
            and "probability_hybrid_ai" in auxiliary_evidence
        )

        if has_auxiliary:
            p_freq_ai = float(auxiliary_evidence["probability_freq_ai"])
            p_spatial_ai = float(auxiliary_evidence["probability_spatial_ai"])
            p_hybrid_ai = float(auxiliary_evidence["probability_hybrid_ai"])
            p_manip = float(auxiliary_evidence.get("probability_manipulated", p_manip_ai))
            p_orig = float(auxiliary_evidence.get("probability_original", norm_manip["probability_original"]))
            domain_type = auxiliary_evidence.get("domain_type", "NATURAL_PHOTO")
            domain_risk = auxiliary_evidence.get("domain_risk", "LOW")
            domain_risk_score = float(auxiliary_evidence.get("domain_risk_score", 0.0))
            domain_signals = auxiliary_evidence.get("domain_signals", {})
        else:
            # No multi-stream data — reliability/consensus/domain steps do not apply
            # Used for frozen backward-compatibility tests (e.g. Test M Phase 13)
            p_freq_ai = float(p_gen_ai)
            p_manip = float(p_manip_ai)
            p_spatial_ai = None
            p_hybrid_ai = None
            p_orig = norm_manip["probability_original"]
            domain_type = "NATURAL_PHOTO"
            domain_risk = "LOW"
            domain_risk_score = 0.0
            domain_signals = {}

        # ---------------------------------------------------------------
        # STEP 5 (computed first so results are available for all steps):
        #   Frozen Strategy E calibration on [P_FREQ_AI, P_MANIPULATED]
        # ---------------------------------------------------------------
        x = np.array([p_freq_ai, p_manip], dtype=np.float64)
        logits = np.dot(self.calib_weights, x) + self.calib_biases
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        max_idx = int(np.argmax(probs))
        calib_prob = float(probs[max_idx])
        calib_label = self.calib_classes[max_idx]

        fusion_probs = {
            cls: round(float(probs[i]), 4) for i, cls in enumerate(self.calib_classes)
        }

        # ---------------------------------------------------------------
        # Reliability metrics (disagreement & domain indicators)
        # ---------------------------------------------------------------
        if p_spatial_ai is not None and p_hybrid_ai is not None:
            gen_values = [p_spatial_ai, p_freq_ai, p_hybrid_ai]
            max_gen = max(gen_values)
            min_gen = min(gen_values)
            generation_disagreement = max_gen - min_gen
            spatial_frequency_disagreement = abs(p_spatial_ai - p_freq_ai)
            hybrid_disagreement = max(
                abs(p_hybrid_ai - p_spatial_ai),
                abs(p_hybrid_ai - p_freq_ai)
            )

            # Classify reliability level taking into account both disagreement and domain risk
            if domain_risk == "HIGH" or generation_disagreement >= HIGH_DISAGREEMENT_THRESHOLD:
                reliability_state = "LOW"
            elif domain_risk == "MEDIUM" or generation_disagreement >= MODERATE_DISAGREEMENT_THRESHOLD:
                reliability_state = "MODERATE"
            else:
                reliability_state = "RELIABLE"
        else:
            # No auxiliary evidence — reliability layer is not applicable
            # Strategy E is the sole authority; treat as reliable
            generation_disagreement = 0.0
            spatial_frequency_disagreement = 0.0
            hybrid_disagreement = 0.0
            reliability_state = "RELIABLE"

        reliability_metrics = {
            "generation_disagreement": round(generation_disagreement, 4),
            "spatial_frequency_disagreement": round(spatial_frequency_disagreement, 4),
            "hybrid_disagreement": round(hybrid_disagreement, 4),
            "domain_risk_score": round(domain_risk_score, 4),
            "domain_risk": domain_risk,
            "domain_type": domain_type,
            "reliability_state": reliability_state,
        }

        # Helper: build a return dict
        def make_result(
            label: str,
            confidence: float,
            reason: str,
            decision_source: str,
            consensus_state: str,
            decision_case: Optional[str] = None,
        ) -> Dict[str, Any]:
            if decision_case is None:
                decision_case = (
                    f"V2_CALIBRATED_{label}"
                    if label != "UNCERTAIN"
                    else "V2_CALIBRATED_UNCERTAIN"
                )
            return {
                "label": label,
                "confidence": round(confidence, 4),
                "fusion_confidence": round(calib_prob, 4),
                "fusion_probabilities": fusion_probs,
                "decision_source": decision_source,
                "consensus_state": consensus_state,
                "reliability_state": reliability_state,
                "reliability_metrics": reliability_metrics,
                "domain_type": domain_type,
                "domain_risk": domain_risk,
                "domain_risk_score": domain_risk_score,
                "reason": reason,
                "decision_case": decision_case,
                "strategy": "v2_calibrated",
            }

        # ---------------------------------------------------------------
        # HIERARCHICAL DECISION LOGIC
        # Applies ONLY when full multi-stream evidence is present.
        # Without auxiliary evidence, Strategy E is the sole authority.
        # ---------------------------------------------------------------
        if has_auxiliary:
            # -----------------------------------------------------------
            # HIERARCHY STEP 4: HIGH DOMAIN-RISK OVERRIDE
            #   If domain_risk == "HIGH", the image exhibits document/scan/
            #   structured-canvas traits outside the validated natural-photo
            #   training distribution.
            #   Do NOT permit automatic AI_GENERATED convictions or automatic
            #   REAL_ORIGINAL convictions based on model agreement alone.
            #   → UNCERTAIN (domain_risk_abstention).
            # -----------------------------------------------------------
            if domain_risk == "HIGH":
                return make_result(
                    "UNCERTAIN",
                    domain_risk_score,
                    (
                        "Generation signals were detected, but the image appears to belong to a domain "
                        "outside the detector's validated operating distribution. The system cannot make "
                        "a reliable AI-generation determination from the available forensic evidence."
                    ),
                    "domain_risk_abstention",
                    "OUT_OF_DOMAIN",
                )

            # -----------------------------------------------------------
            # HIERARCHY STEP 5: LOW DOMAIN RISK (VALIDATED NATURAL OPERATING DOMAIN)
            #   1. Apply Strong AI Consensus
            #   2. Apply Strong Real Consensus
            #   3. Apply High Disagreement Abstention
            #   4. Fall through to Strategy E for normal/borderline cases
            # -----------------------------------------------------------
            if domain_risk == "LOW":
                # Strong AI Consensus (Spatial >= 0.75 AND Frequency >= 0.75)
                strong_ai = (
                    p_spatial_ai >= STRONG_AI_SPATIAL_THRESHOLD
                    and p_freq_ai >= STRONG_AI_FREQ_THRESHOLD
                )

                if strong_ai:
                    if calib_label == "AI_GENERATED":
                        final_conf = calib_prob
                        src = "strategy_e_calibration"
                    else:
                        final_conf = max(p_spatial_ai, p_freq_ai)
                        src = "generation_consensus_override"

                    return make_result(
                        "AI_GENERATED",
                        final_conf,
                        "Spatial and frequency generation analysis independently identify strong synthetic-image evidence.",
                        src,
                        "STRONG_AI",
                    )

                # Strong Real Consensus (Spatial < 0.40 AND Frequency < 0.60 AND Hybrid < 0.20 AND Original >= 0.85)
                strong_real = (
                    p_spatial_ai < STRONG_REAL_SPATIAL_MAX
                    and p_freq_ai < STRONG_REAL_FREQ_MAX
                    and p_hybrid_ai < STRONG_REAL_HYBRID_MAX
                    and p_orig >= STRONG_REAL_ORIG_MIN
                )

                if strong_real:
                    return make_result(
                        "REAL_ORIGINAL",
                        p_orig,
                        "Spatial, frequency, hybrid, and localized-manipulation evidence are consistent with an authentic photograph.",
                        "genuine_photograph_consensus",
                        "STRONG_REAL",
                    )

                # High Disagreement check (models strongly contradict each other)
                if generation_disagreement >= HIGH_DISAGREEMENT_THRESHOLD:
                    return make_result(
                        "UNCERTAIN",
                        generation_disagreement,
                        (
                            "Generation signals are inconsistent across forensic models. "
                            "The image may belong to a domain outside the detector's validated "
                            "operating distribution."
                        ),
                        "high_disagreement_abstention",
                        "CONFLICT",
                    )

            # -----------------------------------------------------------
            # HIERARCHY STEP 6: MEDIUM DOMAIN RISK PATH
            #   Under moderate domain shift:
            #   Do not automatically abstain, but do not force a verdict
            #   if evidence is conflicting or not decisive.
            # -----------------------------------------------------------
            if domain_risk == "MEDIUM":
                if generation_disagreement >= MODERATE_DISAGREEMENT_THRESHOLD or calib_prob < STRATEGY_E_DECISIVE_THRESHOLD:
                    return make_result(
                        "UNCERTAIN",
                        calib_prob,
                        (
                            "Forensic signals are inconsistent or the evidence is in an ambiguous "
                            "region of the detector's operating range under moderate domain risk. "
                            "The system cannot make a reliable AI/real determination from this image alone."
                        ),
                        "moderate_disagreement_abstention",
                        "CONFLICT",
                    )



        # ---------------------------------------------------------------
        # STEP 7: Strategy E below uncertainty threshold → UNCERTAIN
        # ---------------------------------------------------------------
        if calib_prob < self.calib_uncertainty_threshold:
            return make_result(
                "UNCERTAIN",
                calib_prob,
                (
                    f"Calibrated probability fusion evidence is ambiguous: dominant class confidence "
                    f"({calib_prob:.2%}) is below threshold ({self.calib_uncertainty_threshold:.2%})."
                ),
                "strategy_e_calibration",
                "CONFLICT",
            )

        # ---------------------------------------------------------------
        # STEP 8: Return Strategy E result
        # ---------------------------------------------------------------
        if calib_label == "REAL_ORIGINAL":
            reason = (
                f"Calibrated forensic fusion confirms an unaltered genuine camera photograph "
                f"(confidence: {calib_prob:.2%})."
            )
        elif calib_label == "AI_GENERATED":
            reason = (
                f"Calibrated forensic fusion identifies end-to-end AI synthesis artifacts "
                f"(confidence: {calib_prob:.2%})."
            )
        else:  # AI_MANIPULATED
            reason = (
                f"Calibrated forensic fusion identifies authentic origin with localized AI manipulation traces "
                f"(confidence: {calib_prob:.2%})."
            )

        return make_result(
            calib_label,
            calib_prob,
            reason,
            "strategy_e_calibration",
            "CONFLICT",
        )


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
