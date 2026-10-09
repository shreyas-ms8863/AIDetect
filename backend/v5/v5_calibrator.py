"""
AIDetect V5-D Probability Calibrator
====================================
Phase 7: Production Calibrator for V5-D Gated Residual Detector.

Applies post-hoc probability calibration (Platt Scaling or Temperature Scaling)
derived from the 9,000-image V5 validation split (calibration-fit partition).

INVARIANTS:
1. Zero modifications to V5-D model weights or architecture.
2. Evaluates calibration parameters loaded from backend/models/v5/v5_d_calibration.json.
3. Provides fallback pass-through if calibration has not yet been fitted.
4. Preserves ranking and monotonicity (A > 0, T > 0).
"""

import os
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

# Root directory resolution
V5_DIR = Path(__file__).resolve().parent
BACKEND_DIR = V5_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
DEFAULT_CALIBRATION_JSON = BACKEND_DIR / "models" / "v5" / "v5_d_calibration.json"


def _sigmoid(x: float) -> float:
    """Numerically stable scalar sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


class V5DCalibrator:
    """
    Post-hoc probability calibrator for V5-D Gated Residual model outputs.
    Transforms raw logits [z_real, z_ai] or raw probabilities into calibrated probabilities.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        if config_path is None:
            # Check standard locations
            candidates = [
                DEFAULT_CALIBRATION_JSON,
                PROJECT_ROOT / "models" / "v5" / "v5_d_calibration.json",
                Path("models/v5/v5_d_calibration.json"),
            ]
            self.config_path = None
            for cand in candidates:
                if cand.is_file():
                    self.config_path = cand
                    break
        else:
            self.config_path = Path(config_path)

        self.is_loaded = False
        self.method = "raw"
        self.version = "Raw_Softmax_Uncalibrated"
        self.A = 1.0
        self.B = 0.0
        self.T = 1.0
        self.metrics = {}

        if self.config_path and self.config_path.is_file():
            self._load_config(self.config_path)
        else:
            # Informative fallback
            print(f"[INFO] V5-D calibrator config not found at {DEFAULT_CALIBRATION_JSON}. Running in uncalibrated pass-through mode.")

    def _load_config(self, path: Path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.version = data.get("version", "Platt_Scaling_v1")
            self.method = data.get("method", "platt").lower()
            self.A = float(data.get("A", 1.0))
            self.B = float(data.get("B", 0.0))
            
            temp_baseline = data.get("temperature_baseline", {})
            self.T = float(temp_baseline.get("T", 1.0))
            
            self.metrics = data.get("metrics", {})
            self.is_loaded = True
            print(f"[OK] V5-D Calibrator loaded: method={self.method.upper()} ({self.version})")
        except Exception as e:
            print(f"[WARNING] Failed to load V5-D calibration config: {e}. Fallback to raw.")
            self.is_loaded = False

    def calibrate(
        self,
        raw_real_prob: float,
        raw_ai_prob: float,
        logits: Optional[Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Calibrate raw V5-D outputs.

        Args:
          raw_real_prob: Raw probability for REAL (0 to 100 or 0.0 to 1.0).
          raw_ai_prob: Raw probability for AI (0 to 100 or 0.0 to 1.0).
          logits: Optional tuple of raw logits (z_real, z_ai).

        Returns:
          Dict containing:
            - calibrated_ai_probability (percentage 0-100)
            - calibrated_real_probability (percentage 0-100)
            - confidence (percentage 0-100)
            - calibration_method (str)
            - is_calibrated (bool)
        """
        # Normalize probabilities to [0.0, 1.0] for math if given as percentages
        p_real = raw_real_prob / 100.0 if raw_real_prob > 1.0 else raw_real_prob
        p_ai = raw_ai_prob / 100.0 if raw_ai_prob > 1.0 else raw_ai_prob

        if not self.is_loaded:
            # Fallback pass-through
            cal_ai = round(p_ai * 100.0, 2)
            cal_real = round(p_real * 100.0, 2)
            conf = round(max(cal_ai, cal_real), 2)
            return {
                "calibrated_ai_probability": cal_ai,
                "calibrated_real_probability": cal_real,
                "confidence": conf,
                "calibration_method": "Uncalibrated (Fit Pending)",
                "is_calibrated": False
            }

        # Compute logit difference s = z_ai - z_real
        if logits is not None and len(logits) == 2:
            z_real, z_ai = float(logits[0]), float(logits[1])
            s = z_ai - z_real
        else:
            # Invert probabilities into logit difference: log(p_ai / (1 - p_ai))
            p_ai_clipped = max(min(p_ai, 1.0 - 1e-7), 1e-7)
            s = math.log(p_ai_clipped / (1.0 - p_ai_clipped))

        # Apply transformation
        if self.method == "platt":
            # P(AI) = sigmoid(A * s + B), with A > 0
            cal_prob_ai = _sigmoid(self.A * s + self.B)
        elif self.method == "temperature":
            # P(AI) = sigmoid(s / T), with T > 0
            cal_prob_ai = _sigmoid(s / max(self.T, 1e-4))
        else:
            cal_prob_ai = p_ai

        cal_prob_real = 1.0 - cal_prob_ai

        cal_ai_pct = round(cal_prob_ai * 100.0, 2)
        cal_real_pct = round(cal_prob_real * 100.0, 2)
        conf_pct = round(max(cal_ai_pct, cal_real_pct), 2)

        return {
            "calibrated_ai_probability": cal_ai_pct,
            "calibrated_real_probability": cal_real_pct,
            "confidence": conf_pct,
            "calibration_method": self.version,
            "is_calibrated": True
        }

