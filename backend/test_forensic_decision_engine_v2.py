"""
backend/test_forensic_decision_engine_v2.py
===========================================
Comprehensive Test Suite for Phase 14 Production Forensic Decision Engine (v2)
Verifies:
  A) REAL generation + ORIGINAL manipulation -> REAL_ORIGINAL
  B) AI generation + ORIGINAL manipulation -> AI_GENERATED
  C) REAL generation + MANIPULATED -> AI_MANIPULATED
  D) AI generation + MANIPULATED -> validated Strategy E behavior
  E) Borderline probabilities
  F) Invalid probabilities
  G) NaN / Inf values
  H) Probabilities not summing to 1.0
  I) Missing fields
  J) Calibration file loading
  K) Wrong feature order detection
  L) Deterministic repeated inference
  M) Phase 13 Frozen Test Reproduction (861 / 861 matches required)
  N) Model checkpoint hash verification
"""

import sys
import math
import csv
import json
import hashlib
import unittest
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.forensic_decision_engine import (
    ForensicDecisionEngine,
    VALID_FINAL_STATES,
    DEFAULT_CALIBRATION_PATH,
)

GEN_CKPT_PATH = PROJECT_DIR / "models" / "frequency_resnet50_v4.pth"
MANIP_CKPT_PATH = PROJECT_DIR / "models" / "manipulation_frequency_resnet50_v1.pth"

EXPECTED_GEN_HASH = "2f106288c53ac8728f581f87e49848fc616cb7c30db52a42dec1342cc5d74dbf"
EXPECTED_MANIP_HASH = "8bd929125f379c3abdb71ca2249b2252482122973b4225b80a9ab65c68c86889"

PHASE13_PRED_CSV = BASE_DIR / "phase13_results" / "frozen_test_predictions.csv"


class TestForensicDecisionEngineV2(unittest.TestCase):

    def setUp(self):
        # Baseline engine
        self.engine_baseline = ForensicDecisionEngine(strategy="baseline")
        # Calibrated engine (Production v2)
        self.engine_calib = ForensicDecisionEngine(strategy="calibrated")

    def _assert_valid_state(self, res, expected_label=None):
        self.assertIn("final", res)
        fin = res["final"]
        label = fin.get("label")
        self.assertIn(label, VALID_FINAL_STATES, f"Label '{label}' not in {VALID_FINAL_STATES}")
        self.assertIn("confidence", fin)
        self.assertIsInstance(fin["confidence"], (int, float))
        self.assertGreaterEqual(fin["confidence"], 0.0)
        self.assertLessEqual(fin["confidence"], 1.0)
        self.assertTrue(len(fin.get("reason", "")) > 0)
        if expected_label:
            self.assertEqual(label, expected_label)

    # -----------------------------------------------------------------------
    # TEST A: REAL generation + ORIGINAL manipulation -> REAL_ORIGINAL
    # -----------------------------------------------------------------------
    def test_A_real_generation_original_manipulation(self):
        gen = {"probability_real": 0.95, "probability_ai_generated": 0.05}
        manip = {"probability_original": 0.92, "probability_ai_manipulated": 0.08}
        
        # Test Calibrated Strategy
        res_calib = self.engine_calib.decide(gen, manip)
        self._assert_valid_state(res_calib, "REAL_ORIGINAL")
        self.assertEqual(res_calib["final"]["strategy"], "v2_calibrated")

        # Test Baseline Strategy
        res_base = self.engine_baseline.decide(gen, manip)
        self._assert_valid_state(res_base, "REAL_ORIGINAL")
        self.assertEqual(res_base["final"]["strategy"], "v1_baseline")

    # -----------------------------------------------------------------------
    # TEST B: AI generation + ORIGINAL manipulation -> AI_GENERATED
    # -----------------------------------------------------------------------
    def test_B_ai_generation_original_manipulation(self):
        gen = {"probability_real": 0.02, "probability_ai_generated": 0.98}
        manip = {"probability_original": 0.95, "probability_ai_manipulated": 0.05}
        
        res = self.engine_calib.decide(gen, manip)
        self._assert_valid_state(res, "AI_GENERATED")
        self.assertEqual(res["final"]["strategy"], "v2_calibrated")

    # -----------------------------------------------------------------------
    # TEST C: REAL generation + MANIPULATED -> AI_MANIPULATED
    # -----------------------------------------------------------------------
    def test_C_real_generation_manipulated(self):
        gen = {"probability_real": 0.95, "probability_ai_generated": 0.05}
        manip = {"probability_original": 0.05, "probability_ai_manipulated": 0.95}
        
        res = self.engine_calib.decide(gen, manip)
        self._assert_valid_state(res, "AI_MANIPULATED")
        self.assertEqual(res["final"]["strategy"], "v2_calibrated")

    # -----------------------------------------------------------------------
    # TEST D: AI generation + MANIPULATED (Conflict Region) -> AI_MANIPULATED
    # -----------------------------------------------------------------------
    def test_D_ai_generation_manipulated_conflict(self):
        # Both models flag synthetic traces (typical for localized diffusion edit)
        gen = {"probability_real": 0.15, "probability_ai_generated": 0.85}
        manip = {"probability_original": 0.10, "probability_ai_manipulated": 0.90}
        
        # Calibrated strategy correctly prioritizes AI_MANIPULATED in conflict
        res = self.engine_calib.decide(gen, manip)
        self._assert_valid_state(res, "AI_MANIPULATED")
        self.assertEqual(res["final"]["decision_case"], "V2_CALIBRATED_AI_MANIPULATED")
        self.assertIn("fusion_probabilities", res["final"])

    # -----------------------------------------------------------------------
    # TEST E: Borderline probabilities
    # -----------------------------------------------------------------------
    def test_E_borderline_probabilities(self):
        gen = {"probability_real": 0.51, "probability_ai_generated": 0.49}
        manip = {"probability_original": 0.52, "probability_ai_manipulated": 0.48}
        
        # In baseline, max probability < 0.55 triggers UNCERTAIN
        res_base = self.engine_baseline.decide(gen, manip)
        self._assert_valid_state(res_base, "UNCERTAIN")
        self.assertEqual(res_base["final"]["decision_case"], "CASE_4_LOW_CONFIDENCE")

    # -----------------------------------------------------------------------
    # TEST F: Invalid probabilities (out of [0, 1])
    # -----------------------------------------------------------------------
    def test_F_invalid_probabilities_range(self):
        gen = {"probability_real": 1.5, "probability_ai_generated": -0.5}
        manip = {"probability_original": 0.70, "probability_ai_manipulated": 0.30}
        
        res = self.engine_calib.decide(gen, manip)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_INVALID")

    # -----------------------------------------------------------------------
    # TEST G: NaN / Inf values
    # -----------------------------------------------------------------------
    def test_G_nan_inf_values(self):
        gen = {"probability_real": float("nan"), "probability_ai_generated": 0.5}
        manip = {"probability_original": 0.70, "probability_ai_manipulated": 0.30}
        
        res = self.engine_calib.decide(gen, manip)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_INVALID")

        gen_inf = {"probability_real": float("inf"), "probability_ai_generated": 0.5}
        res_inf = self.engine_calib.decide(gen_inf, manip)
        self.assertEqual(res_inf["final"]["label"], "UNCERTAIN")
        self.assertEqual(res_inf["final"]["decision_case"], "CASE_4_INVALID")

    # -----------------------------------------------------------------------
    # TEST H: Probabilities not summing to 1.0
    # -----------------------------------------------------------------------
    def test_H_probabilities_not_summing_to_one(self):
        gen = {"probability_real": 0.80, "probability_ai_generated": 0.50}  # sum = 1.3
        manip = {"probability_original": 0.70, "probability_ai_manipulated": 0.30}
        
        res = self.engine_calib.decide(gen, manip)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_INVALID")

    # -----------------------------------------------------------------------
    # TEST I: Missing fields
    # -----------------------------------------------------------------------
    def test_I_missing_fields(self):
        gen = {"probability_real": 0.80}  # missing probability_ai_generated
        manip = {"probability_original": 0.70, "probability_ai_manipulated": 0.30}
        
        res = self.engine_calib.decide(gen, manip)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_INVALID")

        res_none = self.engine_calib.decide(None, manip)
        self.assertEqual(res_none["final"]["label"], "UNCERTAIN")
        self.assertEqual(res_none["final"]["decision_case"], "CASE_4_INVALID")

    # -----------------------------------------------------------------------
    # TEST J: Calibration file loading
    # -----------------------------------------------------------------------
    def test_J_calibration_file_loading(self):
        self.assertTrue(DEFAULT_CALIBRATION_PATH.exists(), f"Calibration file missing: {DEFAULT_CALIBRATION_PATH}")
        engine = ForensicDecisionEngine(strategy="calibrated", calibration_path=DEFAULT_CALIBRATION_PATH)
        self.assertEqual(engine.calib_weights.shape, (3, 2))
        self.assertEqual(engine.calib_biases.shape, (3,))
        self.assertEqual(engine.calib_classes, ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"])

    # -----------------------------------------------------------------------
    # TEST K: Wrong feature order detection
    # -----------------------------------------------------------------------
    def test_K_wrong_feature_order_detection(self):
        # Create temporary corrupted calibration file with wrong feature order
        tmp_calib_path = BASE_DIR / "phase12_results" / "tmp_corrupt_calib.json"
        try:
            with open(DEFAULT_CALIBRATION_PATH, "r", encoding="utf-8") as f:
                calib = json.load(f)
            calib["feature_names"] = ["p_manip_manipulated", "p_gen_ai"]  # Reversed order!
            with open(tmp_calib_path, "w", encoding="utf-8") as f:
                json.dump(calib, f)
            
            with self.assertRaises(ValueError) as ctx:
                ForensicDecisionEngine(strategy="calibrated", calibration_path=tmp_calib_path)
            self.assertIn("Invalid calibration feature order", str(ctx.exception))
        finally:
            if tmp_calib_path.exists():
                tmp_calib_path.unlink()

    # -----------------------------------------------------------------------
    # TEST L: Deterministic repeated inference
    # -----------------------------------------------------------------------
    def test_L_deterministic_repeated_inference(self):
        gen = {"probability_real": 0.2345, "probability_ai_generated": 0.7655}
        manip = {"probability_original": 0.3124, "probability_ai_manipulated": 0.6876}
        
        res1 = self.engine_calib.decide(gen, manip)
        res2 = self.engine_calib.decide(gen, manip)
        res3 = self.engine_calib.decide(gen, manip)

        self.assertEqual(res1["final"]["label"], res2["final"]["label"])
        self.assertEqual(res1["final"]["label"], res3["final"]["label"])
        self.assertEqual(res1["final"]["confidence"], res2["final"]["confidence"])
        self.assertEqual(res1["final"]["confidence"], res3["final"]["confidence"])
        self.assertEqual(res1["final"]["fusion_probabilities"], res2["final"]["fusion_probabilities"])

    # -----------------------------------------------------------------------
    # TEST M: Phase 13 Frozen Test Reproduction (861 / 861 Matches Required)
    # -----------------------------------------------------------------------
    def test_M_phase13_frozen_test_reproduction(self):
        self.assertTrue(PHASE13_PRED_CSV.exists(), f"Phase 13 predictions missing: {PHASE13_PRED_CSV}")
        with open(PHASE13_PRED_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        self.assertEqual(len(rows), 861, f"Expected 861 test rows, found {len(rows)}")
        
        matches = 0
        mismatches = []

        for r in rows:
            gen_out = {
                "probability_real": float(r["p_gen_real"]),
                "probability_ai_generated": float(r["p_gen_ai"]),
            }
            manip_out = {
                "probability_original": float(r["p_manip_original"]),
                "probability_ai_manipulated": float(r["p_manip_manipulated"]),
            }

            res = self.engine_calib.decide(gen_out, manip_out, strategy="calibrated")
            new_state = res["final"]["label"]
            phase13_expected = r["strategy_e_state"]

            if new_state == phase13_expected:
                matches += 1
            else:
                mismatches.append({
                    "image_path": r["image_path"],
                    "expected": phase13_expected,
                    "got": new_state,
                    "p_gen_ai": r["p_gen_ai"],
                    "p_manip": r["p_manip_manipulated"],
                })

        self.assertEqual(
            matches, 861,
            f"Phase 13 reproduction failed! {matches} / 861 matches. Mismatches: {mismatches[:5]}"
        )

    # -----------------------------------------------------------------------
    # TEST N: Model Checkpoint Hash Verification
    # -----------------------------------------------------------------------
    def test_N_model_checkpoint_hashes(self):
        self.assertTrue(GEN_CKPT_PATH.exists(), f"Generation checkpoint missing: {GEN_CKPT_PATH}")
        self.assertTrue(MANIP_CKPT_PATH.exists(), f"Manipulation checkpoint missing: {MANIP_CKPT_PATH}")

        def get_sha256(path):
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            return h.hexdigest()

        gen_sha = get_sha256(GEN_CKPT_PATH)
        manip_sha = get_sha256(MANIP_CKPT_PATH)

        self.assertEqual(gen_sha, EXPECTED_GEN_HASH, f"Generation hash mismatch: {gen_sha}")
        self.assertEqual(manip_sha, EXPECTED_MANIP_HASH, f"Manipulation hash mismatch: {manip_sha}")


if __name__ == "__main__":
    unittest.main()
