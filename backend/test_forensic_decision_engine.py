"""
Unit Tests for Phase 10 Forensic Decision Engine
================================================
Verifies all 8 required deterministic combinations and edge cases:
  1. Generation AI + Manipulation Original
  2. Generation AI + Manipulation Manipulated
  3. Generation REAL + Manipulation Original
  4. Generation REAL + Manipulation Manipulated
  5. Low-confidence Generation
  6. Low-confidence Manipulation
  7. Conflicting Evidence
  8. Invalid / Missing Model Output

Ensures strict return state membership in:
  {REAL_ORIGINAL, AI_GENERATED, AI_MANIPULATED, UNCERTAIN}
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forensic_decision_engine import ForensicDecisionEngine, VALID_FINAL_STATES

class TestForensicDecisionEngine(unittest.TestCase):

    def setUp(self):
        self.engine = ForensicDecisionEngine(
            min_confidence=0.55,
            conflict_strong_original=0.85,
            competing_margin=0.05,
        )

    def _assert_valid_state(self, result):
        final = result.get("final", {})
        label = final.get("label")
        self.assertIn(
            label,
            VALID_FINAL_STATES,
            f"Returned state '{label}' is not in valid states: {VALID_FINAL_STATES}"
        )
        self.assertIn("confidence", final)
        self.assertIn("reason", final)
        self.assertIsInstance(final["confidence"], (int, float))
        self.assertGreaterEqual(final["confidence"], 0.0)
        self.assertLessEqual(final["confidence"], 1.0)
        self.assertTrue(len(final["reason"]) > 0)

    # -----------------------------------------------------------------------
    # TEST 1: Generation AI + Manipulation Original
    # -----------------------------------------------------------------------
    def test_01_generation_ai_manipulation_original(self):
        gen = {"probability_real": 0.05, "probability_ai_generated": 0.95}
        manip = {"probability_original": 0.70, "probability_ai_manipulated": 0.30}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "AI_GENERATED")
        self.assertEqual(res["final"]["decision_case"], "CASE_1_AI_GENERATED")
        self.assertAlmostEqual(res["final"]["confidence"], 0.95, places=2)

    # -----------------------------------------------------------------------
    # TEST 2: Generation AI + Manipulation Manipulated
    # -----------------------------------------------------------------------
    def test_02_generation_ai_manipulation_manipulated(self):
        # Clear generation dominance (> competing margin)
        gen = {"probability_real": 0.02, "probability_ai_generated": 0.98}
        manip = {"probability_original": 0.20, "probability_ai_manipulated": 0.80}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "AI_GENERATED")
        self.assertEqual(res["final"]["decision_case"], "CASE_1_AI_GENERATED")
        self.assertAlmostEqual(res["final"]["confidence"], 0.98, places=2)

    # -----------------------------------------------------------------------
    # TEST 3: Generation REAL + Manipulation Original
    # -----------------------------------------------------------------------
    def test_03_generation_real_manipulation_original(self):
        gen = {"probability_real": 0.92, "probability_ai_generated": 0.08}
        manip = {"probability_original": 0.88, "probability_ai_manipulated": 0.12}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "REAL_ORIGINAL")
        self.assertEqual(res["final"]["decision_case"], "CASE_2_REAL_ORIGINAL")
        self.assertAlmostEqual(res["final"]["confidence"], 0.88, places=2)

    # -----------------------------------------------------------------------
    # TEST 4: Generation REAL + Manipulation Manipulated
    # -----------------------------------------------------------------------
    def test_04_generation_real_manipulation_manipulated(self):
        gen = {"probability_real": 0.89, "probability_ai_generated": 0.11}
        manip = {"probability_original": 0.08, "probability_ai_manipulated": 0.92}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "AI_MANIPULATED")
        self.assertEqual(res["final"]["decision_case"], "CASE_3_AI_MANIPULATED")
        self.assertAlmostEqual(res["final"]["confidence"], 0.92, places=2)

    # -----------------------------------------------------------------------
    # TEST 5: Low-confidence Generation
    # -----------------------------------------------------------------------
    def test_05_low_confidence_generation(self):
        # Generation is 0.52 / 0.48 (below 0.55 min_confidence)
        gen = {"probability_real": 0.48, "probability_ai_generated": 0.52}
        manip = {"probability_original": 0.10, "probability_ai_manipulated": 0.90}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_LOW_CONFIDENCE")
        self.assertIn("Generation detector confidence", res["final"]["reason"])

    # -----------------------------------------------------------------------
    # TEST 6: Low-confidence Manipulation
    # -----------------------------------------------------------------------
    def test_06_low_confidence_manipulation(self):
        # Manipulation is 0.53 / 0.47 (below 0.55 min_confidence)
        gen = {"probability_real": 0.90, "probability_ai_generated": 0.10}
        manip = {"probability_original": 0.47, "probability_ai_manipulated": 0.53}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_LOW_CONFIDENCE")
        self.assertIn("Manipulation detector confidence", res["final"]["reason"])

    # -----------------------------------------------------------------------
    # TEST 7: Conflicting Evidence
    # -----------------------------------------------------------------------
    def test_07a_conflicting_evidence_strong_original_vs_moderate_gen_ai(self):
        # Generation predicts AI (0.65), but Manipulation asserts pristine camera photo with 0.92
        gen = {"probability_real": 0.35, "probability_ai_generated": 0.65}
        manip = {"probability_original": 0.92, "probability_ai_manipulated": 0.08}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_CONTRADICTORY")
        self.assertIn("conflicting", res["final"]["reason"].lower())

    def test_07b_conflicting_evidence_competing_synthetic_hypotheses(self):
        # Both models flag AI with identical high confidence within competing margin (0.80 vs 0.81)
        gen = {"probability_real": 0.20, "probability_ai_generated": 0.80}
        manip = {"probability_original": 0.19, "probability_ai_manipulated": 0.81}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_CONTRADICTORY")
        self.assertIn("competing", res["final"]["reason"].lower())

    # -----------------------------------------------------------------------
    # TEST 8: Invalid / Missing Model Output
    # -----------------------------------------------------------------------
    def test_08a_missing_model_output(self):
        res = self.engine.decide(None, {"probability_original": 0.8, "probability_ai_manipulated": 0.2})
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_INVALID")

    def test_08b_non_summing_probabilities(self):
        gen = {"probability_real": 0.70, "probability_ai_generated": 0.70}  # Sum = 1.40
        manip = {"probability_original": 0.80, "probability_ai_manipulated": 0.20}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_INVALID")

    def test_08c_out_of_bounds_probabilities(self):
        gen = {"probability_real": -0.10, "probability_ai_generated": 1.10}
        manip = {"probability_original": 0.80, "probability_ai_manipulated": 0.20}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_INVALID")

    def test_08d_nan_probabilities(self):
        gen = {"probability_real": float("nan"), "probability_ai_generated": 0.5}
        manip = {"probability_original": 0.80, "probability_ai_manipulated": 0.20}
        res = self.engine.decide(gen, manip)
        self._assert_valid_state(res)
        self.assertEqual(res["final"]["label"], "UNCERTAIN")
        self.assertEqual(res["final"]["decision_case"], "CASE_4_INVALID")

if __name__ == "__main__":
    unittest.main()
