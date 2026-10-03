"""
backend/phase12_decision_strategies.py
======================================
Candidate Decision Strategies for Phase 12 Development Study.
Investigates semantic decision logic on development set ONLY.

Strictly preserves:
- Frozen models
- Untouched Phase 11 test set
- 4 output states: REAL_ORIGINAL, AI_GENERATED, AI_MANIPULATED, UNCERTAIN
"""

from typing import Dict, Any, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# CANDIDATE A: CURRENT RULE (Exact Baseline)
# ---------------------------------------------------------------------------

def strategy_a_current_rule(
    p_real: float,
    p_gen_ai: float,
    p_orig: float,
    p_manip_ai: float,
    min_confidence: float = 0.55,
    conflict_strong_original: float = 0.85,
    competing_margin: float = 0.05,
) -> Tuple[str, float, str]:
    """
    Candidate A: Replicates exact baseline logic of ForensicDecisionEngine.
    """
    gen_label = "AI_GENERATED" if p_gen_ai >= 0.50 else "REAL"
    gen_conf = max(p_real, p_gen_ai)
    manip_label = "AI_MANIPULATED" if p_manip_ai >= 0.50 else "ORIGINAL_REAL"
    manip_conf = max(p_orig, p_manip_ai)

    # 1. Low confidence check
    if gen_conf < min_confidence or manip_conf < min_confidence:
        return "UNCERTAIN", round(min(gen_conf, manip_conf), 4), "CASE_4_LOW_CONFIDENCE"

    # 2. Contradictory evidence
    if gen_label == "AI_GENERATED" and p_orig >= conflict_strong_original and p_gen_ai < 0.90:
        return "UNCERTAIN", round(min(gen_conf, manip_conf), 4), "CASE_4_CONTRADICTORY"

    if gen_label == "AI_GENERATED" and manip_label == "AI_MANIPULATED":
        if abs(p_gen_ai - p_manip_ai) <= competing_margin and p_gen_ai >= 0.70 and p_manip_ai >= 0.70:
            return "UNCERTAIN", round(min(gen_conf, manip_conf), 4), "CASE_4_CONTRADICTORY"

    # 3. Deterministic primary cases
    if gen_label == "AI_GENERATED":
        return "AI_GENERATED", round(p_gen_ai, 4), "CASE_1_AI_GENERATED"

    if gen_label == "REAL":
        if manip_label == "ORIGINAL_REAL":
            return "REAL_ORIGINAL", round(min(p_real, p_orig), 4), "CASE_2_REAL_ORIGINAL"
        if manip_label == "AI_MANIPULATED":
            return "AI_MANIPULATED", round(p_manip_ai, 4), "CASE_3_AI_MANIPULATED"

    return "UNCERTAIN", 0.0, "CASE_4_FALLBACK"


# ---------------------------------------------------------------------------
# CANDIDATE B: GENERATION-FIRST (Strict Hierarchy with Variable Thresholds)
# ---------------------------------------------------------------------------

def strategy_b_generation_first(
    p_real: float,
    p_gen_ai: float,
    p_orig: float,
    p_manip_ai: float,
    th_gen: float = 0.55,
    th_manip: float = 0.55,
) -> Tuple[str, float, str]:
    """
    Candidate B: Generation detector dominates whenever p_gen_ai >= th_gen.
    Manipulation detector is queried strictly if generation detector verifies REAL.
    """
    if p_gen_ai >= th_gen:
        return "AI_GENERATED", round(p_gen_ai, 4), "GEN_FIRST_AI_GENERATED"
    elif p_real >= th_gen:
        if p_manip_ai >= th_manip:
            return "AI_MANIPULATED", round(p_manip_ai, 4), "GEN_FIRST_AI_MANIPULATED"
        elif p_orig >= th_manip:
            return "REAL_ORIGINAL", round(min(p_real, p_orig), 4), "GEN_FIRST_REAL_ORIGINAL"
        else:
            return "UNCERTAIN", round(max(p_orig, p_manip_ai), 4), "GEN_FIRST_UNCERTAIN_MANIP"
    else:
        return "UNCERTAIN", round(max(p_real, p_gen_ai), 4), "GEN_FIRST_UNCERTAIN_GEN"


# ---------------------------------------------------------------------------
# CANDIDATE C: MANIPULATION-FIRST IN CONFLICT REGION
# ---------------------------------------------------------------------------

def strategy_c_manipulation_priority_in_conflict(
    p_real: float,
    p_gen_ai: float,
    p_orig: float,
    p_manip_ai: float,
    th_gen: float = 0.55,
    th_manip: float = 0.55,
) -> Tuple[str, float, str]:
    """
    Candidate C: In the conflict region (both models flag AI >= threshold),
    give priority to AI_MANIPULATED, because localized diffusion inpainting
    injects high-frequency synthesis traces that trigger the generation model.
    """
    # Low confidence region
    gen_conf = max(p_real, p_gen_ai)
    manip_conf = max(p_orig, p_manip_ai)
    if gen_conf < th_gen or manip_conf < th_manip:
        return "UNCERTAIN", round(min(gen_conf, manip_conf), 4), "MANIP_PRIORITY_LOW_CONF"

    # Region A: Both agree on genuine original
    if p_real >= th_gen and p_orig >= th_manip:
        return "REAL_ORIGINAL", round(min(p_real, p_orig), 4), "MANIP_PRIORITY_REAL_ORIGINAL"

    # Region C: Generation says real, Manipulation says manipulated
    if p_real >= th_gen and p_manip_ai >= th_manip:
        return "AI_MANIPULATED", round(p_manip_ai, 4), "MANIP_PRIORITY_AI_MANIPULATED_ISOLATED"

    # Region D (Conflict Region): Both flag AI
    if p_gen_ai >= th_gen and p_manip_ai >= th_manip:
        return "AI_MANIPULATED", round(p_manip_ai, 4), "MANIP_PRIORITY_AI_MANIPULATED_CONFLICT"

    # Region B: Generation says AI, Manipulation says pristine original
    if p_gen_ai >= th_gen and p_orig >= th_manip:
        return "AI_GENERATED", round(p_gen_ai, 4), "MANIP_PRIORITY_AI_GENERATED"

    return "UNCERTAIN", 0.0, "MANIP_PRIORITY_FALLBACK"


# ---------------------------------------------------------------------------
# CANDIDATE D: PROBABILITY-MARGIN / DOMINANCE BASED DECISION
# ---------------------------------------------------------------------------

def strategy_d_margin_based(
    p_real: float,
    p_gen_ai: float,
    p_orig: float,
    p_manip_ai: float,
    th_gen: float = 0.55,
    th_manip: float = 0.55,
    margin_threshold: float = 0.10,
) -> Tuple[str, float, str]:
    """
    Candidate D: Uses probability margins in the conflict region.
    If both flag AI, evaluate delta = p_manip_ai - p_gen_ai.
    If manipulation evidence is substantially stronger (delta > margin), AI_MANIPULATED.
    If generation evidence is substantially stronger (delta < -margin), AI_GENERATED.
    If within margin, flag UNCERTAIN (ambiguous synthetic attribution).
    """
    gen_conf = max(p_real, p_gen_ai)
    manip_conf = max(p_orig, p_manip_ai)

    if gen_conf < th_gen or manip_conf < th_manip:
        return "UNCERTAIN", round(min(gen_conf, manip_conf), 4), "MARGIN_LOW_CONF"

    # Pristine real
    if p_real >= th_gen and p_orig >= th_manip:
        return "REAL_ORIGINAL", round(min(p_real, p_orig), 4), "MARGIN_REAL_ORIGINAL"

    # Isolated generation
    if p_gen_ai >= th_gen and p_orig >= th_manip:
        return "AI_GENERATED", round(p_gen_ai, 4), "MARGIN_AI_GENERATED_PURE"

    # Isolated manipulation
    if p_real >= th_gen and p_manip_ai >= th_manip:
        return "AI_MANIPULATED", round(p_manip_ai, 4), "MARGIN_AI_MANIPULATED_PURE"

    # Conflict region: both detect AI
    if p_gen_ai >= th_gen and p_manip_ai >= th_manip:
        delta = p_manip_ai - p_gen_ai
        if delta > margin_threshold:
            return "AI_MANIPULATED", round(p_manip_ai, 4), "MARGIN_AI_MANIP_DOMINANT"
        elif delta < -margin_threshold:
            return "AI_GENERATED", round(p_gen_ai, 4), "MARGIN_AI_GEN_DOMINANT"
        else:
            return "UNCERTAIN", round(min(p_gen_ai, p_manip_ai), 4), "MARGIN_COMPETING_AMBIGUITY"

    return "UNCERTAIN", 0.0, "MARGIN_FALLBACK"


# ---------------------------------------------------------------------------
# CANDIDATE E: LOGISTIC REGRESSION META-CLASSIFIER (2D PROBABILITY FUSION)
# ---------------------------------------------------------------------------

class LogisticMetaClassifierStrategy:
    """
    Candidate E: Simple multinomial logistic regression using ONLY:
    features = [p_gen_ai, p_manip_ai].
    Trained strictly on development data splits (via Stratified K-Fold CV).
    Produces calibrated out-of-fold 3-class probabilities via softmax.
    Zero deep neural network, zero pixel features, zero Phase 11 test data.
    Pure PyTorch implementation (Linear + CrossEntropy) to avoid Windows AppLocker issues.
    """
    def __init__(self, uncertainty_threshold: float = 0.50):
        self.uncertainty_threshold = uncertainty_threshold
        self.class_names = ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"]
        self.linear = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        import torch.nn as nn
        
        # Compute balanced class weights
        classes, counts = np.unique(y, return_counts=True)
        weights = len(y) / (len(classes) * counts.astype(float))
        weight_tensor = torch.tensor(weights, dtype=torch.float32)

        x_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)

        # Multinomial Logistic Regression: 2 inputs -> 3 logits
        self.linear = nn.Linear(2, 3)
        torch.manual_seed(42)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        optimizer = torch.optim.LBFGS(self.linear.parameters(), lr=0.1, max_iter=200)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        def closure():
            optimizer.zero_grad()
            out = self.linear(x_t)
            loss = criterion(out, y_t)
            loss.backward()
            return loss

        optimizer.step(closure)

    def predict_one(self, p_gen_ai: float, p_manip_ai: float) -> Tuple[str, float, str]:
        import torch
        if self.linear is None:
            raise RuntimeError("LogisticMetaClassifier must be fit before prediction.")
        with torch.no_grad():
            x_t = torch.tensor([[p_gen_ai, p_manip_ai]], dtype=torch.float32)
            logits = self.linear(x_t)
            probs = torch.softmax(logits, dim=1).numpy()[0]
        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])
        pred_label = self.class_names[max_idx]

        if max_prob < self.uncertainty_threshold:
            return "UNCERTAIN", round(max_prob, 4), "LOGISTIC_LOW_PROB"
        return pred_label, round(max_prob, 4), f"LOGISTIC_{pred_label}"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        if self.linear is None:
            raise RuntimeError("LogisticMetaClassifier must be fit before prediction.")
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32)
            logits = self.linear(x_t)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs

    def load_calibration_file(self, json_path: str):
        import json
        import torch
        import torch.nn as nn
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        w = np.array(data["average_weights"], dtype=np.float32)
        b = np.array(data["average_biases"], dtype=np.float32)
        self.linear = nn.Linear(2, 3)
        with torch.no_grad():
            self.linear.weight.copy_(torch.from_numpy(w))
            self.linear.bias.copy_(torch.from_numpy(b))
        self.uncertainty_threshold = data.get("uncertainty_threshold", self.uncertainty_threshold)
        self.calibration_metadata = {
            "source": data.get("source"),
            "created_at": data.get("created_at"),
            "average_weights": w.tolist(),
            "average_biases": b.tolist(),
        }
