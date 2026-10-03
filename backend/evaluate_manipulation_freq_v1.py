"""
AIDetect Phase 7: Test Evaluation & Forensic Analysis Pipeline (Frequency Model)
=================================================================================
Evaluates the best frequency checkpoint (models/manipulation_frequency_resnet50_v1.pth)
on the held-out manipulation_v1/test/ split:
  1. Overall Test Metrics (Accuracy, Precision, Recall, F1, FPR, FNR)
  2. Category-Wise Metrics across 6 manipulation types + ORIGINAL_REAL
  3. Pair-Aware Consistency Analysis (Both original & edit correctly identified)
  4. Generates:
     - manipulation_v1/results/frequency_test_metrics.json
     - manipulation_v1/results/frequency_test_predictions.csv
     - manipulation_v1/results/frequency_confusion_matrix.csv
     - manipulation_v1/results/frequency_category_metrics.json
     - manipulation_v1/results/frequency_pair_metrics.json
"""

import os
import sys
import csv
import json
import time
import datetime
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, Counter

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from manipulation_freq_v1_models import ManipulationFrequencyResNet50V1
from manipulation_freq_v1_dataset import (
    ManipulationFreqV1Dataset, collate_manipulation_freq,
    METADATA_CSV, MANIP_DIR,
)

BASE_DIR        = Path(__file__).resolve().parent
PROJECT_DIR     = BASE_DIR.parent
RESULTS_DIR     = MANIP_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH      = PROJECT_DIR / "models" / "manipulation_frequency_resnet50_v1.pth"
PAIRS_PATH      = MANIP_DIR / "metadata" / "pair_relationships.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("AIDETECT PHASE 7: FREQUENCY-DOMAIN MANIPULATION DETECTOR TEST EVALUATION")
print("=" * 70)
print(f"Device        : {DEVICE}")
print(f"Checkpoint    : {MODEL_PATH}")
print(f"Test Split    : manipulation_v1/test/")
print(f"Timestamp     : {datetime.datetime.now().isoformat()}")
print()
sys.stdout.flush()

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Trained checkpoint not found: {MODEL_PATH}")

# ---------------------------------------------------------------------------
# LOAD CHECKPOINT
# ---------------------------------------------------------------------------

ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
model = ManipulationFrequencyResNet50V1(dropout_p=0.3)

if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
    state_dict = ckpt["model_state_dict"]
    best_epoch = ckpt.get("epoch", "?")
    best_val_f1 = ckpt.get("best_val_f1", "?")
    print(f"Checkpoint Metadata: Epoch={best_epoch}, Best Val F1={best_val_f1}")
else:
    state_dict = ckpt
    print("Loaded raw state dict.")

cleaned_state = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
model.load_state_dict(cleaned_state, strict=True)
model.to(DEVICE)
model.eval()

# ---------------------------------------------------------------------------
# LOAD TEST DATASET
# ---------------------------------------------------------------------------

test_ds = ManipulationFreqV1Dataset("test", augment=False)
test_loader = DataLoader(
    test_ds, batch_size=16, shuffle=False,
    collate_fn=collate_manipulation_freq, drop_last=False
)
print(f"Loaded Test Samples: {len(test_ds)} images")
sys.stdout.flush()

# ---------------------------------------------------------------------------
# RUN INFERENCE
# ---------------------------------------------------------------------------

all_targets = []
all_preds   = []
all_probs_real = []
all_probs_manip = []
all_metas   = []

t0 = time.time()
with torch.no_grad():
    for x, y, metas in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu()
        preds = logits.argmax(1).cpu().tolist()
        
        all_targets.extend(y.tolist())
        all_preds.extend(preds)
        all_probs_real.extend(probs[:, 0].tolist())
        all_probs_manip.extend(probs[:, 1].tolist())
        all_metas.extend(metas)

elapsed = time.time() - t0
print(f"Inference complete: {len(all_targets)} images evaluated in {elapsed:.2f}s")
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 1. OVERALL TEST METRICS
# ---------------------------------------------------------------------------

acc  = accuracy_score(all_targets, all_preds)
prec = precision_score(all_targets, all_preds, average="macro", zero_division=0)
rec  = recall_score(all_targets, all_preds, average="macro", zero_division=0)
f1   = f1_score(all_targets, all_preds, average="macro", zero_division=0)
cm   = confusion_matrix(all_targets, all_preds, labels=[0, 1])

tn, fp, fn, tp = cm.ravel()
real_recall  = tn / (tn + fp + 1e-8)
manip_recall = tp / (tp + fn + 1e-8)

fpr = fp / (fp + tn + 1e-8)
fnr = fn / (fn + tp + 1e-8)

overall_metrics = {
    "total_test_images": len(all_targets),
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1": round(float(f1), 4),
    "real_recall": round(float(real_recall), 4),
    "ai_manipulated_recall": round(float(manip_recall), 4),
    "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    "FPR": round(float(fpr), 4),
    "FNR": round(float(fnr), 4),
    "confusion_matrix": cm.tolist(),
}

print("-" * 60)
print("1. OVERALL TEST PERFORMANCE (FREQUENCY DETECTOR)")
print("-" * 60)
print(f"Accuracy                 : {acc * 100:.2f}%")
print(f"Precision (Macro)        : {prec:.4f}")
print(f"Recall (Macro)           : {rec:.4f}")
print(f"F1 Score (Macro)         : {f1:.4f}")
print(f"ORIGINAL_REAL Recall     : {real_recall * 100:.2f}%  (TN={tn}, FP={fp})")
print(f"AI_MANIPULATED Recall    : {manip_recall * 100:.2f}%  (TP={tp}, FN={fn})")
print(f"False Positive Rate (FPR): {fpr * 100:.2f}%")
print(f"False Negative Rate (FNR): {fnr * 100:.2f}%")
print(f"Confusion Matrix         :\n{cm}")
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 2. CATEGORY-WISE FORENSIC EVALUATION
# ---------------------------------------------------------------------------

print("-" * 60)
print("2. CATEGORY-WISE EVALUATION")
print("-" * 60)

categories = [
    "object_insertion",
    "object_replacement",
    "inpainting",
    "face_modification",
    "object_removal",
    "background_replacement"
]

category_metrics = {}

# Evaluate each manipulation category
for cat in categories:
    cat_indices = [i for i, m in enumerate(all_metas) if m["manipulation_type"] == cat and m["label"] == 1]
    if not cat_indices:
        continue
    c_targets = [all_targets[i] for i in cat_indices]
    c_preds   = [all_preds[i] for i in cat_indices]
    
    n_images = len(cat_indices)
    correct  = sum(1 for p in c_preds if p == 1)
    incorrect = n_images - correct
    c_acc    = correct / n_images
    c_rec    = correct / n_images  # In single-class subcategory, accuracy == recall
    
    category_metrics[cat] = {
        "num_manipulated_images": n_images,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": round(c_acc, 4),
        "recall": round(c_rec, 4),
    }
    print(f"  {cat:<25}: n={n_images:>2} | Correct={correct:>2} | Acc={c_acc*100:>5.1f}% | Recall={c_rec:>6.4f}")

# Also evaluate ORIGINAL_REAL
real_indices = [i for i, m in enumerate(all_metas) if m["label"] == 0]
n_real_test = len(real_indices)
real_correct = sum(1 for i in real_indices if all_preds[i] == 0)
real_incorrect = n_real_test - real_correct
real_acc = real_correct / n_real_test

category_metrics["ORIGINAL_REAL"] = {
    "num_original_images": n_real_test,
    "correct": real_correct,
    "incorrect": real_incorrect,
    "accuracy": round(real_acc, 4),
    "recall": round(real_acc, 4),
}
print(f"  {'ORIGINAL_REAL':<25}: n={n_real_test:>2} | Correct={real_correct:>2} | Acc={real_acc*100:>5.1f}% | Recall={real_acc:>6.4f}")
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 3. PAIR-AWARE CONSISTENCY ANALYSIS
# ---------------------------------------------------------------------------

print("-" * 60)
print("3. PAIR-AWARE FORENSIC CONSISTENCY ANALYSIS")
print("-" * 60)

with open(PAIRS_PATH, encoding="utf-8") as f:
    pair_map = json.load(f)

# Build quick lookup from image_id to prediction
pred_lookup = {m["image_id"]: all_preds[i] for i, m in enumerate(all_metas)}

# Filter pairs belonging to the test split
test_orig_ids = [m["image_id"] for m in all_metas if m["label"] == 0]
test_pairs_evaluated = 0
consistent_pairs = 0
pair_results_detail = []

for orig_img_id in test_orig_ids:
    if orig_img_id not in pred_lookup:
        continue
    orig_pred = pred_lookup[orig_img_id]
    manip_ids = pair_map.get(orig_img_id, [])
    
    for manip_id in manip_ids:
        if manip_id in pred_lookup:
            test_pairs_evaluated += 1
            manip_pred = pred_lookup[manip_id]
            
            # Pair Success: Original predicted as REAL (0) AND Manipulated predicted as MANIPULATED (1)
            is_success = (orig_pred == 0 and manip_pred == 1)
            if is_success:
                consistent_pairs += 1
                
            pair_results_detail.append({
                "original_img_id": orig_img_id,
                "manip_img_id": manip_id,
                "original_prediction": "ORIGINAL_REAL" if orig_pred == 0 else "AI_MANIPULATED",
                "manipulated_prediction": "AI_MANIPULATED" if manip_pred == 1 else "ORIGINAL_REAL",
                "pair_success": is_success,
            })

pair_success_rate = (consistent_pairs / test_pairs_evaluated) if test_pairs_evaluated > 0 else 0.0

pair_metrics = {
    "total_test_pairs": test_pairs_evaluated,
    "successful_pairs": consistent_pairs,
    "pair_success_rate": round(pair_success_rate, 4),
    "criterion": "Original predicted as ORIGINAL_REAL (0) AND Manipulated predicted as AI_MANIPULATED (1)",
    "detailed_pairs": pair_results_detail,
}

print(f"Total Test Pairs Evaluated: {test_pairs_evaluated}")
print(f"Pairs Meeting Criterion   : {consistent_pairs} / {test_pairs_evaluated}")
print(f"Pair-Level Success Rate   : {pair_success_rate * 100:.2f}%")
print()
sys.stdout.flush()

# ---------------------------------------------------------------------------
# 4. SAVE OUTPUTS
# ---------------------------------------------------------------------------

print("Saving results to manipulation_v1/results/...")

# 1. frequency_test_metrics.json
with open(RESULTS_DIR / "frequency_test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(overall_metrics, f, indent=2)
print("  - frequency_test_metrics.json")

# 2. frequency_category_metrics.json
with open(RESULTS_DIR / "frequency_category_metrics.json", "w", encoding="utf-8") as f:
    json.dump(category_metrics, f, indent=2)
print("  - frequency_category_metrics.json")

# 3. frequency_pair_metrics.json
with open(RESULTS_DIR / "frequency_pair_metrics.json", "w", encoding="utf-8") as f:
    json.dump(pair_metrics, f, indent=2)
print("  - frequency_pair_metrics.json")

# 4. frequency_confusion_matrix.csv
with open(RESULTS_DIR / "frequency_confusion_matrix.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["", "Pred_ORIGINAL_REAL", "Pred_AI_MANIPULATED"])
    writer.writerow(["True_ORIGINAL_REAL", tn, fp])
    writer.writerow(["True_AI_MANIPULATED", fn, tp])
print("  - frequency_confusion_matrix.csv")

# 5. frequency_test_predictions.csv
with open(RESULTS_DIR / "frequency_test_predictions.csv", "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "image_id", "original_id", "manipulation_type", "ground_truth",
        "true_label", "predicted_label", "predicted_class",
        "prob_real", "prob_manipulated", "is_correct"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(all_metas)):
        m = all_metas[i]
        writer.writerow({
            "image_id": m["image_id"],
            "original_id": m["original_id"],
            "manipulation_type": m["manipulation_type"],
            "ground_truth": m["ground_truth"],
            "true_label": all_targets[i],
            "predicted_label": all_preds[i],
            "predicted_class": "ORIGINAL_REAL" if all_preds[i] == 0 else "AI_MANIPULATED",
            "prob_real": round(all_probs_real[i], 4),
            "prob_manipulated": round(all_probs_manip[i], 4),
            "is_correct": int(all_preds[i] == all_targets[i]),
        })
print("  - frequency_test_predictions.csv")

print()
print("=" * 70)
print("FREQUENCY EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 70)
sys.stdout.flush()

