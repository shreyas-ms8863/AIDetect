"""
AIDetect Phase 11: Real-World Integrated Forensic Evaluation
============================================================
Evaluates the complete end-to-end frozen forensic pipeline:
  1. Frozen V4 Generation Detector: models/frequency_resnet50_v4.pth
  2. Frozen Phase 7 Manipulation Detector: models/manipulation_frequency_resnet50_v1.pth
  3. Authoritative Forensic Decision Engine: backend/forensic_decision_engine.py

Primary Quantitative Labeled Test Sets (861 images total):
  - Group 1: dataset_v4/test/real/ (296 images)       -> Ground Truth: REAL_ORIGINAL
  - Group 2: dataset_v4/test/ai/ (317 images)         -> Ground Truth: AI_GENERATED
  - Group 3: manipulation_v1/test/original/ (94 images)-> Ground Truth: REAL_ORIGINAL
  - Group 4: manipulation_v1/test/manipulated/ (154)   -> Ground Truth: AI_MANIPULATED

Unlabeled / Qualitative Data (evaluated separately):
  - backend/external_test/ (86 images)
  - backend/real_world_test/ (19 images)

Outputs written strictly to: backend/integrated_results/
"""

import os
import sys
import csv
import json
import time
import random
import hashlib
import datetime
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from forensic_inference import ForensicInferencePipeline
from forensic_decision_engine import ForensicDecisionEngine, VALID_FINAL_STATES

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

BASE_DIR        = Path(__file__).resolve().parent
PROJECT_DIR     = BASE_DIR.parent
RESULTS_DIR     = BASE_DIR / "integrated_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR       = PROJECT_DIR / "models"
GEN_CKPT_PATH   = MODEL_DIR / "frequency_resnet50_v4.pth"
MANIP_CKPT_PATH = MODEL_DIR / "manipulation_frequency_resnet50_v1.pth"

DATASET_V4_TEST = PROJECT_DIR / "dataset_v4" / "test"
MANIP_V1_TEST   = PROJECT_DIR / "manipulation_v1" / "test"
EXT_TEST_DIR    = BASE_DIR / "external_test"
REAL_WORLD_DIR  = BASE_DIR / "real_world_test"
PAIR_JSON_PATH  = PROJECT_DIR / "manipulation_v1" / "metadata" / "pair_relationships.json"

EXPECTED_GEN_HASH   = "2f106288c53ac8728f581f87e49848fc616cb7c30db52a42dec1342cc5d74dbf"
EXPECTED_MANIP_HASH = "8bd929125f379c3abdb71ca2249b2252482122973b4225b80a9ab65c68c86889"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# ---------------------------------------------------------------------------
# DATASET COLLECTION
# ---------------------------------------------------------------------------

def collect_primary_evaluation_samples() -> List[Dict[str, Any]]:
    samples = []
    
    # Group 1: dataset_v4/test/real/ -> REAL_ORIGINAL
    g1_files = sorted(list((DATASET_V4_TEST / "real").rglob("*.*")))
    for p in g1_files:
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            subcat = p.parent.name  # modern, historical, genimage
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "dataset_v4/test/real",
                "source_category": subcat,
                "ground_truth": "REAL_ORIGINAL",
                "image_id": p.stem,
            })
            
    # Group 2: dataset_v4/test/ai/ -> AI_GENERATED
    g2_files = sorted(list((DATASET_V4_TEST / "ai").rglob("*.*")))
    for p in g2_files:
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            subcat = p.parent.name
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "dataset_v4/test/ai",
                "source_category": subcat,
                "ground_truth": "AI_GENERATED",
                "image_id": p.stem,
            })
            
    # Group 3: manipulation_v1/test/original/ -> REAL_ORIGINAL
    g3_files = sorted(list((MANIP_V1_TEST / "original").rglob("*.png")))
    for p in g3_files:
        if p.is_file():
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "manipulation_v1/test/original",
                "source_category": "unaltered_source",
                "ground_truth": "REAL_ORIGINAL",
                "image_id": p.stem.replace("orig_", ""),
            })

    # Group 4: manipulation_v1/test/manipulated/ -> AI_MANIPULATED
    g4_files = sorted(list((MANIP_V1_TEST / "manipulated").rglob("*.png")))
    for p in g4_files:
        if p.is_file():
            cat = p.parent.name  # inpainting, object_insertion, etc.
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": "manipulation_v1/test/manipulated",
                "source_category": cat,
                "ground_truth": "AI_MANIPULATED",
                "image_id": p.stem.replace("manip_", ""),
            })

    return samples

def collect_unlabeled_samples(folder: Path, dataset_name: str) -> List[Dict[str, Any]]:
    samples = []
    if not folder.exists():
        return samples
    for p in sorted(list(folder.rglob("*.*"))):
        if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            sub = p.parent.name
            samples.append({
                "filepath": p,
                "rel_path": str(p.relative_to(PROJECT_DIR)).replace("\\", "/"),
                "source_dataset": dataset_name,
                "source_category": sub,
                "image_id": p.name,
            })
    return samples

# ---------------------------------------------------------------------------
# MAIN EVALUATION SCRIPT
# ---------------------------------------------------------------------------

def run_integrated_evaluation():
    print("=" * 75)
    print("AIDETECT PHASE 11: REAL-WORLD INTEGRATED EVALUATION")
    print("=" * 75)
    t_start = time.time()
    timestamp = datetime.datetime.now().isoformat()
    print(f"Timestamp   : {timestamp}")
    print(f"Device      : {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU Model   : {torch.cuda.get_device_name(0)}")
    print(f"Results Dir : {RESULTS_DIR}")
    print()

    # 1. Checkpoint Verification (Before)
    print("-" * 65)
    print("1. MODEL CHECKPOINT VERIFICATION (PRE-EVALUATION)")
    print("-" * 65)
    
    assert GEN_CKPT_PATH.exists(), f"Generation checkpoint missing: {GEN_CKPT_PATH}"
    assert MANIP_CKPT_PATH.exists(), f"Manipulation checkpoint missing: {MANIP_CKPT_PATH}"
    
    gen_hash_pre = compute_file_sha256(GEN_CKPT_PATH)
    manip_hash_pre = compute_file_sha256(MANIP_CKPT_PATH)
    
    print(f"Generation Model   : {GEN_CKPT_PATH.name} ({GEN_CKPT_PATH.stat().st_size:,} bytes)")
    print(f"  SHA-256          : {gen_hash_pre}")
    print(f"  Expected Hash    : {EXPECTED_GEN_HASH}")
    gen_match_pre = (gen_hash_pre == EXPECTED_GEN_HASH)
    print(f"  Match Status     : {'MATCH' if gen_match_pre else 'MISMATCH'}")
    assert gen_match_pre, "Generation checkpoint hash mismatch!"

    print(f"Manipulation Model : {MANIP_CKPT_PATH.name} ({MANIP_CKPT_PATH.stat().st_size:,} bytes)")
    print(f"  SHA-256          : {manip_hash_pre}")
    print(f"  Expected Hash    : {EXPECTED_MANIP_HASH}")
    manip_match_pre = (manip_hash_pre == EXPECTED_MANIP_HASH)
    print(f"  Match Status     : {'MATCH' if manip_match_pre else 'MISMATCH'}")
    assert manip_match_pre, "Manipulation checkpoint hash mismatch!"
    print()

    # 2. Dataset Inspection & Leakage Audit
    print("-" * 65)
    print("2. DATASET INSPECTION & DATA LEAKAGE AUDIT")
    print("-" * 65)
    
    primary_samples = collect_primary_evaluation_samples()
    print(f"Primary Labeled Test Samples Total : {len(primary_samples)}")
    
    # Class breakdown
    class_counts = Counter(s["ground_truth"] for s in primary_samples)
    print(f"  - REAL_ORIGINAL : {class_counts['REAL_ORIGINAL']}")
    print(f"  - AI_GENERATED  : {class_counts['AI_GENERATED']}")
    print(f"  - AI_MANIPULATED: {class_counts['AI_MANIPULATED']}")
    
    # Verify exact counts match expectations: 296 + 317 + 94 + 154 = 861
    assert len(primary_samples) == 861, f"Expected 861 primary test images, found {len(primary_samples)}"
    assert class_counts["REAL_ORIGINAL"] == 390
    assert class_counts["AI_GENERATED"] == 317
    assert class_counts["AI_MANIPULATED"] == 154
    print("  Counts strictly match expected authoritative test splits.")
    
    # Verify zero duplicate paths
    unique_paths = set(s["filepath"] for s in primary_samples)
    assert len(unique_paths) == len(primary_samples), "Duplicate paths found in test samples!"
    print("  Zero duplicate file paths confirmed.")

    # Inspect Unlabeled / Casual Test Datasets
    ext_samples = collect_unlabeled_samples(EXT_TEST_DIR, "backend/external_test")
    real_samples = collect_unlabeled_samples(REAL_WORLD_DIR, "backend/real_world_test")
    print(f"Unlabeled Evaluation Samples:")
    print(f"  - backend/external_test/   : {len(ext_samples)} images (no authoritative provenance/ground-truth)")
    print(f"  - backend/real_world_test/ : {len(real_samples)} images (no authoritative provenance/ground-truth)")
    print("  These sets will be reported separately as Qualitative/Unlabeled evaluation.")
    print()

    # 3. Initialize Frozen Pipeline
    print("-" * 65)
    print("3. INITIALIZING FROZEN INFERENCE PIPELINE")
    print("-" * 65)
    pipeline = ForensicInferencePipeline(
        generation_ckpt=GEN_CKPT_PATH,
        manipulation_ckpt=MANIP_CKPT_PATH,
        generation_model_type="frequency",
        device=DEVICE,
        min_confidence=0.55,
    )
    print("Pipeline loaded with frozen checkpoints.")
    print()

    # 4. Primary Labeled Evaluation Inference
    print("-" * 65)
    print(f"4. RUNNING INTEGRATED INFERENCE ON {len(primary_samples)} LABELED TEST IMAGES")
    print("-" * 65)
    
    evaluated_records = []
    t_eval_start = time.time()
    
    for idx, s in enumerate(primary_samples):
        res = pipeline.predict(s["filepath"])
        
        gen = res["generation"]
        manip = res["manipulation"]
        final = res["final"]
        
        rec = {
            "image_path": s["rel_path"],
            "source_dataset": s["source_dataset"],
            "source_category": s["source_category"],
            "ground_truth": s["ground_truth"],
            "image_id": s["image_id"],
            
            "generation_real_probability": gen["probability_real"],
            "generation_ai_probability": gen["probability_ai_generated"],
            "generation_prediction": gen["label"],
            "generation_confidence": gen["confidence"],
            
            "manipulation_original_probability": manip["probability_original"],
            "manipulation_manipulated_probability": manip["probability_ai_manipulated"],
            "manipulation_prediction": manip["label"],
            "manipulation_confidence": manip["confidence"],
            
            "final_state": final["label"],
            "final_confidence": final["confidence"],
            "decision_case": final["decision_case"],
            "decision_reason": final["reason"],
            "is_correct": int(final["label"] == s["ground_truth"]),
        }
        evaluated_records.append(rec)
        
        if (idx + 1) % 200 == 0 or (idx + 1) == len(primary_samples):
            print(f"  Processed {idx + 1:>3} / {len(primary_samples)} images ({time.time() - t_eval_start:.1f}s)...")
            sys.stdout.flush()

    eval_duration = time.time() - t_eval_start
    print(f"Evaluation finished in {eval_duration:.2f}s ({len(primary_samples)/eval_duration:.1f} img/s).")
    print()

    # 5. Quantitative 3-Class Decision Metrics
    print("-" * 65)
    print("5. QUANTITATIVE 3-CLASS FORENSIC DECISION METRICS")
    print("-" * 65)
    
    classes_3 = ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED"]
    final_states_4 = ["REAL_ORIGINAL", "AI_GENERATED", "AI_MANIPULATED", "UNCERTAIN"]
    
    # 3x4 Confusion matrix
    # Rows: ground truth, Columns: predicted state
    cm_3x4 = {gt: {pred: 0 for pred in final_states_4} for gt in classes_3}
    for r in evaluated_records:
        cm_3x4[r["ground_truth"]][r["final_state"]] += 1
        
    total_eval = len(evaluated_records)
    total_correct = sum(1 for r in evaluated_records if r["final_state"] == r["ground_truth"])
    overall_accuracy = total_correct / total_eval
    
    total_uncertain = sum(1 for r in evaluated_records if r["final_state"] == "UNCERTAIN")
    uncertain_rate = total_uncertain / total_eval
    
    # Per-class metrics
    per_class_metrics = {}
    for c in classes_3:
        n_gt = sum(cm_3x4[c].values())
        tp = cm_3x4[c][c]
        fp = sum(cm_3x4[other_gt][c] for other_gt in classes_3 if other_gt != c)
        fn = n_gt - tp  # All cases where ground truth was c but predicted state was not c (includes UNCERTAIN)
        n_unc = cm_3x4[c]["UNCERTAIN"]
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / n_gt if n_gt > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        unc_rate_c = n_unc / n_gt if n_gt > 0 else 0.0
        
        per_class_metrics[c] = {
            "n_samples": n_gt,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "uncertain_count": n_unc,
            "uncertain_rate": round(unc_rate_c, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
        }
        
    macro_prec = float(np.mean([per_class_metrics[c]["precision"] for c in classes_3]))
    macro_rec  = float(np.mean([per_class_metrics[c]["recall"] for c in classes_3]))
    macro_f1   = float(np.mean([per_class_metrics[c]["f1"] for c in classes_3]))

    print(f"Overall 3-Class Accuracy : {overall_accuracy * 100:.2f}% ({total_correct} / {total_eval})")
    print(f"Macro Precision          : {macro_prec:.4f}")
    print(f"Macro Recall             : {macro_rec:.4f}")
    print(f"Macro F1 Score           : {macro_f1:.4f}")
    print(f"Overall UNCERTAIN Rate   : {uncertain_rate * 100:.2f}% ({total_uncertain} / {total_eval})")
    print()
    print("Per-Class Forensic Breakdown:")
    for c in classes_3:
        m = per_class_metrics[c]
        print(f"  {c:<15} (n={m['n_samples']:>3}): Recall={m['recall']*100:>5.2f}% | Precision={m['precision']*100:>5.2f}% | F1={m['f1']:.4f} | UNCERTAIN={m['uncertain_rate']*100:.2f}% ({m['uncertain_count']} imgs)")
    print()

    # Print 3x4 Confusion Matrix
    print("Confusion Matrix (Rows: Ground Truth | Columns: Predicted Final State):")
    print(f"{'Ground Truth':<16} | {'REAL_ORIGINAL':<13} | {'AI_GENERATED':<12} | {'AI_MANIPULATED':<14} | {'UNCERTAIN':<9} | {'Total':>5}")
    print("-" * 85)
    for c in classes_3:
        row = cm_3x4[c]
        print(f"{c:<16} | {row['REAL_ORIGINAL']:>13} | {row['AI_GENERATED']:>12} | {row['AI_MANIPULATED']:>14} | {row['UNCERTAIN']:>9} | {sum(row.values()):>5}")
    print("-" * 85)
    print()

    # 6. Independent Branch Evaluations
    print("-" * 65)
    print("6. INDEPENDENT COMPONENT BRANCH PERFORMANCE")
    print("-" * 65)
    
    # Generation Branch: Binary task (0=REAL, 1=AI)
    # Labeled where ground truth is authentically either genuine camera origin or AI generated.
    # Group 1 (REAL, 296) + Group 2 (AI, 317) + Group 3 (REAL, 94) = 707 canonical images
    gen_eval_records = [r for r in evaluated_records if r["source_dataset"] in [
        "dataset_v4/test/real", "dataset_v4/test/ai", "manipulation_v1/test/original"
    ]]
    gen_targets = [1 if r["ground_truth"] == "AI_GENERATED" else 0 for r in gen_eval_records]
    gen_preds   = [1 if r["generation_prediction"] == "AI_GENERATED" else 0 for r in gen_eval_records]
    
    g_tp = sum(1 for t, p in zip(gen_targets, gen_preds) if t == 1 and p == 1)
    g_tn = sum(1 for t, p in zip(gen_targets, gen_preds) if t == 0 and p == 0)
    g_fp = sum(1 for t, p in zip(gen_targets, gen_preds) if t == 0 and p == 1)
    g_fn = sum(1 for t, p in zip(gen_targets, gen_preds) if t == 1 and p == 0)
    
    g_acc  = (g_tp + g_tn) / len(gen_targets)
    g_prec = g_tp / (g_tp + g_fp) if (g_tp + g_fp) > 0 else 0.0
    g_rec  = g_tp / (g_tp + g_fn) if (g_tp + g_fn) > 0 else 0.0
    g_f1   = 2 * g_prec * g_rec / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0
    
    print("A) Generation Branch (Frozen V4 Frequency Model):")
    print(f"   Evaluated on {len(gen_eval_records)} unaltered genuine/generated test images:")
    print(f"   Accuracy : {g_acc * 100:.2f}%")
    print(f"   Precision: {g_prec:.4f}")
    print(f"   Recall   : {g_rec:.4f}")
    print(f"   F1 Score : {g_f1:.4f}")
    print(f"   Confusion Matrix: TP={g_tp}, TN={g_tn}, FP={g_fp}, FN={g_fn}")
    print()

    # Manipulation Branch: Binary task (0=ORIGINAL_REAL, 1=AI_MANIPULATED)
    # Evaluated on manipulation_v1/test/ (248 images: 94 original, 154 manipulated)
    manip_eval_records = [r for r in evaluated_records if r["source_dataset"].startswith("manipulation_v1/test")]
    manip_targets = [1 if r["ground_truth"] == "AI_MANIPULATED" else 0 for r in manip_eval_records]
    manip_preds   = [1 if r["manipulation_prediction"] == "AI_MANIPULATED" else 0 for r in manip_eval_records]
    
    m_tp = sum(1 for t, p in zip(manip_targets, manip_preds) if t == 1 and p == 1)
    m_tn = sum(1 for t, p in zip(manip_targets, manip_preds) if t == 0 and p == 0)
    m_fp = sum(1 for t, p in zip(manip_targets, manip_preds) if t == 0 and p == 1)
    m_fn = sum(1 for t, p in zip(manip_targets, manip_preds) if t == 1 and p == 0)
    
    m_acc  = (m_tp + m_tn) / len(manip_targets)
    m_prec = m_tp / (m_tp + m_fp) if (m_tp + m_fp) > 0 else 0.0
    m_rec  = m_tp / (m_tp + m_fn) if (m_tp + m_fn) > 0 else 0.0
    m_f1   = 2 * m_prec * m_rec / (m_prec + m_rec) if (m_prec + m_rec) > 0 else 0.0
    
    print("B) Manipulation Branch (Frozen Phase 7 Frequency Model):")
    print(f"   Evaluated on {len(manip_eval_records)} manipulation_v1 test images (94 Orig, 154 Manip):")
    print(f"   Accuracy : {m_acc * 100:.2f}%")
    print(f"   Precision: {m_prec:.4f}")
    print(f"   Recall   : {m_rec:.4f}")
    print(f"   F1 Score : {m_f1:.4f}")
    print(f"   Confusion Matrix: TP={m_tp}, TN={m_tn}, FP={m_fp}, FN={m_fn}")
    print()

    # 7. Decision Engine Behavior & Cross-Branch Agreement
    print("-" * 65)
    print("7. DECISION ENGINE BEHAVIOR & DISAGREEMENT ANALYSIS")
    print("-" * 65)
    
    disagreements = Counter()
    case_counts   = Counter()
    
    for r in evaluated_records:
        pair_state = f"Gen={r['generation_prediction']} | Manip={r['manipulation_prediction']}"
        disagreements[pair_state] += 1
        case_counts[r["decision_case"]] += 1
        
    print("Cross-Branch Combination Frequencies:")
    for pair, cnt in disagreements.most_common():
        print(f"  - {pair:<45}: {cnt:>3} ({cnt/total_eval*100:5.2f}%)")
    print()
    print("Decision Engine Case Trigger Frequencies:")
    for c_name, cnt in case_counts.most_common():
        print(f"  - {c_name:<30}: {cnt:>3} ({cnt/total_eval*100:5.2f}%)")
    print()

    # 8. Confidence Analysis
    print("-" * 65)
    print("8. CONFIDENCE STATISTICS")
    print("-" * 65)
    
    conf_by_gt = defaultdict(list)
    conf_by_pred = defaultdict(list)
    for r in evaluated_records:
        conf_by_gt[r["ground_truth"]].append(r["final_confidence"])
        conf_by_pred[r["final_state"]].append(r["final_confidence"])
        
    print("A) Confidence by Ground-Truth Class:")
    conf_gt_stats = {}
    for c in classes_3:
        vals = conf_by_gt[c]
        conf_gt_stats[c] = {
            "mean": round(float(np.mean(vals)), 4),
            "median": round(float(np.median(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }
        print(f"  {c:<15}: Mean={conf_gt_stats[c]['mean']:.4f} | Median={conf_gt_stats[c]['median']:.4f} | Min={conf_gt_stats[c]['min']:.4f} | Max={conf_gt_stats[c]['max']:.4f}")
    print()

    print("B) Confidence by Final Decision State:")
    conf_pred_stats = {}
    for s in final_states_4:
        vals = conf_by_pred[s]
        if vals:
            conf_pred_stats[s] = {
                "mean": round(float(np.mean(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }
            print(f"  {s:<15}: Mean={conf_pred_stats[s]['mean']:.4f} | Median={conf_pred_stats[s]['median']:.4f} | Min={conf_pred_stats[s]['min']:.4f} | Max={conf_pred_stats[s]['max']:.4f}")
        else:
            conf_pred_stats[s] = {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
            print(f"  {s:<15}: No samples")
    print()

    # 9. Category Analysis for Manipulated Test Set
    print("-" * 65)
    print("9. CATEGORY-WISE ANALYSIS (manipulation_v1/test/manipulated/)")
    print("-" * 65)
    
    manip_only_records = [r for r in evaluated_records if r["source_dataset"] == "manipulation_v1/test/manipulated"]
    cat_groups = defaultdict(list)
    for r in manip_only_records:
        cat_groups[r["source_category"]].append(r)
        
    category_summary = []
    print(f"{'Category':<25} | {'N':>3} | {'Correct AI_MANIP':<16} | {'Final Acc':<10} | {'Recall':<8} | {'UNCERTAIN Rate':<14}")
    print("-" * 88)
    for cat in sorted(cat_groups.keys()):
        group = cat_groups[cat]
        n_c = len(group)
        correct_c = sum(1 for r in group if r["final_state"] == "AI_MANIPULATED")
        acc_c = correct_c / n_c
        rec_c = correct_c / n_c
        unc_c = sum(1 for r in group if r["final_state"] == "UNCERTAIN") / n_c
        
        entry = {
            "category": cat,
            "sample_count": n_c,
            "correct_final_decisions": correct_c,
            "final_accuracy": round(acc_c, 4),
            "ai_manipulated_recall": round(rec_c, 4),
            "uncertain_rate": round(unc_c, 4),
            "is_small_sample": (n_c < 10),
        }
        category_summary.append(entry)
        note = " [small sample]" if n_c < 10 else ""
        print(f"{cat:<25} | {n_c:>3} | {correct_c:>16} | {acc_c*100:>9.2f}% | {rec_c:>8.4f} | {unc_c*100:>12.2f}%{note}")
    print("-" * 88)
    print()

    # 10. Pair-Level Analysis on manipulation_v1 Pairs
    print("-" * 65)
    print("10. PAIR-AWARE FORENSIC CONSISTENCY ANALYSIS")
    print("-" * 65)
    
    pair_metrics_summary = {}
    pair_rows = []
    if PAIR_JSON_PATH.exists():
        with open(PAIR_JSON_PATH, "r", encoding="utf-8") as f:
            pair_map = json.load(f)
            
        pred_by_img_id = {r["image_id"]: r for r in evaluated_records if r["source_dataset"].startswith("manipulation_v1/test")}
        
        # Test original IDs
        test_orig_ids = [r["image_id"] for r in evaluated_records if r["source_dataset"] == "manipulation_v1/test/original"]
        
        both_correct = 0
        orig_corr_manip_inc = 0
        orig_inc_manip_corr = 0
        both_incorrect = 0
        total_pairs_eval = 0
        
        for orig_id in test_orig_ids:
            if orig_id not in pred_by_img_id:
                continue
            orig_rec = pred_by_img_id[orig_id]
            orig_ok = (orig_rec["final_state"] == "REAL_ORIGINAL")
            
            manip_list = pair_map.get(f"m1_test_orig_{orig_id}", [])
            for manip_full_id in manip_list:
                manip_id = manip_full_id.replace("m1_test_manip_", "")
                if manip_id in pred_by_img_id:
                    manip_rec = pred_by_img_id[manip_id]
                    manip_ok = (manip_rec["final_state"] == "AI_MANIPULATED")
                    
                    total_pairs_eval += 1
                    if orig_ok and manip_ok:
                        both_correct += 1
                        case_pair = "BOTH_CORRECT"
                    elif orig_ok and not manip_ok:
                        orig_corr_manip_inc += 1
                        case_pair = "ORIGINAL_CORRECT_MANIP_INCORRECT"
                    elif not orig_ok and manip_ok:
                        orig_inc_manip_corr += 1
                        case_pair = "ORIGINAL_INCORRECT_MANIP_CORRECT"
                    else:
                        both_incorrect += 1
                        case_pair = "BOTH_INCORRECT"
                        
                    pair_rows.append({
                        "original_id": orig_id,
                        "manipulated_id": manip_id,
                        "original_final_state": orig_rec["final_state"],
                        "manipulated_final_state": manip_rec["final_state"],
                        "pair_outcome": case_pair,
                        "is_pair_success": int(orig_ok and manip_ok),
                    })

        pair_rate = both_correct / total_pairs_eval if total_pairs_eval > 0 else 0.0
        pair_metrics_summary = {
            "total_test_pairs": total_pairs_eval,
            "both_correct": both_correct,
            "original_correct_manip_incorrect": orig_corr_manip_inc,
            "original_incorrect_manip_correct": orig_inc_manip_corr,
            "both_incorrect": both_incorrect,
            "pair_level_success_rate": round(pair_rate, 4),
        }
        
        print(f"Total Test Pairs Evaluated                 : {total_pairs_eval}")
        print(f"  - Both Correct (Original=REAL, Manip=MANIP): {both_correct:>3} ({both_correct/total_pairs_eval*100:>5.2f}%)")
        print(f"  - Original Correct / Manipulated Incorrect : {orig_corr_manip_inc:>3} ({orig_corr_manip_inc/total_pairs_eval*100:>5.2f}%)")
        print(f"  - Original Incorrect / Manipulated Correct : {orig_inc_manip_corr:>3} ({orig_inc_manip_corr/total_pairs_eval*100:>5.2f}%)")
        print(f"  - Both Incorrect                           : {both_incorrect:>3} ({both_incorrect/total_pairs_eval*100:>5.2f}%)")
        print(f"Pair-Level Success Rate                    : {pair_rate * 100:.2f}%")
    else:
        print("Pair relationships JSON not found; pair analysis skipped.")
    print()

    # 11. Unlabeled / Qualitative Evaluation
    print("-" * 65)
    print("11. UNLABELED / QUALITATIVE EVALUATION (SEPARATE FROM ACCURACY)")
    print("-" * 65)
    print("Explicit Notice: The datasets below lack verified ground-truth provenance.")
    print("Distributions reflect raw model inferences and are NOT accuracy measurements.\n")
    
    unlabeled_eval_summary = {}
    
    for unlab_name, unlab_list in [("external_test", ext_samples), ("real_world_test", real_samples)]:
        if not unlab_list:
            continue
        print(f"Dataset: {unlab_name} ({len(unlab_list)} images):")
        u_records = []
        u_final_counts = Counter()
        u_gen_counts   = Counter()
        u_manip_counts = Counter()
        u_confs        = []
        
        for item in unlab_list:
            res = pipeline.predict(item["filepath"])
            gen = res["generation"]
            manip = res["manipulation"]
            fin = res["final"]
            
            u_final_counts[fin["label"]] += 1
            u_gen_counts[gen["label"]] += 1
            u_manip_counts[manip["label"]] += 1
            u_confs.append(fin["confidence"])
            
            u_records.append({
                "image_path": item["rel_path"],
                "source_dataset": item["source_dataset"],
                "source_category": item["source_category"],
                "ground_truth": "UNLABELED",
                "generation_real_probability": gen["probability_real"],
                "generation_ai_probability": gen["probability_ai_generated"],
                "generation_prediction": gen["label"],
                "manipulation_original_probability": manip["probability_original"],
                "manipulation_manipulated_probability": manip["probability_ai_manipulated"],
                "manipulation_prediction": manip["label"],
                "final_state": fin["label"],
                "final_confidence": fin["confidence"],
                "decision_case": fin["decision_case"],
                "decision_reason": fin["reason"],
            })
            evaluated_records.append(u_records[-1])
            
        unlabeled_eval_summary[unlab_name] = {
            "total_images": len(unlab_list),
            "final_state_counts": dict(u_final_counts),
            "generation_prediction_counts": dict(u_gen_counts),
            "manipulation_prediction_counts": dict(u_manip_counts),
            "mean_confidence": round(float(np.mean(u_confs)), 4),
            "median_confidence": round(float(np.median(u_confs)), 4),
        }
        
        print(f"  Final Decision Distribution:")
        for s in final_states_4:
            c = u_final_counts[s]
            print(f"    - {s:<15}: {c:>2} ({c/len(unlab_list)*100:5.1f}%)")
        print(f"  Generation Predictions  : {dict(u_gen_counts)}")
        print(f"  Manipulation Predictions: {dict(u_manip_counts)}")
        print(f"  Mean Final Confidence   : {unlabeled_eval_summary[unlab_name]['mean_confidence']:.4f}")
        print()

    # 12. Checkpoint Verification (Post-Evaluation)
    print("-" * 65)
    print("12. MODEL CHECKPOINT VERIFICATION (POST-EVALUATION)")
    print("-" * 65)
    gen_hash_post = compute_file_sha256(GEN_CKPT_PATH)
    manip_hash_post = compute_file_sha256(MANIP_CKPT_PATH)
    
    gen_match_post = (gen_hash_post == EXPECTED_GEN_HASH)
    manip_match_post = (manip_hash_post == EXPECTED_MANIP_HASH)
    
    print(f"Generation Model   : {'MATCH' if gen_match_post else 'MISMATCH'} ({gen_hash_post})")
    print(f"Manipulation Model : {'MATCH' if manip_match_post else 'MISMATCH'} ({manip_hash_post})")
    assert gen_match_post and manip_match_post, "Checkpoints altered during evaluation!"
    print("-> Confirmed: Checkpoints remained 100% frozen and invariant.")
    print()

    # 13. Save Output Artifacts
    print("-" * 65)
    print("13. SAVING INTEGRATED EVALUATION ARTIFACTS")
    print("-" * 65)
    
    # 1. integrated_predictions.csv
    pred_csv_path = RESULTS_DIR / "integrated_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        fields = [
            "image_path", "source_dataset", "source_category", "ground_truth",
            "generation_real_probability", "generation_ai_probability", "generation_prediction",
            "manipulation_original_probability", "manipulation_manipulated_probability", "manipulation_prediction",
            "final_state", "final_confidence", "decision_reason", "decision_case", "is_correct"
        ]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(evaluated_records)
    print(f"  - {pred_csv_path.name} ({len(evaluated_records)} rows)")

    # 2. integrated_metrics.json
    metrics_json_path = RESULTS_DIR / "integrated_metrics.json"
    full_metrics = {
        "evaluation_timestamp": timestamp,
        "total_primary_labeled": total_eval,
        "overall_3class_accuracy": round(overall_accuracy, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "overall_uncertain_rate": round(uncertain_rate, 4),
        "per_class_metrics": per_class_metrics,
        "confusion_matrix_3x4": cm_3x4,
        "generation_branch_binary": {
            "n_evaluated": len(gen_eval_records),
            "accuracy": round(g_acc, 4),
            "precision": round(g_prec, 4),
            "recall": round(g_rec, 4),
            "f1": round(g_f1, 4),
            "tp": g_tp, "tn": g_tn, "fp": g_fp, "fn": g_fn
        },
        "manipulation_branch_binary": {
            "n_evaluated": len(manip_eval_records),
            "accuracy": round(m_acc, 4),
            "precision": round(m_prec, 4),
            "recall": round(m_rec, 4),
            "f1": round(m_f1, 4),
            "tp": m_tp, "tn": m_tn, "fp": m_fp, "fn": m_fn
        },
        "cross_branch_frequencies": dict(disagreements),
        "decision_case_frequencies": dict(case_counts),
        "confidence_by_ground_truth": conf_gt_stats,
        "confidence_by_final_state": conf_pred_stats,
        "pair_analysis": pair_metrics_summary,
        "unlabeled_evaluation": unlabeled_eval_summary,
    }
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)
    print(f"  - {metrics_json_path.name}")

    # 3. integrated_confusion_matrix.csv
    cm_csv_path = RESULTS_DIR / "integrated_confusion_matrix.csv"
    with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Ground_Truth", "Pred_REAL_ORIGINAL", "Pred_AI_GENERATED", "Pred_AI_MANIPULATED", "Pred_UNCERTAIN", "Total"])
        for c in classes_3:
            r = cm_3x4[c]
            writer.writerow([c, r["REAL_ORIGINAL"], r["AI_GENERATED"], r["AI_MANIPULATED"], r["UNCERTAIN"], sum(r.values())])
    print(f"  - {cm_csv_path.name}")

    # 4. integrated_category_metrics.csv
    cat_csv_path = RESULTS_DIR / "integrated_category_metrics.csv"
    with open(cat_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "category", "sample_count", "correct_final_decisions", "final_accuracy",
            "ai_manipulated_recall", "uncertain_rate", "is_small_sample"
        ])
        writer.writeheader()
        writer.writerows(category_summary)
    print(f"  - {cat_csv_path.name}")

    # 5. integrated_pair_metrics.csv
    pair_csv_path = RESULTS_DIR / "integrated_pair_metrics.csv"
    with open(pair_csv_path, "w", newline="", encoding="utf-8") as f:
        if pair_rows:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)
    print(f"  - {pair_csv_path.name} ({len(pair_rows)} pairs)")

    # 6. integrated_model_hashes.json
    hash_json_path = RESULTS_DIR / "integrated_model_hashes.json"
    with open(hash_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "models": {
                "generation_model": {
                    "path": str(GEN_CKPT_PATH.relative_to(PROJECT_DIR)),
                    "expected_sha256": EXPECTED_GEN_HASH,
                    "sha256_pre": gen_hash_pre,
                    "sha256_post": gen_hash_post,
                    "status": "MATCH" if gen_match_post else "MISMATCH"
                },
                "manipulation_model": {
                    "path": str(MANIP_CKPT_PATH.relative_to(PROJECT_DIR)),
                    "expected_sha256": EXPECTED_MANIP_HASH,
                    "sha256_pre": manip_hash_pre,
                    "sha256_post": manip_hash_post,
                    "status": "MATCH" if manip_match_post else "MISMATCH"
                }
            }
        }, f, indent=2)
    print(f"  - {hash_json_path.name}")

    # 7. integrated_summary.txt
    summary_txt_path = RESULTS_DIR / "integrated_summary.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write("AIDETECT PHASE 11: INTEGRATED FORENSIC EVALUATION SUMMARY REPORT\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Primary Labeled Test Samples: {total_eval}\n\n")
        f.write("1. MODEL CHECKPOINT VERIFICATION:\n")
        f.write(f"  Generation Model   : {GEN_CKPT_PATH.name} -> {gen_hash_post} (MATCH)\n")
        f.write(f"  Manipulation Model : {MANIP_CKPT_PATH.name} -> {manip_hash_post} (MATCH)\n\n")
        f.write("2. INTEGRATED 3-CLASS DECISION PERFORMANCE:\n")
        f.write(f"  Overall Accuracy : {overall_accuracy * 100:.2f}%\n")
        f.write(f"  Macro Precision  : {macro_prec:.4f}\n")
        f.write(f"  Macro Recall     : {macro_rec:.4f}\n")
        f.write(f"  Macro F1         : {macro_f1:.4f}\n")
        f.write(f"  UNCERTAIN Rate   : {uncertain_rate * 100:.2f}%\n\n")
        f.write("  Per-Class Metrics:\n")
        for c in classes_3:
            m = per_class_metrics[c]
            f.write(f"    - {c:<15}: Recall={m['recall']*100:5.2f}% | Precision={m['precision']*100:5.2f}% | F1={m['f1']:.4f} | UNCERTAIN={m['uncertain_rate']*100:.2f}%\n")
        f.write("\n3. INDEPENDENT BRANCH PERFORMANCE:\n")
        f.write(f"  Generation Branch (n={len(gen_eval_records)})  : Acc={g_acc*100:.2f}%, Prec={g_prec:.4f}, Rec={g_rec:.4f}, F1={g_f1:.4f}\n")
        f.write(f"  Manipulation Branch (n={len(manip_eval_records)}): Acc={m_acc*100:.2f}%, Prec={m_prec:.4f}, Rec={m_rec:.4f}, F1={m_f1:.4f}\n\n")
        f.write("4. PAIR ANALYSIS:\n")
        f.write(f"  Total Pairs: {pair_metrics_summary.get('total_test_pairs', 0)}\n")
        f.write(f"  Both Correct: {pair_metrics_summary.get('both_correct', 0)}\n")
        f.write(f"  Pair Success Rate: {pair_metrics_summary.get('pair_level_success_rate', 0)*100:.2f}%\n\n")
        f.write("5. UNLABELED DATA EVALUATION:\n")
        for un_k, un_v in unlabeled_eval_summary.items():
            f.write(f"  {un_k} (n={un_v['total_images']}): Final Decisions={un_v['final_state_counts']}, Mean Conf={un_v['mean_confidence']:.4f}\n")
    print(f"  - {summary_txt_path.name}")

    # Optional: Confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 5))
        matrix_data = [[cm_3x4[gt][pred] for pred in final_states_4] for gt in classes_3]
        plt.imshow(matrix_data, cmap="Blues", interpolation="nearest")
        plt.title("Integrated Forensic Confusion Matrix", fontsize=12)
        plt.colorbar()
        plt.xticks(range(4), final_states_4, rotation=20, ha="right")
        plt.yticks(range(3), classes_3)
        for i in range(3):
            for j in range(4):
                val = matrix_data[i][j]
                color = "white" if val > np.max(matrix_data) / 2 else "black"
                plt.text(j, i, str(val), ha="center", va="center", color=color, fontweight="bold")
        plt.xlabel("Predicted Final State")
        plt.ylabel("Ground Truth")
        plt.tight_layout()
        plot_path = RESULTS_DIR / "integrated_confusion_matrix.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()
        print(f"  - {plot_path.name}")
    except Exception as e:
        print(f"  (Plot generation skipped: {e})")

    print()
    print("=" * 75)
    print("PHASE 11 INTEGRATED EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 75)
    sys.stdout.flush()

if __name__ == "__main__":
    run_integrated_evaluation()

