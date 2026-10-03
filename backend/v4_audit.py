"""
AIDetect V4 Source Integrity Audit + Smoke Test
================================================
Audits recreated V4 source files against training artifacts.
Verifies all pipeline components without training or modifying anything.

Run from VS Code integrated terminal:
    cd C:\\Users\\Shreyas\\OneDrive\\Desktop\\AIDetect\\backend
    .\\venv\\Scripts\\python.exe -u v4_audit.py
"""

import os
import sys
import ast
import json
import time
import inspect
import importlib
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

PASS = "[PASS]"
FAIL = "[FAIL]"
INFO = "[INFO]"
WARN = "[WARN]"

results = {}

def banner(title):
    print()
    print("=" * 70)
    print("  " + title)
    print("=" * 70)
    sys.stdout.flush()

def section(title):
    print()
    print("-" * 60)
    print("  " + title)
    print("-" * 60)
    sys.stdout.flush()

def record(key, status, detail=""):
    results[key] = (status == PASS)
    tag = status
    msg = "  {} {}".format(tag, key)
    if detail:
        msg += "\n         -> {}".format(detail)
    print(msg)
    sys.stdout.flush()

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------

banner("V4 SOURCE INTEGRITY AUDIT")
print("  Working directory : {}".format(BASE_DIR))
print("  Python            : {}".format(sys.executable))
import torch
print("  PyTorch           : {}".format(torch.__version__))
print("  CUDA available    : {}".format(torch.cuda.is_available()))
if torch.cuda.is_available():
    print("  GPU               : {}".format(torch.cuda.get_device_name(0)))
print()
sys.stdout.flush()

# ===========================================================================
# 1. FILE INTEGRITY
# ===========================================================================

banner("1. FILE INTEGRITY CHECK")

V4_FILES = {
    "v4_dataset.py":    BASE_DIR / "v4_dataset.py",
    "v4_models.py":     BASE_DIR / "v4_models.py",
    "train_v4.py":      BASE_DIR / "train_v4.py",
    "evaluate_v4.py":   BASE_DIR / "evaluate_v4.py",
    "v4_sanity_check.py": BASE_DIR / "v4_sanity_check.py",
}

for name, path in V4_FILES.items():
    exists = path.exists()
    if not exists:
        record(name + " exists", FAIL, "FILE NOT FOUND")
        continue
    size = path.stat().st_size
    if size == 0:
        record(name + " non-empty", FAIL, "FILE IS EMPTY (0 bytes)")
        continue
    record(name + " exists ({} bytes)".format(size), PASS)

    # Syntax check
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        record(name + " syntax valid", PASS)
    except SyntaxError as e:
        record(name + " syntax valid", FAIL, str(e))
    except Exception as e:
        record(name + " syntax valid", FAIL, str(e))

# ===========================================================================
# 2. DATASET PIPELINE CHECK
# ===========================================================================

banner("2. DATASET PIPELINE CHECK")

# Reload fresh (bypass __pycache__)
import importlib.util

def load_module_fresh(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

try:
    ds_mod = load_module_fresh("v4_dataset", BASE_DIR / "v4_dataset.py")
    record("v4_dataset imports cleanly", PASS)
except Exception as e:
    record("v4_dataset imports cleanly", FAIL, traceback.format_exc())
    ds_mod = None

if ds_mod:
    # Label convention
    cm = ds_mod.CLASS_MAPPING
    record("CLASS_MAPPING REAL=0", PASS if cm.get("REAL") == 0 else FAIL,
           "got {}".format(cm.get("REAL")))
    record("CLASS_MAPPING AI=1",   PASS if cm.get("AI") == 1 else FAIL,
           "got {}".format(cm.get("AI")))

    itc = ds_mod.IDX_TO_CLASS
    record("IDX_TO_CLASS 0='REAL'", PASS if itc.get(0) == "REAL" else FAIL)
    record("IDX_TO_CLASS 1='AI'",   PASS if itc.get(1) == "AI"   else FAIL)

    # Metadata path
    meta = ds_mod.METADATA_CSV
    record("METADATA_CSV exists", PASS if meta.exists() else FAIL,
           str(meta))

    # Dataset counts
    section("2a. Metadata Counts")
    try:
        train_rows = ds_mod.load_metadata("train")
        val_rows   = ds_mod.load_metadata("validation")
        test_rows  = ds_mod.load_metadata("test")

        train_real = sum(1 for r in train_rows if r["ground_truth"] == "REAL")
        train_ai   = sum(1 for r in train_rows if r["ground_truth"] == "AI")
        val_real   = sum(1 for r in val_rows if r["ground_truth"] == "REAL")
        val_ai     = sum(1 for r in val_rows if r["ground_truth"] == "AI")
        test_real  = sum(1 for r in test_rows if r["ground_truth"] == "REAL")
        test_ai    = sum(1 for r in test_rows if r["ground_truth"] == "AI")

        print("    Train  : REAL={} AI={} Total={}".format(
            train_real, train_ai, len(train_rows)))
        print("    Val    : REAL={} AI={} Total={}".format(
            val_real, val_ai, len(val_rows)))
        print("    Test   : REAL={} AI={} Total={}".format(
            test_real, test_ai, len(test_rows)))
        sys.stdout.flush()

        record("Train total = 2837", PASS if len(train_rows) == 2837 else FAIL,
               "got {}".format(len(train_rows)))
        record("Val   total = 607",  PASS if len(val_rows)   == 607  else FAIL,
               "got {}".format(len(val_rows)))
        record("Test  total = 613",  PASS if len(test_rows)  == 613  else FAIL,
               "got {}".format(len(test_rows)))
    except Exception as e:
        record("load_metadata", FAIL, str(e))

    # Transforms inspection
    section("2b. Transform Inspection")
    try:
        train_xform = ds_mod.SPATIAL_TRAIN_XFORM
        val_xform   = ds_mod.SPATIAL_VAL_XFORM
        fft_xform   = ds_mod.FFT_TRANSFORM

        train_str = str(train_xform)
        val_str   = str(val_xform)

        has_rrc    = "RandomResizedCrop" in train_str
        has_flip   = "RandomHorizontalFlip" in train_str
        has_jitter = "ColorJitter" in train_str
        has_blur   = "GaussianBlur" in train_str
        has_jpeg   = "JPEGSimulation" in train_str or "JPEG" in train_str

        record("Train: RandomResizedCrop present",  PASS if has_rrc    else FAIL)
        record("Train: RandomHorizontalFlip present", PASS if has_flip   else FAIL)
        record("Train: ColorJitter present",         PASS if has_jitter else FAIL)
        record("Train: GaussianBlur present",        PASS if has_blur   else FAIL)
        record("Train: JPEGSimulation present",      PASS if has_jpeg   else FAIL)

        has_resize     = "Resize" in val_str
        has_centercrop = "CenterCrop" in val_str
        has_normalize  = "Normalize" in val_str

        record("Val: Resize present",     PASS if has_resize     else FAIL)
        record("Val: CenterCrop present", PASS if has_centercrop else FAIL)
        record("Val: Normalize present",  PASS if has_normalize  else FAIL)

        print("    FFT Transform class: {}".format(type(fft_xform).__name__))
        record("FFT transform is AuthoritativeFFTTransform",
               PASS if "AuthoritativeFFT" in type(fft_xform).__name__ else FAIL)
        sys.stdout.flush()
    except Exception as e:
        record("transform inspection", FAIL, str(e))

    # PREPROCESSING_DOC keys
    section("2c. PREPROCESSING_DOC")
    pdoc = getattr(ds_mod, "PREPROCESSING_DOC", {})
    for key in ["spatial_train", "spatial_val", "fft", "label_convention", "input_size"]:
        record("PREPROCESSING_DOC has '{}'".format(key),
               PASS if key in pdoc else FAIL)

# ===========================================================================
# 3. FFT PIPELINE CHECK
# ===========================================================================

banner("3. FFT PIPELINE VERIFICATION")

if ds_mod:
    section("3a. FFT Source Code Inspection")
    try:
        fft_src = (BASE_DIR / "v4_dataset.py").read_text(encoding="utf-8")
        print("    AuthoritativeFFTTransform source extracted from v4_dataset.py")
        sys.stdout.flush()

        checks = {
            "fft2 called":                 "fft.fft2" in fft_src or "torch.fft.fft2" in fft_src,
            "fftshift called":             "fftshift" in fft_src,
            "abs magnitude":               "torch.abs" in fft_src,
            "log1p applied":               "log1p" in fft_src,
            "per-channel loop":            "for c in range" in fft_src,
            "min-max normalization":       "ch_min" in fft_src or "m_min" in fft_src or "min_val" in fft_src,
            "F.interpolate resize":        "F.interpolate" in fft_src,
            "bilinear mode":               "bilinear" in fft_src,
            "align_corners=False":         "align_corners=False" in fft_src,
            "ImageNet normalize":           "Normalize" in fft_src,
            "native resolution (no pre-resize)": "resize" not in fft_src.split("class AuthoritativeFFTTransform")[1].split("def __call__")[1].split("torch.fft.fft2")[0].lower(),
        }
        for check, ok in checks.items():
            record("FFT: " + check, PASS if ok else FAIL)

        # Functional test
        section("3b. FFT Functional Test")
        from PIL import Image
        import numpy as np
        sample_path, _, _ = ds_mod.V4Dataset("test", mode="frequency", augment=False).samples[0]
        img = Image.open(sample_path).convert("RGB")
        orig_w, orig_h = img.size
        print("    Sample image: {} x {} px".format(orig_w, orig_h))

        t = ds_mod.FFT_TRANSFORM(img)
        record("FFT output shape (3,224,224)",
               PASS if tuple(t.shape) == (3,224,224) else FAIL,
               "got {}".format(tuple(t.shape)))
        record("FFT output finite (no NaN/Inf)",
               PASS if torch.isfinite(t).all().item() else FAIL)
        record("FFT output is float32",
               PASS if t.dtype == torch.float32 else FAIL,
               "got {}".format(t.dtype))

        print("    FFT tensor range: [{:.4f}, {:.4f}]".format(
            t.min().item(), t.max().item()))

        # Cross-check with manual pipeline
        import torchvision.transforms as tvt
        import torch.nn.functional as F
        x = tvt.ToTensor()(img)
        fft = torch.fft.fft2(x)
        fft = torch.fft.fftshift(fft, dim=(-2,-1))
        mag = torch.log1p(torch.abs(fft))
        for c in range(3):
            mn, mx = mag[c].min(), mag[c].max()
            mag[c] = (mag[c] - mn) / (mx - mn + 1e-8)
        mag = F.interpolate(mag.unsqueeze(0), size=(224,224), mode="bilinear", align_corners=False).squeeze(0)
        norm = tvt.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        mag = norm(mag)
        max_diff = float((t - mag).abs().max())
        record("FFT matches manual pipeline (max_diff < 1e-4)",
               PASS if max_diff < 1e-4 else FAIL,
               "max_diff={:.8f}".format(max_diff))
        sys.stdout.flush()

    except Exception as e:
        record("FFT inspection", FAIL, traceback.format_exc())

# ===========================================================================
# 4. MODEL ARCHITECTURES
# ===========================================================================

banner("4. MODEL ARCHITECTURE CHECK")

try:
    mdl_mod = load_module_fresh("v4_models", BASE_DIR / "v4_models.py")
    record("v4_models imports cleanly", PASS)
except Exception as e:
    record("v4_models imports cleanly", FAIL, str(e))
    mdl_mod = None

if mdl_mod:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Constants
    record("SPATIAL_FEATURE_DIM = 2048",
           PASS if mdl_mod.SPATIAL_FEATURE_DIM == 2048 else FAIL,
           "got {}".format(mdl_mod.SPATIAL_FEATURE_DIM))
    record("FREQUENCY_FEATURE_DIM = 256",
           PASS if mdl_mod.FREQUENCY_FEATURE_DIM == 256 else FAIL,
           "got {}".format(mdl_mod.FREQUENCY_FEATURE_DIM))
    record("FUSED_DIM = 2304",
           PASS if mdl_mod.FUSED_DIM == 2304 else FAIL,
           "got {}".format(mdl_mod.FUSED_DIM))
    record("NUM_CLASSES = 2",
           PASS if mdl_mod.NUM_CLASSES == 2 else FAIL,
           "got {}".format(mdl_mod.NUM_CLASSES))

    section("4a. SpatialV4")
    try:
        m = mdl_mod.SpatialV4().to(DEVICE)
        m.eval()
        dummy = torch.randn(2,3,224,224).to(DEVICE)
        with torch.no_grad():
            out = m(dummy)
        record("SpatialV4 forward shape [2,2]",
               PASS if tuple(out.shape)==(2,2) else FAIL, str(tuple(out.shape)))

        # Check it uses pretrained weights
        models_src = (BASE_DIR / "v4_models.py").read_text(encoding="utf-8")
        src = models_src
        record("SpatialV4 uses ResNet50_Weights.DEFAULT",
               PASS if "ResNet50_Weights.DEFAULT" in src else FAIL)
        record("SpatialV4 has Linear head",
               PASS if "nn.Linear" in src else FAIL)

        # Check head dim
        fc = m.model.fc
        record("SpatialV4 fc: in_features=2048",
               PASS if fc.in_features == 2048 else FAIL,
               "got {}".format(fc.in_features))
        record("SpatialV4 fc: out_features=2",
               PASS if fc.out_features == 2 else FAIL,
               "got {}".format(fc.out_features))
        del m
        torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
    except Exception as e:
        record("SpatialV4 check", FAIL, traceback.format_exc())

    section("4b. FrequencyV4")
    try:
        m = mdl_mod.FrequencyV4().to(DEVICE)
        m.eval()
        dummy_fft = torch.randn(2,3,224,224).to(DEVICE)
        with torch.no_grad():
            out = m(dummy_fft)
        record("FrequencyV4 forward shape [2,2]",
               PASS if tuple(out.shape)==(2,2) else FAIL, str(tuple(out.shape)))

        src = models_src
        record("FrequencyV4 uses ResNet50_Weights.DEFAULT",
               PASS if "ResNet50_Weights.DEFAULT" in src else FAIL)
        fc = m.model.fc
        record("FrequencyV4 fc: out_features=2",
               PASS if fc.out_features==2 else FAIL, str(fc.out_features))
        del m
        torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
    except Exception as e:
        record("FrequencyV4 check", FAIL, traceback.format_exc())

    section("4c. HybridV4")
    try:
        m = mdl_mod.HybridV4().to(DEVICE)
        m.eval()
        dummy_s = torch.randn(2,3,224,224).to(DEVICE)
        dummy_f = torch.randn(2,3,224,224).to(DEVICE)

        with torch.no_grad():
            sf = torch.flatten(m.spatial_features(dummy_s), 1)
            ff = torch.flatten(m.frequency_features(dummy_f), 1)
        record("HybridV4 spatial branch dim = 2048",
               PASS if sf.shape[1]==2048 else FAIL, str(sf.shape[1]))
        record("HybridV4 frequency branch dim = 256",
               PASS if ff.shape[1]==256 else FAIL, str(ff.shape[1]))
        record("HybridV4 fused dim = 2304",
               PASS if sf.shape[1]+ff.shape[1]==2304 else FAIL,
               "{}+{}={}".format(sf.shape[1], ff.shape[1], sf.shape[1]+ff.shape[1]))

        with torch.no_grad():
            out = m(dummy_s, dummy_f)
        record("HybridV4 forward shape [2,2]",
               PASS if tuple(out.shape)==(2,2) else FAIL, str(tuple(out.shape)))

        # Classifier dims
        clf = m.classifier
        record("HybridV4 classifier[0]: Linear(2304,512)",
               PASS if isinstance(clf[0], torch.nn.Linear)
               and clf[0].in_features==2304 and clf[0].out_features==512 else FAIL,
               "in={} out={}".format(clf[0].in_features, clf[0].out_features))
        record("HybridV4 classifier[3]: Linear(512,2)",
               PASS if isinstance(clf[3], torch.nn.Linear)
               and clf[3].in_features==512 and clf[3].out_features==2 else FAIL,
               "in={} out={}".format(clf[3].in_features, clf[3].out_features))
        record("HybridV4 classifier[2]: Dropout(0.3)",
               PASS if isinstance(clf[2], torch.nn.Dropout)
               and abs(clf[2].p - 0.3) < 0.001 else FAIL,
               "p={}".format(clf[2].p if isinstance(clf[2], torch.nn.Dropout) else "N/A"))

        # Verify pretrained spatial
        src = models_src
        record("HybridV4 spatial uses ResNet50_Weights.DEFAULT",
               PASS if "ResNet50_Weights.DEFAULT" in src else FAIL)
        record("HybridV4 fusion is concatenation (torch.cat)",
               PASS if "torch.cat" in src else FAIL)

        # Conv branch layers
        freq_layers = list(m.frequency_features.children())
        conv_layers = [l for l in freq_layers if isinstance(l, torch.nn.Conv2d)]
        record("HybridV4 frequency branch has 4 Conv2d layers",
               PASS if len(conv_layers)==4 else FAIL,
               "found {}".format(len(conv_layers)))
        record("HybridV4 freq Conv2d channels: 3->32->64->128->256",
               PASS if [c.in_channels for c in conv_layers]==[3,32,64,128]
               and [c.out_channels for c in conv_layers]==[32,64,128,256] else FAIL,
               "in={} out={}".format(
                   [c.in_channels for c in conv_layers],
                   [c.out_channels for c in conv_layers]))

        del m
        torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
    except Exception as e:
        record("HybridV4 check", FAIL, traceback.format_exc())

# ===========================================================================
# 5. TRAINING CONFIGURATION CHECK
# ===========================================================================

banner("5. TRAINING CONFIGURATION CHECK")

try:
    trn_src = (BASE_DIR / "train_v4.py").read_text(encoding="utf-8")

    cfg_checks = {
        "SEED = 42":                      "SEED         = 42" in trn_src or "SEED=42" in trn_src,
        "BATCH_SIZE = 16":                "BATCH_SIZE   = 16" in trn_src or "BATCH_SIZE=16" in trn_src,
        "MAX_EPOCHS = 15":                "MAX_EPOCHS   = 15" in trn_src or "MAX_EPOCHS=15" in trn_src,
        "LR = 1e-4":                      "LR           = 1e-4" in trn_src or "LR = 1e-4" in trn_src,
        "WEIGHT_DECAY = 1e-3":            "WEIGHT_DECAY = 1e-3" in trn_src or "WEIGHT_DECAY=1e-3" in trn_src,
        "T_MAX = 15":                     "T_MAX        = 15" in trn_src or "T_MAX = 15" in trn_src,
        "ETA_MIN = 1e-6":                 "ETA_MIN      = 1e-6" in trn_src or "ETA_MIN = 1e-6" in trn_src,
        "ES_PATIENCE = 5":                "ES_PATIENCE  = 5" in trn_src or "ES_PATIENCE = 5" in trn_src,
        "AdamW optimizer":                "AdamW" in trn_src,
        "CosineAnnealingLR":              "CosineAnnealingLR" in trn_src,
        "AMP (autocast)":                 "autocast" in trn_src,
        "GradScaler":                     "GradScaler" in trn_src,
        "Early stopping on val F1":       "val_f1" in trn_src and "ES_PATIENCE" in trn_src,
        "Best checkpoint by val F1":      "best_val_f1" in trn_src,
        "Class weights":                  "class_weights" in trn_src,
        "CrossEntropyLoss with weights":  "CrossEntropyLoss(weight=" in trn_src,
        "Seed set for torch":             "torch.manual_seed" in trn_src,
        "Seed set for numpy":             "np.random.seed" in trn_src,
        "Seed set for random":            "random.seed" in trn_src,
        "CUDA seed":                      "cuda.manual_seed" in trn_src,
        "V3 checkpoint check before train":"spatial_resnet50_v3" in trn_src,
        "Checkpoint saves model_state_dict":"model_state_dict" in trn_src,
        "Checkpoint saves class_mapping": "class_mapping" in trn_src,
        "Checkpoint saves preprocessing": "preprocessing" in trn_src,
        "Three models trained":           "SpatialV4" in trn_src and "FrequencyV4" in trn_src and "HybridV4" in trn_src,
        "Results saved to v4_results/":   "v4_results" in trn_src,
    }
    for check, ok in cfg_checks.items():
        record("Train config: " + check, PASS if ok else FAIL)
except Exception as e:
    record("train_v4.py read", FAIL, str(e))

# ===========================================================================
# 6. EVALUATION LOGIC CHECK
# ===========================================================================

banner("6. EVALUATION LOGIC CHECK")

try:
    eval_src = (BASE_DIR / "evaluate_v4.py").read_text(encoding="utf-8")

    eval_checks = {
        "Loads from test split only":         "test" in eval_src and "augment=False" in eval_src,
        "No augment on test":                 'augment=False' in eval_src,
        "accuracy_score":                     "accuracy_score" in eval_src,
        "precision_score":                    "precision_score" in eval_src,
        "recall_score":                       "recall_score" in eval_src,
        "f1_score":                           "f1_score" in eval_src,
        "confusion_matrix":                   "confusion_matrix" in eval_src,
        "TP/TN/FP/FN computed":               '"TP"' in eval_src and '"TN"' in eval_src,
        "REAL recall computed":               "real_recall" in eval_src,
        "AI recall computed":                 "ai_recall" in eval_src,
        "Domain-specific metrics":            "domain_metrics" in eval_src and "ALL_DOMAINS" in eval_src,
        "v4_test_metrics.json saved":         "v4_test_metrics.json" in eval_src,
        "v4_test_predictions.csv saved":      "v4_test_predictions.csv" in eval_src,
        "v4_test_domain_metrics.json saved":  "v4_test_domain_metrics.json" in eval_src,
        "Confusion matrix json saved":        "v4_confusion_matrix" in eval_src,
        "V3 checkpoints verified":            "spatial_resnet50_v3" in eval_src,
        "Note: test set not for selection":   "NOT used for model selection" in eval_src or "not used for model selection" in eval_src.lower(),
        "Loads from checkpoint (model_state_dict)": "model_state_dict" in eval_src,
        "Separate REAL/AI domain lists":      "DOMAINS_REAL" in eval_src and "DOMAINS_AI" in eval_src,
    }
    for check, ok in eval_checks.items():
        record("Eval: " + check, PASS if ok else FAIL)
except Exception as e:
    record("evaluate_v4.py read", FAIL, str(e))

# ===========================================================================
# 7. CHECKPOINT VERIFICATION
# ===========================================================================

banner("7. CHECKPOINT VERIFICATION")

MODEL_DIR = PROJECT_DIR / "models"

# V4 checkpoints
section("7a. V4 Checkpoints")
V4_CKPTS = {
    "spatial_resnet50_v4.pth":        None,
    "frequency_resnet50_v4.pth":      None,
    "hybrid_resnet50_fft_v4.pth":     None,
}

for fname in V4_CKPTS:
    p = MODEL_DIR / fname
    if not p.exists():
        record(fname + " exists", FAIL, "MISSING")
        continue
    size_mb = p.stat().st_size / 1e6
    record("{} exists ({:.1f} MB)".format(fname, size_mb), PASS)

    try:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            keys = list(ckpt.keys())
            has_state     = "model_state_dict" in ckpt
            has_version   = "version" in ckpt
            has_mapping   = "class_mapping" in ckpt
            has_preproc   = "preprocessing" in ckpt
            has_epoch     = "epoch" in ckpt
            has_val_f1    = "best_val_f1" in ckpt

            record(fname + ": model_state_dict present", PASS if has_state   else FAIL)
            record(fname + ": version=V4",
                   PASS if ckpt.get("version") == "V4" else FAIL,
                   "got {}".format(ckpt.get("version")))
            record(fname + ": class_mapping correct",
                   PASS if ckpt.get("class_mapping") == {"0":"REAL","1":"AI"} else FAIL,
                   str(ckpt.get("class_mapping")))
            record(fname + ": preprocessing embedded", PASS if has_preproc else FAIL)
            record(fname + ": epoch recorded",
                   PASS if has_epoch else FAIL,
                   "epoch={}".format(ckpt.get("epoch")))
            record(fname + ": best_val_f1 recorded",
                   PASS if has_val_f1 else FAIL,
                   "val_f1={:.4f}".format(float(ckpt.get("best_val_f1",0))))

            print("    {} metadata summary:".format(fname))
            print("      model_name  : {}".format(ckpt.get("model_name","?")))
            print("      version     : {}".format(ckpt.get("version","?")))
            print("      epoch       : {}".format(ckpt.get("epoch","?")))
            print("      best_val_f1 : {:.4f}".format(float(ckpt.get("best_val_f1",0))))
            print("      input_size  : {}".format(ckpt.get("input_size","?")))
            print("      class_mapping: {}".format(ckpt.get("class_mapping","?")))
            sys.stdout.flush()

            V4_CKPTS[fname] = ckpt
        else:
            record(fname + ": has metadata dict", FAIL, "raw state dict (no metadata)")
    except Exception as e:
        record(fname + ": loadable", FAIL, str(e))

# V3 checkpoints (must exist, must not be modified)
section("7b. V3 Checkpoints (Preservation Check)")
V3_SIZES_EXPECTED = {
    "spatial_resnet50_v3.pth":    94370439,
    "frequency_resnet50_v3.pth":  94371091,
    "hybrid_resnet50_fft_v3.pth": 100657695,
}
for fname, expected_size in V3_SIZES_EXPECTED.items():
    p = MODEL_DIR / fname
    if not p.exists():
        record("V3 " + fname + " exists", FAIL, "MISSING!")
        continue
    actual_size = p.stat().st_size
    record("V3 {} exists ({} bytes)".format(fname, actual_size), PASS)
    record("V3 {} size unchanged".format(fname),
           PASS if actual_size == expected_size else WARN,
           "expected={} actual={}".format(expected_size, actual_size))

# ===========================================================================
# 8. CROSS-CHECK AGAINST TRAINING ARTIFACTS
# ===========================================================================

banner("8. CROSS-CHECK AGAINST TRAINING ARTIFACTS")

section("8a. Training Config JSON vs Source")
config_path = BASE_DIR / "v4_results" / "configs" / "v4_training_config.json"
if config_path.exists():
    with open(config_path) as f:
        saved_cfg = json.load(f)
    print("  Saved config keys: {}".format(list(saved_cfg.keys())))

    src_checks = {
        "seed":          (saved_cfg.get("seed"), 42),
        "batch_size":    (saved_cfg.get("batch_size"), 16),
        "max_epochs":    (saved_cfg.get("max_epochs"), 15),
        "learning_rate": (saved_cfg.get("learning_rate"), 1e-4),
        "weight_decay":  (saved_cfg.get("weight_decay"), 1e-3),
        "optimizer":     (saved_cfg.get("optimizer"), "AdamW"),
    }
    for key, (got, expected) in src_checks.items():
        record("Config: {} = {}".format(key, expected),
               PASS if got == expected else FAIL,
               "got {}".format(got))
    sys.stdout.flush()
else:
    record("v4_training_config.json exists", FAIL, str(config_path))

section("8b. Training Summary vs Checkpoints")
summary_path = BASE_DIR / "v4_results" / "metrics" / "v4_training_summary.json"
if summary_path.exists():
    with open(summary_path) as f:
        summary = json.load(f)
    for r in summary.get("results", []):
        model_name = r["model"]
        best_epoch = r["best_epoch"]
        best_f1    = r["best_val_f1"]
        print("  {}: best_epoch={} best_val_f1={:.4f}".format(
            model_name, best_epoch, best_f1))

        # Match against checkpoint
        fname_map = {
            "Spatial V4":   "spatial_resnet50_v4.pth",
            "Frequency V4": "frequency_resnet50_v4.pth",
            "Hybrid V4":    "hybrid_resnet50_fft_v4.pth",
        }
        ckpt_data = V4_CKPTS.get(fname_map.get(model_name, ""), None)
        if ckpt_data and isinstance(ckpt_data, dict):
            ckpt_epoch = ckpt_data.get("epoch", -1)
            ckpt_f1    = float(ckpt_data.get("best_val_f1", -1))
            epoch_ok   = ckpt_epoch == best_epoch
            f1_ok      = abs(ckpt_f1 - best_f1) < 1e-6
            record("{}: checkpoint epoch matches summary".format(model_name),
                   PASS if epoch_ok else FAIL,
                   "summary={} ckpt={}".format(best_epoch, ckpt_epoch))
            record("{}: checkpoint F1 matches summary".format(model_name),
                   PASS if f1_ok else FAIL,
                   "summary={:.6f} ckpt={:.6f}".format(best_f1, ckpt_f1))
    sys.stdout.flush()
else:
    record("v4_training_summary.json exists", FAIL, str(summary_path))

section("8c. Test Metrics File Verification")
test_metrics_path = BASE_DIR / "v4_results" / "validation_results" / "v4_test_metrics.json"
if test_metrics_path.exists():
    with open(test_metrics_path) as f:
        test_metrics = json.load(f)
    for model_name in ["Spatial V4", "Frequency V4", "Hybrid V4"]:
        m = test_metrics.get("results", {}).get(model_name, {}).get("overall", {})
        print("  {}: acc={:.2f}% f1={:.4f} real_r={:.4f} ai_r={:.4f}".format(
            model_name,
            m.get("accuracy",0)*100, m.get("f1",0),
            m.get("real_recall",0), m.get("ai_recall",0)))
    sys.stdout.flush()
    record("v4_test_metrics.json exists and readable", PASS)
else:
    record("v4_test_metrics.json exists", FAIL, str(test_metrics_path))

# ===========================================================================
# 9. SMOKE TEST
# ===========================================================================

banner("9. SMOKE TEST (one batch, forward pass, no training)")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
record("CUDA device active", PASS if DEVICE.type=="cuda" else FAIL,
       "device={}".format(DEVICE))

if ds_mod and mdl_mod:
    section("9a. Dataset Load + One Batch (Spatial)")
    try:
        from torch.utils.data import DataLoader
        ds = ds_mod.V4Dataset("test", mode="spatial", augment=False)
        loader = DataLoader(ds, batch_size=4, shuffle=False,
                           collate_fn=ds_mod.collate_spatial)
        batch = next(iter(loader))
        x, y, domains = batch
        record("Spatial batch x shape (4,3,224,224)",
               PASS if tuple(x.shape)==(4,3,224,224) else FAIL, str(tuple(x.shape)))
        record("Spatial batch labels in {0,1}",
               PASS if all(l in (0,1) for l in y.tolist()) else FAIL,
               "labels={}".format(y.tolist()))
        record("Spatial batch domains are strings",
               PASS if all(isinstance(d,str) for d in domains) else FAIL,
               "domains={}".format(domains))
        print("    Labels: {} -> {}".format(
            y.tolist(), ["REAL" if l==0 else "AI" for l in y.tolist()]))
        print("    Domains: {}".format(domains))
        sys.stdout.flush()
    except Exception as e:
        record("Spatial DataLoader", FAIL, traceback.format_exc())

    section("9b. Dataset Load + One Batch (Frequency)")
    try:
        ds_f = ds_mod.V4Dataset("test", mode="frequency", augment=False)
        loader_f = DataLoader(ds_f, batch_size=4, shuffle=False,
                             collate_fn=ds_mod.collate_spatial)
        batch_f = next(iter(loader_f))
        xf, yf, domains_f = batch_f
        record("Frequency batch xf shape (4,3,224,224)",
               PASS if tuple(xf.shape)==(4,3,224,224) else FAIL, str(tuple(xf.shape)))
        record("Frequency batch finite",
               PASS if torch.isfinite(xf).all().item() else FAIL)
        print("    FFT range: [{:.4f}, {:.4f}]".format(xf.min().item(), xf.max().item()))
        sys.stdout.flush()
    except Exception as e:
        record("Frequency DataLoader", FAIL, traceback.format_exc())

    section("9c. Dataset Load + One Batch (Hybrid)")
    try:
        ds_h = ds_mod.V4Dataset("test", mode="hybrid", augment=False)
        loader_h = DataLoader(ds_h, batch_size=4, shuffle=False,
                             collate_fn=ds_mod.collate_hybrid)
        batch_h = next(iter(loader_h))
        xs, xf, yh, domains_h = batch_h
        record("Hybrid batch xs shape (4,3,224,224)",
               PASS if tuple(xs.shape)==(4,3,224,224) else FAIL, str(tuple(xs.shape)))
        record("Hybrid batch xf shape (4,3,224,224)",
               PASS if tuple(xf.shape)==(4,3,224,224) else FAIL, str(tuple(xf.shape)))
        record("Hybrid spatial != FFT input",
               PASS if not torch.allclose(xs, xf, atol=0.01) else FAIL)
        sys.stdout.flush()
    except Exception as e:
        record("Hybrid DataLoader", FAIL, traceback.format_exc())

    section("9d. Forward Passes (from SAVED CHECKPOINTS)")

    def load_checkpoint_for_test(model_obj, fname):
        p = MODEL_DIR / fname
        ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        cleaned = {(k[7:] if k.startswith("module.") else k): v for k,v in state.items()}
        model_obj.load_state_dict(cleaned, strict=True)
        model_obj.to(DEVICE)
        model_obj.eval()
        return model_obj

    # Spatial V4
    try:
        m_s = load_checkpoint_for_test(mdl_mod.SpatialV4(), "spatial_resnet50_v4.pth")
        xi = x.to(DEVICE)
        with torch.no_grad():
            out = m_s(xi)
        probs = torch.softmax(out, 1)
        record("SpatialV4 (checkpoint) output shape (4,2)",
               PASS if tuple(out.shape)==(4,2) else FAIL, str(tuple(out.shape)))
        record("SpatialV4 on CUDA", PASS if out.device.type=="cuda" else FAIL,
               "device={}".format(out.device))
        print("    Predictions: {} Confidences: {}".format(
            ["REAL" if p==0 else "AI" for p in out.argmax(1).tolist()],
            ["{:.2f}%".format(float(probs[i, out.argmax(1)[i]])*100) for i in range(4)]))
        del m_s
        torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
        sys.stdout.flush()
    except Exception as e:
        record("SpatialV4 checkpoint forward", FAIL, traceback.format_exc())

    # Frequency V4
    try:
        m_f = load_checkpoint_for_test(mdl_mod.FrequencyV4(), "frequency_resnet50_v4.pth")
        xfi = xf.to(DEVICE)
        with torch.no_grad():
            out = m_f(xfi)
        record("FrequencyV4 (checkpoint) output shape (4,2)",
               PASS if tuple(out.shape)==(4,2) else FAIL, str(tuple(out.shape)))
        del m_f
        torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
        sys.stdout.flush()
    except Exception as e:
        record("FrequencyV4 checkpoint forward", FAIL, traceback.format_exc())

    # Hybrid V4
    try:
        m_h = load_checkpoint_for_test(mdl_mod.HybridV4(), "hybrid_resnet50_fft_v4.pth")
        xsi = xs.to(DEVICE)
        xfi2 = xf.to(DEVICE)
        with torch.no_grad():
            out = m_h(xsi, xfi2)
        record("HybridV4 (checkpoint) output shape (4,2)",
               PASS if tuple(out.shape)==(4,2) else FAIL, str(tuple(out.shape)))
        del m_h
        torch.cuda.empty_cache() if DEVICE.type=="cuda" else None
        sys.stdout.flush()
    except Exception as e:
        record("HybridV4 checkpoint forward", FAIL, traceback.format_exc())

# ===========================================================================
# 10. FINAL REPORT
# ===========================================================================

banner("10. FINAL INTEGRITY REPORT")

# Group results
groups = {
    "Dataset pipeline":        [k for k in results if any(x in k for x in
                                ["CLASS_MAPPING","IDX_TO_CLASS","METADATA","train total",
                                 "Val ","Test ","Train:","Val:","PREPROCESSING_DOC"])],
    "FFT pipeline":            [k for k in results if "FFT" in k],
    "Spatial architecture":    [k for k in results if "SpatialV4" in k and "checkpoint" not in k],
    "Frequency architecture":  [k for k in results if "FrequencyV4" in k and "checkpoint" not in k],
    "Hybrid architecture":     [k for k in results if "HybridV4" in k and "checkpoint" not in k],
    "Training configuration":  [k for k in results if "Train config" in k],
    "Evaluation logic":        [k for k in results if "Eval:" in k],
    "Checkpoint loading":      [k for k in results if any(x in k for x in
                                ["_v4.pth","checkpoint epoch","checkpoint F1"])],
    "V3 preservation":         [k for k in results if "V3 " in k],
    "Smoke test":              [k for k in results if any(x in k for x in
                                ["batch","DataLoader","forward","CUDA device"])],
}

all_pass = True
group_results = {}
for group, keys in groups.items():
    if not keys:
        continue
    group_pass = all(results[k] for k in keys if k in results)
    group_results[group] = group_pass
    status = PASS if group_pass else FAIL
    print("  {} {}".format(status, group))
    if not group_pass:
        all_pass = False
        for k in keys:
            if k in results and not results[k]:
                print("      FAILED: {}".format(k))

print()
print("=" * 70)
if all_pass:
    print("  V4 SOURCE INTEGRITY: VERIFIED")
else:
    print("  V4 SOURCE INTEGRITY: NOT FULLY VERIFIED (see failures above)")
print("=" * 70)

n_pass = sum(1 for v in results.values() if v)
n_fail = sum(1 for v in results.values() if not v)
print("  Total checks: {}  |  PASS: {}  |  FAIL: {}".format(
    len(results), n_pass, n_fail))

if n_fail > 0:
    print("\n  Failed checks:")
    for k, v in results.items():
        if not v:
            print("    [FAIL] {}".format(k))

print()
print("  NOTE: DO NOT proceed to AI-MANIPULATED detection.")
print("  STOP here and wait for next instruction.")
print()
sys.stdout.flush()
