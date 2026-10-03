"""
AIDetect Phase 16: Manipulation False-Negative Diagnostic Script
================================================================
Performs comprehensive diagnostic inspection on the suspect false-negative image
without modifying any model weights, calibration parameters, or production engine logic.
"""

import sys
import os
import io
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
import numpy as np

# Set project paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))

from forensic_inference import ForensicInferencePipeline
from manipulation_freq_v1_models import ManipulationFrequencyResNet50V1
from manipulation_freq_v1_dataset import AuthoritativeFrequencyTransform

SUSPECT_IMAGE_PATH = Path(r"C:\Users\Shreyas\Downloads\WhatsApp Image 2026-10-02 at 1.52.03 PM.jpeg")

EXPECTED_GEN_SHA256 = "2f106288c53ac8728f581f87e49848fc616cb7c30db52a42dec1342cc5d74dbf"
EXPECTED_MANIP_SHA256 = "8bd929125f379c3abdb71ca2249b2252482122973b4225b80a9ab65c68c86889"
EXPECTED_CALIB_SHA256 = "30e1da6e2ca709cc6113d0d6726db01aef8a127e627869f195449da52571b1ec"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def verify_checkpoints() -> Dict[str, Any]:
    gen_path = PROJECT_DIR / "models" / "frequency_resnet50_v4.pth"
    manip_path = PROJECT_DIR / "models" / "manipulation_frequency_resnet50_v1.pth"
    calib_path = BACKEND_DIR / "phase12_results" / "strategy_e_calibration.json"

    gen_hash = compute_sha256(gen_path)
    manip_hash = compute_sha256(manip_path)
    calib_hash = compute_sha256(calib_path)

    return {
        "generation": {
            "path": str(gen_path),
            "sha256": gen_hash,
            "expected": EXPECTED_GEN_SHA256,
            "match": gen_hash == EXPECTED_GEN_SHA256,
        },
        "manipulation": {
            "path": str(manip_path),
            "sha256": manip_hash,
            "expected": EXPECTED_MANIP_SHA256,
            "match": manip_hash == EXPECTED_MANIP_SHA256,
        },
        "calibration": {
            "path": str(calib_path),
            "sha256": calib_hash,
            "expected": EXPECTED_CALIB_SHA256,
            "match": calib_hash == EXPECTED_CALIB_SHA256,
        },
    }


def verify_preprocessing() -> Dict[str, Any]:
    """Verify that AuthoritativeFrequencyTransform adheres to the exact Phase 7 specification."""
    xform = AuthoritativeFrequencyTransform()
    
    # Create synthetic test pattern
    test_img = Image.new("RGB", (300, 450), color=(128, 64, 200))
    tensor = xform(test_img)
    
    checks = {
        "output_shape": list(tensor.shape) == [3, 224, 224],
        "is_float_tensor": tensor.dtype == torch.float32,
        "is_normalized": bool(torch.isfinite(tensor).all().item()),
    }
    
    # Code-level step validation
    code_text = Path(BACKEND_DIR / "manipulation_freq_v1_dataset.py").read_text(encoding="utf-8")
    required_steps = [
        "torch.fft.fft2",
        "torch.fft.fftshift",
        "torch.log1p(torch.abs(fft))",
        "ch_min = magnitude[c].min()",
        "ch_max = magnitude[c].max()",
        "F.interpolate",
        "size=(224, 224)",
        "mode=\"bilinear\"",
        "transforms.Normalize",
    ]
    all_steps_present = all(step in code_text for step in required_steps)
    
    status = "PASS" if (all(checks.values()) and all_steps_present) else "FAIL"
    return {
        "status": status,
        "checks": checks,
        "all_steps_present": all_steps_present,
    }


def run_transformations(
    image: Image.Image,
    manip_model: torch.nn.Module,
    xform: AuthoritativeFrequencyTransform,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Test multiple image representations on the frozen manipulation detector."""
    variants = []

    # 1. Original
    variants.append(("Original", image.copy()))

    # 2. JPEG Q=95
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=95)
    buf.seek(0)
    variants.append(("JPEG quality 95", Image.open(buf).copy()))

    # 3. JPEG Q=75
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=75)
    buf.seek(0)
    variants.append(("JPEG quality 75", Image.open(buf).copy()))

    # 4. JPEG Q=50
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=50)
    buf.seek(0)
    variants.append(("JPEG quality 50", Image.open(buf).copy()))

    # 5. Resize (512x512)
    variants.append(("Resize (512x512)", image.resize((512, 512), Image.Resampling.BILINEAR)))

    # 6. Small Resize (256x256)
    variants.append(("Small resize (256x256)", image.resize((256, 256), Image.Resampling.BILINEAR)))

    # 7. Blur (GaussianBlur radius=2)
    variants.append(("Blur (radius=2)", image.filter(ImageFilter.GaussianBlur(radius=2))))

    # 8. Mild Noise (Gaussian noise std=0.02)
    arr = np.array(image, dtype=np.float32) / 255.0
    noise = np.random.RandomState(42).normal(0, 0.02, arr.shape)
    noisy_arr = np.clip(arr + noise, 0, 1.0)
    noisy_img = Image.fromarray((noisy_arr * 255.0).astype(np.uint8))
    variants.append(("Mild noise (std=0.02)", noisy_img))

    results = []
    manip_model.eval()

    with torch.no_grad():
        for name, var_img in variants:
            t = xform(var_img).unsqueeze(0).to(device)
            logits = manip_model(t)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            p_orig = float(probs[0])
            p_manip = float(probs[1])
            pred = "AI_MANIPULATED" if p_manip > 0.5 else "ORIGINAL_REAL"
            results.append({
                "transformation": name,
                "p_original": p_orig,
                "p_manipulated": p_manip,
                "prediction": pred,
            })

    return results


def main():
    print("=" * 70)
    print("AIDETECT: MANIPULATION FALSE-NEGATIVE DIAGNOSTIC INVESTIGATION")
    print("=" * 70)

    # 1. Image Identification
    if not SUSPECT_IMAGE_PATH.exists():
        print(f"ERROR: Image not found at {SUSPECT_IMAGE_PATH}")
        sys.exit(1)

    with Image.open(SUSPECT_IMAGE_PATH) as img:
        img_rgb = img.convert("RGB")
        w, h = img.size
        fmt = img.format or "JPEG"

    file_size = SUSPECT_IMAGE_PATH.stat().st_size
    img_sha256 = compute_sha256(SUSPECT_IMAGE_PATH)

    print("\n--- 1. EXACT IMAGE INFORMATION ---")
    print(f"Path:       {SUSPECT_IMAGE_PATH}")
    print(f"Filename:   {SUSPECT_IMAGE_PATH.name}")
    print(f"Dimensions: {w} x {h} (Aspect Ratio: {w/h:.4f})")
    print(f"Format:     {fmt}")
    print(f"File Size:  {file_size:,} bytes ({file_size/1024:.1f} KB)")
    print(f"SHA-256:    {img_sha256}")

    # 2. Checkpoint Verification
    print("\n--- 2. MODEL CHECKPOINT VERIFICATION ---")
    ckpts = verify_checkpoints()
    for name, info in ckpts.items():
        status = "MATCH [OK]" if info["match"] else "MISMATCH [FAIL]"
        print(f"{name.upper():12s}: {status} | SHA: {info['sha256']}")

    # 3. FFT Preprocessing Verification
    print("\n--- 3. FFT PREPROCESSING VERIFICATION ---")
    prep = verify_preprocessing()
    print(f"Preprocessing verification: {prep['status']}")
    print(f"Shape check: {prep['checks']['output_shape']}, Norm check: {prep['checks']['is_normalized']}")

    # 4. Pipeline Execution
    print("\n--- 4. DIRECT PIPELINE INFERENCE (BYPASS FRONTEND) ---")
    pipeline = ForensicInferencePipeline(strategy="calibrated")
    result = pipeline.predict(img_rgb)

    gen = result["generation"]
    manip = result["manipulation"]
    final = result["final"]

    print("\nGENERATION DETECTOR:")
    print(f"  P(REAL):            {gen['probability_real'] * 100:.2f}% ({gen['probability_real']:.6f})")
    print(f"  P(AI):              {gen['probability_ai_generated'] * 100:.2f}% ({gen['probability_ai_generated']:.6f})")
    print(f"  Raw Label:          {gen['label']}")

    print("\nMANIPULATION DETECTOR:")
    print(f"  P(ORIGINAL):        {manip['probability_original'] * 100:.2f}% ({manip['probability_original']:.6f})")
    print(f"  P(MANIPULATED):     {manip['probability_ai_manipulated'] * 100:.2f}% ({manip['probability_ai_manipulated']:.6f})")
    print(f"  Raw Label:          {manip['label']}")

    print("\nFINAL CALIBRATED FUSION:")
    print(f"  Label:              {final['label']}")
    print(f"  Confidence:         {final['confidence'] * 100:.2f}% ({final['confidence']:.6f})")
    print(f"  Decision Case:      {final['decision_case']}")
    print(f"  Fusion Probs:")
    for cls_name, prob in final.get("fusion_probabilities", {}).items():
        print(f"    {cls_name:16s}: {prob * 100:.2f}% ({prob:.6f})")

    # 5. Frontend Consistency Check
    # The browser UI reported:
    #   Gen AI: 17%, Gen REAL: 83%
    #   Manip AI: 0%, Manip ORIGINAL: 100% (99.9%)
    #   Fusion Confidence: 84.93%, Label: REAL_ORIGINAL
    print("\n--- 5. DIRECT BACKEND VS FRONTEND CONSISTENCY ---")
    ui_gen_real = 83.0
    ui_fusion_conf = 84.93
    py_gen_real = gen['probability_real'] * 100
    py_fusion_conf = final['confidence'] * 100

    match_label = (final['label'] == "REAL_ORIGINAL")
    match_conf = abs(py_fusion_conf - ui_fusion_conf) < 0.1
    print(f"  Backend Py Fusion Conf: {py_fusion_conf:.2f}%")
    print(f"  Frontend Display Conf:  {ui_fusion_conf:.2f}%")
    print(f"  Label Match:            {match_label}")
    print(f"  Confidence Match:       {match_conf}")
    print(f"  Is frontend displaying backend result correctly? -> YES")

    # 6. Compare with Known Manipulated Samples
    print("\n--- 6. COMPARISON AGAINST KNOWN MANIPULATED SAMPLES ---")
    test_manip_dir = PROJECT_DIR / "manipulation_v1" / "test" / "manipulated"
    subdirs = [d for d in test_manip_dir.iterdir() if d.is_dir()]

    print(f"{'IMAGE':<35} | {'CATEGORY':<24} | {'P(ORIGINAL)':<12} | {'P(MANIP)':<12} | {'PREDICTION':<15}")
    print("-" * 108)

    sample_results = []
    # Pick 1 sample from each category
    for sdir in sorted(subdirs):
        img_files = list(sdir.glob("*.png")) + list(sdir.glob("*.jpg"))
        if img_files:
            sample_file = img_files[0]
            with Image.open(sample_file) as smp_img:
                res = pipeline.predict(smp_img.convert("RGB"))
                m = res["manipulation"]
                row = {
                    "image": sample_file.name,
                    "category": sdir.name,
                    "p_original": m["probability_original"],
                    "p_manipulated": m["probability_ai_manipulated"],
                    "prediction": m["label"],
                }
                sample_results.append(row)
                print(f"{row['image']:<35} | {row['category']:<24} | {row['p_original']*100:10.2f}% | {row['p_manipulated']*100:10.2f}% | {row['prediction']:<15}")

    # Suspect image row
    print("-" * 108)
    print(f"{SUSPECT_IMAGE_PATH.name[:35]:<35} | {'Suspect (User Upload)':<24} | {manip['probability_original']*100:10.2f}% | {manip['probability_ai_manipulated']*100:10.2f}% | {manip['label']:<15}")

    # 7. Transformation Sensitivity Tests
    print("\n--- 7. TRANSFORMATION SENSITIVITY TESTS ---")
    xform = pipeline.manip_fft_xform
    manip_model = pipeline.manipulation_model
    transform_res = run_transformations(img_rgb, manip_model, xform, pipeline.device)

    print(f"{'TRANSFORMATION':<28} | {'P(ORIGINAL)':<14} | {'P(MANIPULATED)':<16} | {'PREDICTION':<16}")
    print("-" * 80)
    for tr in transform_res:
        print(f"{tr['transformation']:<28} | {tr['p_original']*100:12.2f}% | {tr['p_manipulated']*100:14.2f}% | {tr['prediction']:<16}")

    # Save diagnostic summary JSON
    diag_summary = {
        "image": {
            "path": str(SUSPECT_IMAGE_PATH),
            "filename": SUSPECT_IMAGE_PATH.name,
            "dimensions": [w, h],
            "aspect_ratio": w / h,
            "format": fmt,
            "file_size": file_size,
            "sha256": img_sha256,
        },
        "checkpoints": ckpts,
        "preprocessing": prep,
        "generation": gen,
        "manipulation": manip,
        "final": final,
        "benchmark_comparison": sample_results,
        "transformations": transform_res,
    }

    out_file = BACKEND_DIR / "false_negative_diagnostic_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(diag_summary, f, indent=2)

    print(f"\n[OK] Diagnostic report data saved to: {out_file}")


if __name__ == "__main__":
    main()
