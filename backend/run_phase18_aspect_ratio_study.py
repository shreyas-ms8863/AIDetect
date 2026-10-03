"""
AIDetect Phase 18: Aspect-Ratio Invariance & Preprocessing Diagnostic Study
==========================================================================
Investigates the sensitivity of the frozen Phase 7 manipulation frequency detector
to aspect-ratio transformations, cropping, padding, and resizing.

CRITICAL SAFETY:
- Completely diagnostic.
- No model weights, production code, calibration, or training data are modified.
"""

import sys
import os
import io
import csv
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
RESULTS_DIR = BACKEND_DIR / "phase18_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BACKEND_DIR))
from manipulation_freq_v1_models import ManipulationFrequencyResNet50V1
from manipulation_freq_v1_dataset import AuthoritativeFrequencyTransform

SUSPECT_IMAGE_PATH = Path(r"C:\Users\Shreyas\Downloads\WhatsApp Image 2026-10-02 at 1.52.03 PM.jpeg")
MANIP_CKPT_PATH    = PROJECT_DIR / "models" / "manipulation_frequency_resnet50_v1.pth"
GEN_CKPT_PATH      = PROJECT_DIR / "models" / "frequency_resnet50_v4.pth"
CALIB_PATH         = BACKEND_DIR / "phase12_results" / "strategy_e_calibration.json"

EXPECTED_MANIP_SHA256 = "8bd929125f379c3abdb71ca2249b2252482122973b4225b80a9ab65c68c86889"
EXPECTED_GEN_SHA256   = "2f106288c53ac8728f581f87e49848fc616cb7c30db52a42dec1342cc5d74dbf"
EXPECTED_CALIB_SHA256 = "30e1da6e2ca709cc6113d0d6726db01aef8a127e627869f195449da52571b1ec"


def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def pad_image_square(img: Image.Image, mode: str = "reflect") -> Image.Image:
    """Pad image to square (1280x1280) using reflect, edge, or constant mode."""
    arr = np.array(img)  # [H, W, C]
    h, w, c = arr.shape
    max_dim = max(h, w)
    
    pad_h = max_dim - h
    pad_w = max_dim - w
    
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    pad_width = ((top, bottom), (left, right), (0, 0))
    
    if mode == "reflect":
        padded = np.pad(arr, pad_width, mode="reflect")
    elif mode == "edge":
        padded = np.pad(arr, pad_width, mode="edge")
    elif mode == "constant":
        padded = np.pad(arr, pad_width, mode="constant", constant_values=0)
    else:
        raise ValueError(f"Unknown padding mode: {mode}")
        
    return Image.fromarray(padded)


def evaluate_manipulation(
    model: torch.nn.Module,
    xform: AuthoritativeFrequencyTransform,
    img: Image.Image,
    device: torch.device
) -> Tuple[float, float, str]:
    """Run frozen manipulation detector on an image and return (p_orig, p_manip, label)."""
    model.eval()
    with torch.no_grad():
        tensor = xform(img).unsqueeze(0).to(device)
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        p_orig = float(probs[0])
        p_manip = float(probs[1])
        label = "AI_MANIPULATED" if p_manip > 0.50 else "ORIGINAL_REAL"
    return p_orig, p_manip, label


def main():
    print("=" * 80)
    print("AIDETECT PHASE 18: ASPECT-RATIO INVARIANCE & PREPROCESSING DIAGNOSTIC STUDY")
    print("=" * 80)

    # 1. VERIFY CHECKPOINT HASHES
    print("\n--- STEP 1 & 2: VERIFY CHECKPOINT HASHES ---")
    manip_sha = compute_sha256(MANIP_CKPT_PATH)
    gen_sha   = compute_sha256(GEN_CKPT_PATH)
    calib_sha = compute_sha256(CALIB_PATH)

    print(f"Manipulation Model Checkpoint: {MANIP_CKPT_PATH.name}")
    print(f"  Hash:     {manip_sha}")
    print(f"  Expected: {EXPECTED_MANIP_SHA256}")
    if manip_sha != EXPECTED_MANIP_SHA256:
        print("[FATAL] Manipulation checkpoint hash mismatch! STOPPING.")
        sys.exit(1)
    print("  Status:   MATCH [OK]")

    print(f"Generation Model Checkpoint:   {GEN_CKPT_PATH.name}")
    print(f"  Hash:     {gen_sha}")
    print(f"  Status:   MATCH [OK]" if gen_sha == EXPECTED_GEN_SHA256 else "MISMATCH")

    print(f"Strategy E Calibration:        {CALIB_PATH.name}")
    print(f"  Hash:     {calib_sha}")
    print(f"  Status:   MATCH [OK]" if calib_sha == EXPECTED_CALIB_SHA256 else "MISMATCH")

    # 2. LOAD FROZEN MANIPULATION MODEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading frozen manipulation detector on {device}...")
    model = ManipulationFrequencyResNet50V1(dropout_p=0.3)
    ckpt = torch.load(MANIP_CKPT_PATH, map_location=device, weights_only=False)
    sd = ckpt.get("model_state_dict", ckpt)
    cleaned_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    model.load_state_dict(cleaned_sd, strict=True)
    model.to(device).eval()
    print("[OK] Model loaded with strict state_dict matching.")

    xform = AuthoritativeFrequencyTransform()

    # 3. LOAD SUSPECT IMAGE
    if not SUSPECT_IMAGE_PATH.exists():
        print(f"[FATAL] Suspect image not found at {SUSPECT_IMAGE_PATH}")
        sys.exit(1)

    raw_img = Image.open(SUSPECT_IMAGE_PATH).convert("RGB")
    orig_w, orig_h = raw_img.size
    print(f"\nSuspect Image Loaded: {SUSPECT_IMAGE_PATH.name}")
    print(f"Dimensions: {orig_w} x {orig_h} (Aspect Ratio: {orig_w/orig_h:.4f})")

    # 4. BASELINE / CONTROL (Experiment A)
    print("\n" + "=" * 60)
    print("EXPERIMENT A: PRODUCTION CONTROL (NATIVE 587x1280)")
    print("=" * 60)
    b_orig, b_manip, b_label = evaluate_manipulation(model, xform, raw_img, device)
    print(f"Input:         {orig_w} x {orig_h}")
    print(f"Output:        {orig_w} x {orig_h}")
    print(f"P(ORIGINAL):   {b_orig*100:.2f}% ({b_orig:.6f})")
    print(f"P(MANIP):      {b_manip*100:.2f}% ({b_manip:.6f})")
    print(f"Prediction:    {b_label}")
    baseline_p_manip = b_manip

    if abs(b_orig - 0.9993) > 0.01:
        print(f"[WARNING] Baseline differs from expected ~99.93%! Got {b_orig*100:.2f}%")

    # List of all experimental results
    experiments: List[Dict[str, Any]] = []

    def record_exp(exp_name: str, xform_name: str, in_w: int, in_h: int, out_w: int, out_h: int,
                   p_o: float, p_m: float, pred: str, notes: str):
        delta = p_m - baseline_p_manip
        delta_pct = delta * 100
        print("-" * 55)
        print(f"EXPERIMENT:       {exp_name} ({xform_name})")
        print(f"Input:            {in_w} x {in_h}")
        print(f"Output:           {out_w} x {out_h}")
        print(f"P(ORIGINAL):      {p_o*100:.2f}% ({p_o:.6f})")
        print(f"P(MANIPULATED):   {p_m*100:.2f}% ({p_m:.6f})")
        print(f"Prediction:       {pred}")
        print(f"Delta from control: {delta_pct:+.2f}% ({delta:+.6f})")
        print(f"Notes:            {notes}")
        experiments.append({
            "experiment": exp_name,
            "transformation": xform_name,
            "input_width": in_w,
            "input_height": in_h,
            "output_width": out_w,
            "output_height": out_h,
            "P_original": round(p_o * 100, 2),
            "P_manipulated": round(p_m * 100, 2),
            "delta_manipulated": round(delta_pct, 2),
            "prediction": pred,
            "notes": notes,
        })

    record_exp("CONTROL", "native", orig_w, orig_h, orig_w, orig_h, b_orig, b_manip, b_label, "Baseline production native FFT")

    # 5. EXPERIMENT B — CENTER CROP (587x587)
    # y coordinates: center vertically in 1280
    top_cc = (orig_h - orig_w) // 2  # (1280 - 587) // 2 = 346
    bottom_cc = top_cc + orig_w       # 346 + 587 = 933
    img_center_crop = raw_img.crop((0, top_cc, orig_w, bottom_cc))
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_center_crop, device)
    record_exp("CENTER_CROP", "center_587", orig_w, orig_h, 587, 587, p_o, p_m, pred, f"Crop box: (0, {top_cc}, {orig_w}, {bottom_cc})")

    # 6. EXPERIMENT C / G — SQUARE PADDING (1280x1280)
    # A. Reflect Padding
    img_pad_reflect = pad_image_square(raw_img, mode="reflect")
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_pad_reflect, device)
    record_exp("SQUARE_PADDING", "pad_1280_reflect", orig_w, orig_h, 1280, 1280, p_o, p_m, pred, "Reflect pad width: left 346, right 347")

    # B. Edge/Replicate Padding
    img_pad_edge = pad_image_square(raw_img, mode="edge")
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_pad_edge, device)
    record_exp("SQUARE_PADDING", "pad_1280_edge", orig_w, orig_h, 1280, 1280, p_o, p_m, pred, "Edge clamp pad width: left 346, right 347")

    # C. Constant/Zero Padding
    img_pad_const = pad_image_square(raw_img, mode="constant")
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_pad_const, device)
    record_exp("SQUARE_PADDING", "pad_1280_constant0", orig_w, orig_h, 1280, 1280, p_o, p_m, pred, "Zero/black pad width: left 346, right 347")

    # 7. EXPERIMENT D — SQUARE RESIZE TO 1024x1024
    img_resize_1024 = raw_img.resize((1024, 1024), Image.Resampling.BILINEAR)
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_resize_1024, device)
    record_exp("SQUARE_RESIZE", "resize_1024", orig_w, orig_h, 1024, 1024, p_o, p_m, pred, "Direct bilinear stretch to 1024x1024")

    # 8. EXPERIMENT E — SQUARE RESIZE TO 512x512
    img_resize_512 = raw_img.resize((512, 512), Image.Resampling.BILINEAR)
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_resize_512, device)
    record_exp("SQUARE_RESIZE", "resize_512", orig_w, orig_h, 512, 512, p_o, p_m, pred, "Direct bilinear stretch to 512x512")

    # 9. EXPERIMENT F — SQUARE CROPS AT MULTIPLE SCALES
    # Scale 1: 512x512 center
    top_512 = (orig_h - 512) // 2
    left_512 = (orig_w - 512) // 2
    img_center_512 = raw_img.crop((left_512, top_512, left_512 + 512, top_512 + 512))
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_center_512, device)
    record_exp("CROP_SCALE", "center_512", orig_w, orig_h, 512, 512, p_o, p_m, pred, f"Center crop box: ({left_512}, {top_512}, {left_512+512}, {top_512+512})")

    # Scale 2: 384x384 center
    top_384 = (orig_h - 384) // 2
    left_384 = (orig_w - 384) // 2
    img_center_384 = raw_img.crop((left_384, top_384, left_384 + 384, top_384 + 384))
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_center_384, device)
    record_exp("CROP_SCALE", "center_384", orig_w, orig_h, 384, 384, p_o, p_m, pred, f"Center crop box: ({left_384}, {top_384}, {left_384+384}, {top_384+384})")

    # 10. EXPERIMENT I — DIFFERENT VERTICAL POSITION CROPS (587x587)
    # Top Crop: (0, 0, 587, 587)
    img_crop_top = raw_img.crop((0, 0, 587, 587))
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_crop_top, device)
    record_exp("POSITION_CROP", "crop_587_top", orig_w, orig_h, 587, 587, p_o, p_m, pred, "Crop box: (0, 0, 587, 587) - Head/Torso")

    # Center Crop: (0, 346, 587, 933) - already recorded as CENTER_CROP, but let's record in POSITION_CROP for clean comparison
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_center_crop, device)
    record_exp("POSITION_CROP", "crop_587_center", orig_w, orig_h, 587, 587, p_o, p_m, pred, "Crop box: (0, 346, 587, 933) - Midsection")

    # Bottom Crop: (0, 1280-587, 587, 1280) = (0, 693, 587, 1280)
    top_bottom = orig_h - 587
    img_crop_bottom = raw_img.crop((0, top_bottom, 587, orig_h))
    p_o, p_m, pred = evaluate_manipulation(model, xform, img_crop_bottom, device)
    record_exp("POSITION_CROP", "crop_587_bottom", orig_w, orig_h, 587, 587, p_o, p_m, pred, f"Crop box: (0, {top_bottom}, 587, {orig_h}) - Lower body")

    # Find the representation with largest delta so far
    # (excluding control)
    best_candidate = max(experiments[1:], key=lambda x: x["delta_manipulated"])
    best_xform_name = best_candidate["transformation"]
    print(f"\n[INFO] Representation producing largest diagnostic delta: {best_xform_name} (Delta = {best_candidate['delta_manipulated']:+.2f}%)")

    # Function to get transformed image by name
    def get_transformed_image(name: str) -> Image.Image:
        if name == "center_587" or name == "crop_587_center":
            return img_center_crop
        elif name == "pad_1280_reflect":
            return img_pad_reflect
        elif name == "pad_1280_edge":
            return img_pad_edge
        elif name == "pad_1280_constant0":
            return img_pad_const
        elif name == "resize_1024":
            return img_resize_1024
        elif name == "resize_512":
            return img_resize_512
        elif name == "center_512":
            return img_center_512
        elif name == "center_384":
            return img_center_384
        elif name == "crop_587_top":
            return img_crop_top
        elif name == "crop_587_bottom":
            return img_crop_bottom
        return raw_img

    best_img = get_transformed_image(best_xform_name)

    # 11. EXPERIMENT H — JPEG CONTROL (Native vs Best Aspect-Ratio Representation)
    print("\n" + "=" * 60)
    print(f"EXPERIMENT H: JPEG CONTROL (Native vs {best_xform_name})")
    print("=" * 60)

    for q in [95, 75, 50]:
        # Native with JPEG q
        buf_nat = io.BytesIO()
        raw_img.save(buf_nat, format="JPEG", quality=q)
        buf_nat.seek(0)
        img_nat_q = Image.open(buf_nat).copy()
        p_o, p_m, pred = evaluate_manipulation(model, xform, img_nat_q, device)
        record_exp("JPEG_CONTROL", f"native_jpeg_{q}", orig_w, orig_h, orig_w, orig_h, p_o, p_m, pred, f"Native resolution JPEG compression Q={q}")

        # Best representation with JPEG q
        buf_best = io.BytesIO()
        best_img.save(buf_best, format="JPEG", quality=q)
        buf_best.seek(0)
        img_best_q = Image.open(buf_best).copy()
        bw, bh = best_img.size
        p_o, p_m, pred = evaluate_manipulation(model, xform, img_best_q, device)
        record_exp("JPEG_CONTROL", f"{best_xform_name}_jpeg_{q}", orig_w, orig_h, bw, bh, p_o, p_m, pred, f"{best_xform_name} JPEG compression Q={q}")

    # 12. EXPERIMENT 17: COMPARE WITH IN-DISTRIBUTION manipulation_v1 SAMPLES
    print("\n" + "=" * 60)
    print("STEP 17: IN-DISTRIBUTION COMPARISON ON manipulation_v1 TEST SAMPLES")
    print("=" * 60)

    test_manip_dir = PROJECT_DIR / "manipulation_v1" / "test" / "manipulated"
    categories = ["inpainting", "face_modification", "object_insertion", "object_replacement", "object_removal"]
    in_dist_results = []

    for cat in categories:
        cat_dir = test_manip_dir / cat
        img_files = sorted(list(cat_dir.glob("*.png")) + list(cat_dir.glob("*.jpg")))
        if not img_files:
            continue
        sample_path = img_files[0]
        with Image.open(sample_path) as s_img:
            s_rgb = s_img.convert("RGB")
            sw, sh = s_rgb.size

            # 1. Native control
            p_o_nat, p_m_nat, pred_nat = evaluate_manipulation(model, xform, s_rgb, device)

            # 2. Simulated non-square crop (take a vertical 587x1024 or 469x1024 crop to simulate portrait aspect ratio)
            # 1024 x 1024 -> crop to 470 x 1024 (aspect ratio 0.4589 matching suspect image)
            sim_w = int(sh * (587 / 1280))  # approx 470
            sim_left = (sw - sim_w) // 2
            sim_portrait = s_rgb.crop((sim_left, 0, sim_left + sim_w, sh))
            p_o_port, p_m_port, pred_port = evaluate_manipulation(model, xform, sim_portrait, device)

            # 3. Center crop to 587x587 or 512x512
            cc_512 = s_rgb.crop((256, 256, 768, 768))
            p_o_cc, p_m_cc, pred_cc = evaluate_manipulation(model, xform, cc_512, device)

            row = {
                "category": cat,
                "sample_image": sample_path.name,
                "original_dimensions": [sw, sh],
                "native_p_manip": round(p_m_nat * 100, 2),
                "native_pred": pred_nat,
                "simulated_portrait_p_manip": round(p_m_port * 100, 2),
                "simulated_portrait_pred": pred_port,
                "center_crop_512_p_manip": round(p_m_cc * 100, 2),
                "center_crop_pred": pred_cc,
            }
            in_dist_results.append(row)
            print(f"Category: {cat:<22} | Native P(Manip): {row['native_p_manip']:6.2f}% | Sim. Portrait P(Manip): {row['simulated_portrait_p_manip']:6.2f}% | 512 Crop: {row['center_crop_512_p_manip']:6.2f}%")

    # 13. SAVE RESULTS TO CSV & JSON
    csv_file = RESULTS_DIR / "phase18_aspect_ratio_results.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "experiment", "transformation", "input_width", "input_height",
            "output_width", "output_height", "P_original", "P_manipulated",
            "delta_manipulated", "prediction", "notes"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in experiments:
            writer.writerow(row)
    print(f"\n[OK] CSV results saved to: {csv_file}")

    json_file = RESULTS_DIR / "phase18_aspect_ratio_results.json"
    full_output = {
        "metadata": {
            "suspect_image": str(SUSPECT_IMAGE_PATH),
            "original_dimensions": [orig_w, orig_h],
            "aspect_ratio": orig_w / orig_h,
            "manipulation_checkpoint": str(MANIP_CKPT_PATH),
            "checkpoint_sha256": manip_sha,
            "baseline_P_manipulated": round(baseline_p_manip * 100, 2),
        },
        "experiments": experiments,
        "in_distribution_comparison": in_dist_results,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"[OK] JSON results saved to: {json_file}")

    # 14. GENERATE MATPLOTLIB COMPARISON CHART
    plot_file = RESULTS_DIR / "aspect_ratio_probability_comparison.png"
    
    # Filter key conditions for clean plot
    key_conditions = [
        ("Control (Native 587x1280)", "CONTROL", "native"),
        ("Center Crop (587x587)", "CENTER_CROP", "center_587"),
        ("Square Pad Reflect (1280x1280)", "SQUARE_PADDING", "pad_1280_reflect"),
        ("Square Pad Edge (1280x1280)", "SQUARE_PADDING", "pad_1280_edge"),
        ("Square Pad Const (1280x1280)", "SQUARE_PADDING", "pad_1280_constant0"),
        ("Resize 1024x1024", "SQUARE_RESIZE", "resize_1024"),
        ("Resize 512x512", "SQUARE_RESIZE", "resize_512"),
        ("Top Crop (587x587)", "POSITION_CROP", "crop_587_top"),
        ("Center Crop (587x587)", "POSITION_CROP", "crop_587_center"),
        ("Bottom Crop (587x587)", "POSITION_CROP", "crop_587_bottom"),
    ]
    
    labels = []
    p_manips = []
    colors = []
    
    for lbl, exp_name, xform_name in key_conditions:
        match = next((e for e in experiments if e["experiment"] == exp_name and e["transformation"] == xform_name), None)
        if match:
            labels.append(lbl)
            val = match["P_manipulated"]
            p_manips.append(val)
            colors.append("#dc2626" if val >= 50.0 else "#2563eb")
            
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(labels)), p_manips, color=colors, height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("P(AI_MANIPULATED) %", fontsize=11, fontweight="bold")
    ax.set_title("AIDetect Phase 18: Aspect-Ratio & Spatial Preprocessing Sensitivity", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.axvline(50.0, color="#ef4444", linestyle="--", linewidth=1.5, label="Decision Boundary (50%)")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    
    # Add data labels
    for bar, val in zip(bars, p_manips):
        ax.text(val + 1.0, bar.get_y() + bar.get_height() / 2, f"{val:.2f}%", va="center", fontsize=9, fontweight="bold")
        
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"[OK] Comparison plot saved to: {plot_file}")


if __name__ == "__main__":
    main()
