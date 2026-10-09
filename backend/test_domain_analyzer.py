"""
Test harness for domain analyzer across natural and document images.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from domain_analyzer import analyze_image_domain

test_images = [
    ("Known ChatGPT AI", Path(r"C:\Users\Shreyas\Downloads\ChatGPT Image Sep 30, 2026, 11_23_01 PM.png")),
    ("Authentic Selfie", Path(r"C:\Users\Shreyas\Downloads\WhatsApp Image 2026-10-01 at 11.28.52 PM.jpeg")),
    ("V4 Benchmark AI", BASE_DIR.parent / "dataset_v4" / "test" / "ai" / "genimage" / "v4_te_ai_genimage_00001.png"),
    ("V4 Benchmark REAL", BASE_DIR.parent / "dataset_v4" / "test" / "real" / "genimage" / "v4_te_real_genimage_00001.jpg"),
    ("Scan OOD", Path(r"C:\Users\Shreyas\Downloads\scan-1786806054318-1.png")),
    ("User Upload Doc", Path(r"C:\Users\Shreyas\.gemini\antigravity\brain\f0378695-c2ef-449b-a961-b38c55738691\.user_uploaded\media_1791048081287.jpg")),
    ("WhatsApp Doc 1", Path(r"C:\Users\Shreyas\Downloads\WhatsApp Image 2026-09-13 at 4.08.01 PM.jpeg")),
    ("WhatsApp Doc 2", Path(r"C:\Users\Shreyas\Downloads\WhatsApp Image 2026-09-01 at 7.51.44 PM.jpeg")),
    ("WhatsApp Doc 3", Path(r"C:\Users\Shreyas\Downloads\WhatsApp Image 2026-09-21 at 11.25.28 PM.jpeg")),
]

print(f"{'Image':22s} | {'Domain Type':16s} | {'Risk':6s} | {'Score':6s} | {'Text':5s} | {'Canvas':6s} | {'Lines':5s} | {'Mono':5s}")
print("-" * 88)
for label, p in test_images:
    if not p.exists():
        print(f"{label:22s} | MISSING FILE: {p}")
        continue
    res = analyze_image_domain(str(p))
    sig = res["domain_signals"]
    dtype = res["domain_type"]
    drisk = res["domain_risk"]
    dscore = res["domain_risk_score"]
    t_dens = sig["text_region_density"]
    bg_uni = sig["background_uniformity"]
    lines = sig["rectilinear_line_energy"]
    mono = sig["monochromatic_fraction"]
    print(f"{label:22s} | {dtype:16s} | {drisk:6s} | {dscore:6.4f} | {t_dens:5.3f} | {bg_uni:6.3f} | {lines:5.3f} | {mono:5.3f}")

# Also test 5 random real and 5 random AI images from dataset_v4
print("\n--- Additional V4 Natural Benchmark Tests ---")
for folder, name in [("real", "V4 Real Natural"), ("ai", "V4 AI Natural")]:
    flist = sorted((BASE_DIR.parent / "dataset_v4" / "test" / folder / "genimage").glob("*.*"))[:5]
    for p in flist:
        res = analyze_image_domain(str(p))
        print(f"{name} ({p.name}) -> {res['domain_type']} (Risk: {res['domain_risk']}, Score: {res['domain_risk_score']})")

