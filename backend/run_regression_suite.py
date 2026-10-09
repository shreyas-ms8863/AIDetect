"""
Phase 11 False-Negative Resolution: Authoritative 4-Image Regression Suite
Tests the 4 critical forensic images against the production ForensicInferencePipeline.
"""

import sys
import os
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from forensic_inference import ForensicInferencePipeline

TEST_CASES = [
    {
        "id": "CASE_1",
        "description": "Known AI (Suspect ChatGPT Generation)",
        "path": Path(r"C:\Users\Shreyas\Downloads\ChatGPT Image Sep 30, 2026, 11_23_01 PM.png"),
        "expected_verdict": "AI_GENERATED",
    },
    {
        "id": "CASE_2",
        "description": "Known Authentic Smartphone Capture (WhatsApp Selfie)",
        "path": Path(r"C:\Users\Shreyas\Downloads\WhatsApp Image 2026-10-01 at 11.28.52 PM.jpeg"),
        "expected_verdict": "REAL_ORIGINAL",
    },
    {
        "id": "CASE_3",
        "description": "V4 Benchmark AI (GenImage)",
        "path": PROJECT_DIR / "dataset_v4" / "test" / "ai" / "genimage" / "v4_te_ai_genimage_00001.png",
        "expected_verdict": "AI_GENERATED",
    },
    {
        "id": "CASE_4",
        "description": "V4 Benchmark REAL (GenImage)",
        "path": PROJECT_DIR / "dataset_v4" / "test" / "real" / "genimage" / "v4_te_real_genimage_00001.jpg",
        "expected_verdict": "REAL_ORIGINAL",
    },
    {
        "id": "CASE_5",
        "description": "Conflicting Document / Scan Evidence (Reliability-Aware Abstention)",
        "path": Path(r"C:\Users\Shreyas\Downloads\scan-1786806054318-1.png"),
        "expected_verdict": "UNCERTAIN",
    },
    {
        "id": "CASE_6",
        "description": "Genuine Document Image Delivered via Messaging (High Domain-Risk Safe Abstention)",
        "path": Path(r"C:\Users\Shreyas\Downloads\WhatsApp Image 2026-09-13 at 4.08.01 PM.jpeg"),
        "fallback_path": Path(r"C:\Users\Shreyas\.gemini\antigravity\brain\f0378695-c2ef-449b-a961-b38c55738691\.user_uploaded\media_1791048081287.jpg"),
        "expected_verdict": "UNCERTAIN",
    },
]

def run_regression():
    print("=" * 80)
    print("AIDETECT FORENSIC PIPELINE: AUTHORITATIVE REGRESSION SUITE")
    print("=" * 80)

    pipeline = ForensicInferencePipeline(strategy="calibrated")
    print("ForensicInferencePipeline initialized successfully with calibrated strategy.\n")

    all_passed = True

    for i, tc in enumerate(TEST_CASES, start=1):
        path = tc["path"]
        print("-" * 80)
        print(f"TEST CASE {i}: {tc['description']}")
        print(f"File: {path.name}")
        print(f"Path: {path}")

        if not path.exists():
            fallback = tc.get("fallback_path")
            if fallback and fallback.exists():
                path = fallback
                print(f"Using fallback path: {path}")
            else:
                print(f"ERROR: Image file not found at {path}")
                all_passed = False
                continue

        with Image.open(path) as img:
            img = img.convert("RGB")
            res = pipeline.predict(img)

        gen = res["generation"]
        manip = res["manipulation"]
        aux = res["auxiliary"]
        fin = res["final"]
        dom = res.get("domain", {})

        p_spatial_ai = aux["probability_spatial_ai"]
        p_freq_ai = aux["probability_freq_ai"]
        p_hybrid_ai = aux["probability_hybrid_ai"]
        p_manip = aux["probability_manipulated"]
        p_orig = aux["probability_original"]

        fusion_probs = fin.get("fusion_probabilities", {})

        print(f"\nDomain & Operating Range Analysis:")
        print(f"  Domain Type:                    {dom.get('domain_type', fin.get('domain_type', 'N/A'))}")
        print(f"  Domain Risk:                    {dom.get('domain_risk', fin.get('domain_risk', 'N/A'))}")
        print(f"  Domain Risk Score:              {dom.get('domain_risk_score', fin.get('domain_risk_score', 'N/A'))}")
        if "domain_signals" in dom:
            sig = dom["domain_signals"]
            print(f"  Paper Canvas Fraction:          {sig.get('paper_canvas_fraction', 'N/A')}")
            print(f"  Text Glyph Count:               {sig.get('text_glyph_count', 'N/A')}")
            print(f"  Rectilinear Line Energy:        {sig.get('rectilinear_line_energy', 'N/A')}")
            print(f"  Monochromatic Fraction:         {sig.get('monochromatic_fraction', 'N/A')}")

        print(f"\nMulti-Stream Generation & Tampering Probabilities:")
        print(f"  P_SPATIAL_AI: {p_spatial_ai:.4f} ({p_spatial_ai:.2%})")
        print(f"  P_FREQ_AI:    {p_freq_ai:.4f} ({p_freq_ai:.2%})")
        print(f"  P_HYBRID_AI:  {p_hybrid_ai:.4f} ({p_hybrid_ai:.2%})")
        print(f"  P_MANIPULATED:{p_manip:.4f} ({p_manip:.2%})")
        print(f"  P_ORIGINAL:   {p_orig:.4f} ({p_orig:.2%})")

        print(f"\nStrategy E Input Feature Vector:")
        print(f"  [P_FREQ_AI, P_MANIPULATED] = [{p_freq_ai:.4f}, {p_manip:.4f}]")

        print(f"\nStrategy E Output Probabilities:")
        print(f"  REAL_ORIGINAL:  {fusion_probs.get('REAL_ORIGINAL', 0.0):.4f} ({fusion_probs.get('REAL_ORIGINAL', 0.0):.2%})")
        print(f"  AI_GENERATED:   {fusion_probs.get('AI_GENERATED', 0.0):.4f} ({fusion_probs.get('AI_GENERATED', 0.0):.2%})")
        print(f"  AI_MANIPULATED: {fusion_probs.get('AI_MANIPULATED', 0.0):.4f} ({fusion_probs.get('AI_MANIPULATED', 0.0):.2%})")

        print(f"\nReliability & Disagreement Metrics:")
        rel_metrics = fin.get("reliability_metrics", {})
        print(f"  Generation Disagreement:        {rel_metrics.get('generation_disagreement', 'N/A')}")
        print(f"  Spatial-Frequency Disagreement: {rel_metrics.get('spatial_frequency_disagreement', 'N/A')}")
        print(f"  Hybrid Disagreement:            {rel_metrics.get('hybrid_disagreement', 'N/A')}")
        print(f"  Reliability State:              {fin.get('reliability_state', 'N/A')}")

        print(f"\nConsensus State:")
        print(f"  {fin.get('consensus_state', 'UNKNOWN')}")

        print(f"\nFinal Forensic Decision:")
        print(f"  Final verdict:    {fin['label']}")
        print(f"  Final confidence: {fin['confidence']:.4f} ({fin['confidence']:.2%})")
        print(f"  Decision source:  {fin.get('decision_source', 'N/A')}")
        print(f"  Final rationale:  \"{fin['reason']}\"")
        print(f"  Decision case:    {fin['decision_case']}")
        print(f"  Strategy:         {fin['strategy']}")

        passed = (fin["label"] == tc["expected_verdict"])
        status = "PASSED" if passed else "FAILED"
        print(f"\nStatus: {status} (Expected: {tc['expected_verdict']}, Got: {fin['label']})")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    print(f"REGRESSION SUITE RESULT: {'ALL ' + str(len(TEST_CASES)) + ' TESTS PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 80)

    if not all_passed:
        sys.exit(1)

if __name__ == "__main__":
    run_regression()
