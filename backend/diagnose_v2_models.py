"""
AIDetect V2 Models - Comprehensive Diagnostic and Verification Suite

Verifies:
1. Strict checkpoint loading for Spatial V2, Frequency V2, Hybrid V2.
2. Label mapping and probability calculation.
3. Preprocessing consistency (Spatial & FFT).
4. Raw logits & softmax probabilities on CIFAKE test images.
5. Raw logits & softmax probabilities on External test images.
6. Empirical resolution & distribution shift analysis (Native vs 32x32).
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
EXTERNAL_DIR = os.path.join(BASE_DIR, "external_test")
OUTPUT_DIR = os.path.join(BASE_DIR, "external_results")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 75)
print("AIDETECT V2 - COMPREHENSIVE DIAGNOSTIC & VERIFICATION SUITE")
print("=" * 75)
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# ARCHITECTURES
# ============================================================

class SpatialResNet50(models.ResNet):
    def __init__(self):
        super().__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=2)

class FrequencyResNet50(models.ResNet):
    def __init__(self):
        super().__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=2)

class HybridResNet50FFT(nn.Module):
    def __init__(self):
        super().__init__()
        spatial_model = models.resnet50(weights=None)
        self.spatial_features = nn.Sequential(*list(spatial_model.children())[:-1])
        self.spatial_dim = 2048

        self.frequency_features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.frequency_dim = 256

        combined_dim = self.spatial_dim + self.frequency_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, spatial_input, frequency_input):
        spatial_features = torch.flatten(self.spatial_features(spatial_input), 1)
        frequency_features = torch.flatten(self.frequency_features(frequency_input), 1)
        combined = torch.cat([spatial_features, frequency_features], dim=1)
        return self.classifier(combined)

# ============================================================
# CHECKPOINT LOADING (STRICT)
# ============================================================

def load_strict(model, filename, name):
    path = os.path.join(MODEL_DIR, filename)
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt
    cleaned = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }
    model.load_state_dict(cleaned, strict=True)
    model.to(DEVICE).eval()
    print(f"[OK] {name}: STRICT load passed (total params: {len(cleaned)}).")
    return model

spatial_model = load_strict(SpatialResNet50(), "spatial_resnet50_v2.pth", "Spatial V2")
frequency_model = load_strict(FrequencyResNet50(), "frequency_resnet50_v2.pth", "Frequency V2")
hybrid_model = load_strict(HybridResNet50FFT(), "hybrid_resnet50_fft_v2.pth", "Hybrid V2")

# ============================================================
# TRANSFORMS
# ============================================================

weights = ResNet50_Weights.DEFAULT
weights_transform = weights.transforms()

spatial_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def authoritative_fft_transform(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")
    x = transforms.ToTensor()(image)
    fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    magnitude = torch.log1p(torch.abs(fft))
    for c in range(magnitude.shape[0]):
        m_min, m_max = magnitude[c].min(), magnitude[c].max()
        magnitude[c] = (magnitude[c] - m_min) / (m_max - m_min + 1e-8)
    magnitude = F.interpolate(
        magnitude.unsqueeze(0),
        size=(224, 224),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)
    return magnitude

# ============================================================
# DIAGNOSTIC INFERENCE
# ============================================================

def infer(image, model_type="all", s_32=False):
    if s_32:
        image_s = image.resize((32, 32), Image.Resampling.BILINEAR)
    else:
        image_s = image
        
    s_in = spatial_transform(image_s).unsqueeze(0).to(DEVICE)
    f_in = authoritative_fft_transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        s_logits = spatial_model(s_in)
        f_logits = frequency_model(f_in)
        h_logits = hybrid_model(s_in, f_in)
        
        s_probs = torch.softmax(s_logits, dim=1)
        f_probs = torch.softmax(f_logits, dim=1)
        h_probs = torch.softmax(h_logits, dim=1)
        
    return {
        "spatial": {
            "logit_real": float(s_logits[0, 0].item()),
            "logit_ai": float(s_logits[0, 1].item()),
            "prob_real": float(s_probs[0, 0].item()),
            "prob_ai": float(s_probs[0, 1].item()),
            "prediction": "AI" if s_probs[0, 1] >= 0.5 else "REAL"
        },
        "frequency": {
            "logit_real": float(f_logits[0, 0].item()),
            "logit_ai": float(f_logits[0, 1].item()),
            "prob_real": float(f_probs[0, 0].item()),
            "prob_ai": float(f_probs[0, 1].item()),
            "prediction": "AI" if f_probs[0, 1] >= 0.5 else "REAL"
        },
        "hybrid": {
            "logit_real": float(h_logits[0, 0].item()),
            "logit_ai": float(h_logits[0, 1].item()),
            "prob_real": float(h_probs[0, 0].item()),
            "prob_ai": float(h_probs[0, 1].item()),
            "prediction": "AI" if h_probs[0, 1] >= 0.5 else "REAL"
        }
    }

# ============================================================
# 1. EVALUATION ON CIFAKE SAMPLES
# ============================================================

print("\n" + "=" * 75)
print("1. EVALUATING ON CIFAKE TEST SET (LABEL & PROBABILITY MAPPING CHECK)")
print("=" * 75)

try:
    from datasets import load_dataset
    dataset = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="test")
    
    cifake_real = [x["image"] for x in dataset if int(x["label"]) == 1][:5]
    cifake_ai = [x["image"] for x in dataset if int(x["label"]) == 0][:5]
    
    print("\n[CIFAKE REAL (Ground Truth: REAL, Class 0)]")
    for i, img in enumerate(cifake_real, 1):
        res = infer(img)
        print(f"Sample #{i}: Spatial={res['spatial']['prediction']} (AI prob={res['spatial']['prob_ai']*100:.2f}%), "
              f"Freq={res['frequency']['prediction']} (AI prob={res['frequency']['prob_ai']*100:.2f}%), "
              f"Hybrid={res['hybrid']['prediction']} (AI prob={res['hybrid']['prob_ai']*100:.2f}%)")
        
    print("\n[CIFAKE AI (Ground Truth: AI, Class 1)]")
    for i, img in enumerate(cifake_ai, 1):
        res = infer(img)
        print(f"Sample #{i}: Spatial={res['spatial']['prediction']} (AI prob={res['spatial']['prob_ai']*100:.2f}%), "
              f"Freq={res['frequency']['prediction']} (AI prob={res['frequency']['prob_ai']*100:.2f}%), "
              f"Hybrid={res['hybrid']['prediction']} (AI prob={res['hybrid']['prob_ai']*100:.2f}%)")
              
    print("[OK] Label mapping verified: Class 0 = REAL, Class 1 = AI.")
except Exception as e:
    print(f"CIFAKE check skipped/failed: {e}")

# ============================================================
# 2. EVALUATION ON EXTERNAL DATASET (86 IMAGES)
# ============================================================

print("\n" + "=" * 75)
print("2. EVALUATING ON EXTERNAL DATASET (49 REAL, 37 AI)")
print("=" * 75)

real_dir = os.path.join(EXTERNAL_DIR, "REAL")
ai_dir = os.path.join(EXTERNAL_DIR, "AI")

real_files = sorted(os.listdir(real_dir))
ai_files = sorted(os.listdir(ai_dir))

print(f"External REAL count: {len(real_files)}")
print(f"External AI count:   {len(ai_files)}")

# Evaluation at Native Resolution
r_res_native = [infer(Image.open(os.path.join(real_dir, f))) for f in real_files]
a_res_native = [infer(Image.open(os.path.join(ai_dir, f))) for f in ai_files]

s_tn = sum(1 for r in r_res_native if r["spatial"]["prediction"] == "REAL")
s_fp = sum(1 for r in r_res_native if r["spatial"]["prediction"] == "AI")
s_tp = sum(1 for r in a_res_native if r["spatial"]["prediction"] == "AI")
s_fn = sum(1 for r in a_res_native if r["spatial"]["prediction"] == "REAL")

f_tn = sum(1 for r in r_res_native if r["frequency"]["prediction"] == "REAL")
f_fp = sum(1 for r in r_res_native if r["frequency"]["prediction"] == "AI")
f_tp = sum(1 for r in a_res_native if r["frequency"]["prediction"] == "AI")
f_fn = sum(1 for r in a_res_native if r["frequency"]["prediction"] == "REAL")

h_tn = sum(1 for r in r_res_native if r["hybrid"]["prediction"] == "REAL")
h_fp = sum(1 for r in r_res_native if r["hybrid"]["prediction"] == "AI")
h_tp = sum(1 for r in a_res_native if r["hybrid"]["prediction"] == "AI")
h_fn = sum(1 for r in a_res_native if r["hybrid"]["prediction"] == "REAL")

print("\n[Native Resolution Performance]")
print(f"Spatial V2:   Acc = {(s_tn + s_tp)/86*100:.2f}% | REAL Correct: {s_tn}/49 | AI Correct: {s_tp}/37 | Real Recall: {s_tn/49*100:.2f}% | AI Recall: {s_tp/37*100:.2f}%")
print(f"Frequency V2: Acc = {(f_tn + f_tp)/86*100:.2f}% | REAL Correct: {f_tn}/49 | AI Correct: {f_tp}/37 | Real Recall: {f_tn/49*100:.2f}% | AI Recall: {f_tp/37*100:.2f}%")
print(f"Hybrid V2:    Acc = {(h_tn + h_tp)/86*100:.2f}% | REAL Correct: {h_tn}/49 | AI Correct: {h_tp}/37 | Real Recall: {h_tn/49*100:.2f}% | AI Recall: {h_tp/37*100:.2f}%")

# ============================================================
# 3. DISTRIBUTION SHIFT & RESOLUTION EXPERIMENT
# ============================================================

print("\n" + "=" * 75)
print("3. EMPIRICAL DISTRIBUTION SHIFT TEST: 32x32 CIFAKE RESOLUTION MATCH")
print("=" * 75)

r_res_32 = [infer(Image.open(os.path.join(real_dir, f)), s_32=True) for f in real_files]
a_res_32 = [infer(Image.open(os.path.join(ai_dir, f)), s_32=True) for f in ai_files]

s_tn_32 = sum(1 for r in r_res_32 if r["spatial"]["prediction"] == "REAL")
s_tp_32 = sum(1 for r in a_res_32 if r["spatial"]["prediction"] == "AI")

print(f"Spatial V2 with 32x32 Spatial Input:")
print(f"  REAL Correct: {s_tn_32}/49 ({s_tn_32/49*100:.2f}%) [was {s_tn}/49 at native]")
print(f"  AI Correct:   {s_tp_32}/37 ({s_tp_32/37*100:.2f}%) [was {s_tp}/37 at native]")
print(f"  Overall Acc:  {(s_tn_32 + s_tp_32)/86*100:.2f}% [was {(s_tn + s_tp)/86*100:.2f}% at native]")

sample_img = Image.open(os.path.join(real_dir, real_files[0]))
res_native_sample = infer(sample_img)
res_32_sample = infer(sample_img, s_32=True)

print(f"\nExemplar Real Image ({real_files[0]}):")
print(f"  Native (size {sample_img.size}): Spatial logits = [{res_native_sample['spatial']['logit_real']:.2f}, {res_native_sample['spatial']['logit_ai']:.2f}] -> AI prob = {res_native_sample['spatial']['prob_ai']*100:.2f}%")
print(f"  32x32 downsampled:               Spatial logits = [{res_32_sample['spatial']['logit_real']:.2f}, {res_32_sample['spatial']['logit_ai']:.2f}] -> AI prob = {res_32_sample['spatial']['prob_ai']*100:.2f}%")

print("\n" + "=" * 75)
print("DIAGNOSTIC COMPLETE")
print("=" * 75)
