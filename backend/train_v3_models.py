"""
AIDetect V3 Training Script - Multi-Domain & Real-World Robustness

Trains V3 models (Spatial, Frequency, Hybrid) using a balanced multi-generator,
multi-resolution dataset (combining high-resolution real photographs and multiple
diffusion/GAN generators from GenImage with CIFAKE samples).

Preserves all V2 checkpoints and benchmark results.
Saves new V3 checkpoints to:
- models/spatial_resnet50_v3.pth
- models/frequency_resnet50_v3.pth
- models/hybrid_resnet50_fft_v3.pth
"""

import os
import io
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4

print("=" * 75)
print("AIDETECT V3 TRAINING - MULTI-DOMAIN REAL-WORLD GENERALIZATION")
print("=" * 75)
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# LOAD CACHED GENIMAGE + CIFAKE DATA
# ============================================================

print("\nLoading training data...")

shard0 = r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\train-00000-of-00014.parquet"
shard1 = r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\train-00001-of-00014.parquet"

df0 = pd.read_parquet(shard0)
df1 = pd.read_parquet(shard1)
df_all = pd.concat([df0, df1], ignore_index=True)

# Select balanced 1500 Real + 1500 AI from GenImage
gen_real = df_all[df_all["label"] == 0].sample(n=1500, random_state=42)
gen_ai = df_all[df_all["label"] == 1].sample(n=1500, random_state=42)

# Also load 500 Real + 500 AI from CIFAKE to retain low-res competence
from datasets import load_dataset
cifake_ds = load_dataset("dragonintelligence/CIFAKE-image-dataset", split="train")

cifake_real_imgs = []
cifake_ai_imgs = []
for item in cifake_ds:
    lbl = int(item["label"]) # 0=AI, 1=Real in CIFAKE
    if lbl == 1 and len(cifake_real_imgs) < 500:
        cifake_real_imgs.append(item["image"])
    elif lbl == 0 and len(cifake_ai_imgs) < 500:
        cifake_ai_imgs.append(item["image"])
    if len(cifake_real_imgs) == 500 and len(cifake_ai_imgs) == 500:
        break

training_samples = []

# Add GenImage samples
for _, row in gen_real.iterrows():
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    training_samples.append((img, 0)) # 0 = REAL

for _, row in gen_ai.iterrows():
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    training_samples.append((img, 1)) # 1 = AI

# Add CIFAKE samples
for img in cifake_real_imgs:
    training_samples.append((img.convert("RGB"), 0))

for img in cifake_ai_imgs:
    training_samples.append((img.convert("RGB"), 1))

import random
random.seed(42)
random.shuffle(training_samples)

print(f"Total balanced training samples: {len(training_samples)} (2,000 Real + 2,000 AI)")

# ============================================================
# TRANSFORMS
# ============================================================

weights = ResNet50_Weights.DEFAULT
spatial_transform = weights.transforms()

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
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return norm(magnitude)

# ============================================================
# DATASETS
# ============================================================

class SpatialV3Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return spatial_transform(img), label

class FrequencyV3Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return authoritative_fft_transform(img), label

class HybridV3Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return spatial_transform(img), authoritative_fft_transform(img), label

spatial_loader = DataLoader(SpatialV3Dataset(training_samples), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
frequency_loader = DataLoader(FrequencyV3Dataset(training_samples), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
hybrid_loader = DataLoader(HybridV3Dataset(training_samples), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# ============================================================
# 1. TRAIN SPATIAL V3
# ============================================================

print("\n" + "=" * 60)
print("TRAINING SPATIAL V3")
print("=" * 60)

spatial_model = resnet50(weights=weights)
spatial_model.fc = nn.Linear(2048, 2)
spatial_model = spatial_model.to(DEVICE)

optimizer = torch.optim.AdamW(spatial_model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    spatial_model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for x, y in spatial_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = spatial_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {total_loss/total:.4f} - Acc: {correct/total*100:.2f}% ({time.time()-t0:.1f}s)")

torch.save(spatial_model.state_dict(), os.path.join(MODEL_DIR, "spatial_resnet50_v3.pth"))
print("Saved models/spatial_resnet50_v3.pth")

# ============================================================
# 2. TRAIN FREQUENCY V3
# ============================================================

print("\n" + "=" * 60)
print("TRAINING FREQUENCY V3")
print("=" * 60)

frequency_model = resnet50(weights=weights)
frequency_model.fc = nn.Linear(2048, 2)
frequency_model = frequency_model.to(DEVICE)

optimizer = torch.optim.AdamW(frequency_model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    frequency_model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for x, y in frequency_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = frequency_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {total_loss/total:.4f} - Acc: {correct/total*100:.2f}% ({time.time()-t0:.1f}s)")

torch.save(frequency_model.state_dict(), os.path.join(MODEL_DIR, "frequency_resnet50_v3.pth"))
print("Saved models/frequency_resnet50_v3.pth")

# ============================================================
# 3. TRAIN HYBRID V3
# ============================================================

print("\n" + "=" * 60)
print("TRAINING HYBRID V3")
print("=" * 60)

class HybridResNet50FFTV3(nn.Module):
    def __init__(self):
        super().__init__()
        spatial_base = resnet50(weights=weights)
        self.spatial_features = nn.Sequential(*list(spatial_base.children())[:-1])
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

    def forward(self, s, f):
        s_feat = torch.flatten(self.spatial_features(s), 1)
        f_feat = torch.flatten(self.frequency_features(f), 1)
        return self.classifier(torch.cat([s_feat, f_feat], dim=1))

hybrid_model = HybridResNet50FFTV3().to(DEVICE)
optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    hybrid_model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()
    for s_in, f_in, y in hybrid_loader:
        s_in, f_in, y = s_in.to(DEVICE), f_in.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = hybrid_model(s_in, f_in)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    print(f"Epoch {epoch}/{EPOCHS} - Loss: {total_loss/total:.4f} - Acc: {correct/total*100:.2f}% ({time.time()-t0:.1f}s)")

torch.save(hybrid_model.state_dict(), os.path.join(MODEL_DIR, "hybrid_resnet50_fft_v3.pth"))
print("Saved models/hybrid_resnet50_fft_v3.pth")

# ============================================================
# EVALUATION ON USER'S IMAGE & EXTERNAL DATASET
# ============================================================

print("\n" + "=" * 60)
print("TESTING V3 ON USER MIRROR SELFIE (image.png)")
print("=" * 60)

spatial_model.eval()
frequency_model.eval()
hybrid_model.eval()

user_img = Image.open(os.path.join(BASE_DIR, "real_world_test", "REAL", "image.png")).convert("RGB")
s_in = spatial_transform(user_img).unsqueeze(0).to(DEVICE)
f_in = authoritative_fft_transform(user_img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    s_out = spatial_model(s_in)
    f_out = frequency_model(f_in)
    h_out = hybrid_model(s_in, f_in)

    s_prob = torch.softmax(s_out, 1)
    f_prob = torch.softmax(f_out, 1)
    h_prob = torch.softmax(h_out, 1)

print(f"Spatial V3:   Prediction={'REAL' if s_prob[0,0] > 0.5 else 'AI'} | Real Prob: {s_prob[0,0]*100:.2f}% | AI Prob: {s_prob[0,1]*100:.2f}%")
print(f"Frequency V3: Prediction={'REAL' if f_prob[0,0] > 0.5 else 'AI'} | Real Prob: {f_prob[0,0]*100:.2f}% | AI Prob: {f_prob[0,1]*100:.2f}%")
print(f"Hybrid V3:    Prediction={'REAL' if h_prob[0,0] > 0.5 else 'AI'} | Real Prob: {h_prob[0,0]*100:.2f}% | AI Prob: {h_prob[0,1]*100:.2f}%")

print("\n" + "=" * 60)
print("TESTING V3 ON ENTIRE 86-IMAGE EXTERNAL DATASET")
print("=" * 60)

ext_real_dir = os.path.join(BASE_DIR, "external_test", "REAL")
ext_ai_dir = os.path.join(BASE_DIR, "external_test", "AI")

real_files = sorted(os.listdir(ext_real_dir))
ai_files = sorted(os.listdir(ext_ai_dir))

h_real_correct, h_ai_correct = 0, 0
for f in real_files:
    img = Image.open(os.path.join(ext_real_dir, f)).convert("RGB")
    si = spatial_transform(img).unsqueeze(0).to(DEVICE)
    fi = authoritative_fft_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = hybrid_model(si, fi)
    if out.argmax(1).item() == 0:
        h_real_correct += 1

for f in ai_files:
    img = Image.open(os.path.join(ext_ai_dir, f)).convert("RGB")
    si = spatial_transform(img).unsqueeze(0).to(DEVICE)
    fi = authoritative_fft_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = hybrid_model(si, fi)
    if out.argmax(1).item() == 1:
        h_ai_correct += 1

print(f"Hybrid V3 on External Dataset (86 images):")
print(f"  REAL Correct: {h_real_correct}/49 ({h_real_correct/49*100:.2f}%) [was 0/49 with V2!]")
print(f"  AI Correct:   {h_ai_correct}/37 ({h_ai_correct/37*100:.2f}%)")
print(f"  Overall Acc:  {(h_real_correct + h_ai_correct)/86*100:.2f}% [was 43.02% with V2!]")

print("\n" + "=" * 75)
print("V3 TRAINING AND VERIFICATION COMPLETE")
print("=" * 75)
