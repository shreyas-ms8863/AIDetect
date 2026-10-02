"""
AIDetect V3 Robust Training Script - 100% Offline, Fast & GPU-Accelerated

Trained on:
- High-res Real photos (GenImage + Real camera/smartphone photos)
- Modern AI images (Midjourney, Stable Diffusion, DALL-E, ChatGPT)
- CIFAKE Real & AI (for low-res robustness)

Strictly holds out:
- image.png (unseen user mirror selfie)
- 24 External Real camera photos
- 19 External AI images
- CIFAKE held-out test split
"""

import os
import io
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models, transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import pandas as pd
import numpy as np

# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

print("=" * 70)
print("TRAINING AIDETECT V3 ROBUST MODELS (OFFLINE ON GPU)")
print("=" * 70)
print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# -------------------------------------------------------------
# Transforms
# -------------------------------------------------------------
weights = ResNet50_Weights.DEFAULT
spatial_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def authoritative_fft(img):
    if not isinstance(img, Image.Image):
        img = Image.fromarray(img)
    img = img.convert("RGB")
    x = transforms.ToTensor()(img)
    fft = torch.fft.fftshift(torch.fft.fft2(x), dim=(-2, -1))
    mag = torch.log1p(torch.abs(fft))
    for c in range(mag.shape[0]):
        mi, ma = mag[c].min(), mag[c].max()
        mag[c] = (mag[c] - mi) / (ma - mi + 1e-8)
    mag = F.interpolate(mag.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False).squeeze(0)
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return norm(mag)

# -------------------------------------------------------------
# Offline Data Loading
# -------------------------------------------------------------
print("\nLoading datasets completely offline from disk...")

# 1. GenImage shards
shard0 = r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\train-00000-of-00014.parquet"
shard1 = r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--TheKernel01--Tiny-GenImage\snapshots\89c4fe9efd0ebc7ce5c7641ef57d578ccd639c69\data\train-00001-of-00014.parquet"
df_gen = pd.concat([pd.read_parquet(shard0), pd.read_parquet(shard1)], ignore_index=True)

gen_real_df = df_gen[df_gen["label"] == 0].sample(n=800, random_state=42)
gen_ai_df = df_gen[df_gen["label"] == 1].sample(n=800, random_state=42)

# 2. CIFAKE local parquet
cifake_path = r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--dragonintelligence--CIFAKE-image-dataset\snapshots\6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31\data\train-00000-of-00001.parquet"
df_cifake = pd.read_parquet(cifake_path)
cifake_real_df = df_cifake[df_cifake["label"] == 1].sample(n=500, random_state=42) # In CIFAKE, 1 = Real
cifake_ai_df = df_cifake[df_cifake["label"] == 0].sample(n=500, random_state=42)   # In CIFAKE, 0 = AI

# 3. Smartphone & Real-World Cameras
ext_real_dir = os.path.join(BASE_DIR, "external_test", "REAL")
all_ext_real = sorted(os.listdir(ext_real_dir))
ext_real_train = [os.path.join(ext_real_dir, f) for f in all_ext_real[:25]]
ext_real_test = [os.path.join(ext_real_dir, f) for f in all_ext_real[25:]] # 24 held out

ext_ai_dir = os.path.join(BASE_DIR, "external_test", "AI")
all_ext_ai = sorted(os.listdir(ext_ai_dir))
ext_ai_train = [os.path.join(ext_ai_dir, f) for f in all_ext_ai[:18]]
ext_ai_test = [os.path.join(ext_ai_dir, f) for f in all_ext_ai[18:]] # 19 held out

rw_real_dir = os.path.join(BASE_DIR, "real_world_test", "REAL")
rw_ai_dir = os.path.join(BASE_DIR, "real_world_test", "AI")
rw_real_train = [os.path.join(rw_real_dir, f) for f in sorted(os.listdir(rw_real_dir)) if f != "image.png"][:4]
rw_ai_train = [os.path.join(rw_ai_dir, f) for f in sorted(os.listdir(rw_ai_dir))][:5]

print(f"GenImage: {len(gen_real_df)} Real, {len(gen_ai_df)} AI")
print(f"CIFAKE:   {len(cifake_real_df)} Real, {len(cifake_ai_df)} AI")
print(f"Camera:   {len(ext_real_train)+len(rw_real_train)} Real train, {len(ext_ai_train)+len(rw_ai_train)} AI train")
print(f"Held out test sets:")
print(f"  - image.png (User mirror selfie)")
print(f"  - {len(ext_real_test)} external Real camera photos")
print(f"  - {len(ext_ai_test)} external AI images")

# -------------------------------------------------------------
# Precompute Tensors for Super-Fast GPU Training
# -------------------------------------------------------------
print("\nPrecomputing spatial and frequency tensors into memory...")
t0 = time.time()

spatial_tensors = []
freq_tensors = []
labels = []

# Add GenImage
for _, row in gen_real_df.iterrows():
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    spatial_tensors.append(spatial_transform(img))
    freq_tensors.append(authoritative_fft(img))
    labels.append(0)

for _, row in gen_ai_df.iterrows():
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    spatial_tensors.append(spatial_transform(img))
    freq_tensors.append(authoritative_fft(img))
    labels.append(1)

# Add CIFAKE
for _, row in cifake_real_df.iterrows():
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    spatial_tensors.append(spatial_transform(img))
    freq_tensors.append(authoritative_fft(img))
    labels.append(0)

for _, row in cifake_ai_df.iterrows():
    img = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    spatial_tensors.append(spatial_transform(img))
    freq_tensors.append(authoritative_fft(img))
    labels.append(1)

# Add Camera & Smartphone photos (repeated with small augmentations to anchor the real camera manifold)
color_jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)
h_flip = transforms.RandomHorizontalFlip(p=0.5)

for p in ext_real_train + rw_real_train:
    img = Image.open(p).convert("RGB")
    # Base
    spatial_tensors.append(spatial_transform(img))
    freq_tensors.append(authoritative_fft(img))
    labels.append(0)
    # Augmentations
    for _ in range(12):
        aug_img = h_flip(color_jitter(img))
        spatial_tensors.append(spatial_transform(aug_img))
        freq_tensors.append(authoritative_fft(aug_img))
        labels.append(0)

for p in ext_ai_train + rw_ai_train:
    img = Image.open(p).convert("RGB")
    # Base
    spatial_tensors.append(spatial_transform(img))
    freq_tensors.append(authoritative_fft(img))
    labels.append(1)
    # Augmentations
    for _ in range(15):
        aug_img = h_flip(color_jitter(img))
        spatial_tensors.append(spatial_transform(aug_img))
        freq_tensors.append(authoritative_fft(aug_img))
        labels.append(1)

# Convert to batched PyTorch tensors
X_spatial = torch.stack(spatial_tensors)
X_freq = torch.stack(freq_tensors)
Y = torch.tensor(labels, dtype=torch.long)

# Shuffle
perm = torch.randperm(len(Y))
X_spatial = X_spatial[perm]
X_freq = X_freq[perm]
Y = Y[perm]

real_cnt = (Y == 0).sum().item()
ai_cnt = (Y == 1).sum().item()
print(f"Precomputed {len(Y)} samples ({real_cnt} Real, {ai_cnt} AI) in {time.time()-t0:.1f}s")

spatial_loader = DataLoader(TensorDataset(X_spatial, Y), batch_size=BATCH_SIZE, shuffle=True)
freq_loader = DataLoader(TensorDataset(X_freq, Y), batch_size=BATCH_SIZE, shuffle=True)
hybrid_loader = DataLoader(TensorDataset(X_spatial, X_freq, Y), batch_size=BATCH_SIZE, shuffle=True)

# -------------------------------------------------------------
# 1. Train Spatial V3
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("1. TRAINING SPATIAL V3 (ResNet50)")
print("=" * 60)

spatial_model = resnet50(weights=weights)
spatial_model.fc = nn.Linear(2048, 2)
spatial_model = spatial_model.to(DEVICE)

optimizer = torch.optim.AdamW(spatial_model.parameters(), lr=LR, weight_decay=1e-3)
criterion = nn.CrossEntropyLoss()

for ep in range(1, EPOCHS + 1):
    spatial_model.train()
    total_loss, correct, total = 0.0, 0, 0
    t_start = time.time()
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
    print(f"Epoch {ep}/{EPOCHS} - Loss: {total_loss/total:.4f} - Acc: {correct/total*100:.2f}% ({time.time()-t_start:.1f}s)")

torch.save(spatial_model.state_dict(), os.path.join(MODEL_DIR, "spatial_resnet50_v3.pth"))
print("Saved models/spatial_resnet50_v3.pth")

# -------------------------------------------------------------
# 2. Train Frequency V3
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("2. TRAINING FREQUENCY V3 (ResNet50 on Authoritative FFT)")
print("=" * 60)

freq_model = resnet50(weights=weights)
freq_model.fc = nn.Linear(2048, 2)
freq_model = freq_model.to(DEVICE)

optimizer = torch.optim.AdamW(freq_model.parameters(), lr=LR, weight_decay=1e-3)

for ep in range(1, EPOCHS + 1):
    freq_model.train()
    total_loss, correct, total = 0.0, 0, 0
    t_start = time.time()
    for x, y in freq_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = freq_model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct += (out.argmax(1) == y).sum().item()
        total += len(y)
    print(f"Epoch {ep}/{EPOCHS} - Loss: {total_loss/total:.4f} - Acc: {correct/total*100:.2f}% ({time.time()-t_start:.1f}s)")

torch.save(freq_model.state_dict(), os.path.join(MODEL_DIR, "frequency_resnet50_v3.pth"))
print("Saved models/frequency_resnet50_v3.pth")

# -------------------------------------------------------------
# 3. Train Hybrid V3
# -------------------------------------------------------------
print("\n" + "=" * 60)
print("3. TRAINING HYBRID V3 (Spatial ResNet50 + Frequency ConvNet)")
print("=" * 60)

class HybridV3(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet50(weights=weights)
        self.spatial_features = nn.Sequential(*list(base.children())[:-1])
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
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
    def forward(self, s, f):
        s_feat = torch.flatten(self.spatial_features(s), 1)
        f_feat = torch.flatten(self.frequency_features(f), 1)
        return self.classifier(torch.cat([s_feat, f_feat], dim=1))

hybrid_model = HybridV3().to(DEVICE)
optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=LR, weight_decay=1e-3)

for ep in range(1, EPOCHS + 1):
    hybrid_model.train()
    total_loss, correct, total = 0.0, 0, 0
    t_start = time.time()
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
    print(f"Epoch {ep}/{EPOCHS} - Loss: {total_loss/total:.4f} - Acc: {correct/total*100:.2f}% ({time.time()-t_start:.1f}s)")

torch.save(hybrid_model.state_dict(), os.path.join(MODEL_DIR, "hybrid_resnet50_fft_v3.pth"))
print("Saved models/hybrid_resnet50_fft_v3.pth")

# -------------------------------------------------------------
# 4. Strict Evaluation on Held-Out Data
# -------------------------------------------------------------
print("\n" + "=" * 70)
print("EVALUATION ON HELD-OUT DATA")
print("=" * 70)

spatial_model.eval()
freq_model.eval()
hybrid_model.eval()

# 1. User mirror selfie (strictly held out!)
user_path = os.path.join(rw_real_dir, "image.png")
user_img = Image.open(user_path).convert("RGB")
s_x = spatial_transform(user_img).unsqueeze(0).to(DEVICE)
f_x = authoritative_fft(user_img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    s_p = torch.softmax(spatial_model(s_x), 1)[0]
    f_p = torch.softmax(freq_model(f_x), 1)[0]
    h_p = torch.softmax(hybrid_model(s_x, f_x), 1)[0]

print(f"\n--- USER MIRROR SELFIE (image.png - HELD OUT) ---")
print(f"Spatial V3:   {'REAL' if s_p[0] > 0.5 else 'AI'} (REAL: {s_p[0]*100:.2f}%, AI: {s_p[1]*100:.2f}%)")
print(f"Frequency V3: {'REAL' if f_p[0] > 0.5 else 'AI'} (REAL: {f_p[0]*100:.2f}%, AI: {f_p[1]*100:.2f}%)")
print(f"Hybrid V3:    {'REAL' if h_p[0] > 0.5 else 'AI'} (REAL: {h_p[0]*100:.2f}%, AI: {h_p[1]*100:.2f}%)")

# 2. Held-out External Test Photos
def evaluate_set(paths, expected_label, name):
    s_c, f_c, h_c = 0, 0, 0
    for p in paths:
        img = Image.open(p).convert("RGB")
        sx = spatial_transform(img).unsqueeze(0).to(DEVICE)
        fx = authoritative_fft(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if spatial_model(sx).argmax(1).item() == expected_label: s_c += 1
            if freq_model(fx).argmax(1).item() == expected_label: f_c += 1
            if hybrid_model(sx, fx).argmax(1).item() == expected_label: h_c += 1
    n = len(paths)
    exp_str = "REAL" if expected_label == 0 else "AI"
    print(f"\n{name} ({n} images, expected {exp_str}):")
    print(f"  Spatial V3:   {s_c}/{n} ({s_c/n*100:.1f}%)")
    print(f"  Frequency V3: {f_c}/{n} ({f_c/n*100:.1f}%)")
    print(f"  Hybrid V3:    {h_c}/{n} ({h_c/n*100:.1f}%)")

evaluate_set(ext_real_test, 0, "Held-out External REAL Photos")
evaluate_set(ext_ai_test, 1, "Held-out External AI Images")

# 3. Held-out CIFAKE Test Data
test_cifake_path = r"C:\Users\Shreyas\.cache\huggingface\hub\datasets--dragonintelligence--CIFAKE-image-dataset\snapshots\6eebd6090fe3c0d9d44f899a25f0a34e3c2dad31\data\test-00000-of-00001.parquet"
df_test = pd.read_parquet(test_cifake_path)
cifake_test_real = df_test[df_test["label"] == 1].head(100)
cifake_test_ai = df_test[df_test["label"] == 0].head(100)

h_cifake_correct = 0
for _, row in cifake_test_real.iterrows():
    im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    sx = spatial_transform(im).unsqueeze(0).to(DEVICE)
    fx = authoritative_fft(im).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if hybrid_model(sx, fx).argmax(1).item() == 0: h_cifake_correct += 1

for _, row in cifake_test_ai.iterrows():
    im = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
    sx = spatial_transform(im).unsqueeze(0).to(DEVICE)
    fx = authoritative_fft(im).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if hybrid_model(sx, fx).argmax(1).item() == 1: h_cifake_correct += 1

print(f"\nCIFAKE Test Split (200 balanced test images):")
print(f"  Hybrid V3 Accuracy: {h_cifake_correct}/200 ({h_cifake_correct/200*100:.1f}%)")

print("\n" + "=" * 70)
print("V3 TRAINING AND VALIDATION COMPLETE")
print("=" * 70)
