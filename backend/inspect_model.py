import torch
from huggingface_hub import hf_hub_download

MODEL_REPO = "Medsa/ai-image-authenticity-detector"
MODEL_FILE = "detector_scripted.pt"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE
)

model = torch.jit.load(
    model_path,
    map_location="cpu"
)

model.eval()

print("=" * 60)
print("MODEL INSPECTION")
print("=" * 60)

print("\nModel:")
print(model)

print("\nModel code:")
try:
    print(model.code)
except Exception as e:
    print("Could not display model code:", e)

print("\nModel graph:")
try:
    print(model.graph)
except Exception as e:
    print("Could not display graph:", e)

print("=" * 60)
print("\n" + "=" * 60)
print("SPATIAL BRANCH CODE")
print("=" * 60)

try:
    print(model.spatial.code)
except Exception as e:
    print("Could not inspect spatial branch:", e)


print("\n" + "=" * 60)
print("FREQUENCY BRANCH CODE")
print("=" * 60)

try:
    print(model.frequency.code)
except Exception as e:
    print("Could not inspect frequency branch:", e)


print("\n" + "=" * 60)
print("NOISE BRANCH CODE")
print("=" * 60)

try:
    print(model.noise.code)
except Exception as e:
    print("Could not inspect noise branch:", e)


print("\n" + "=" * 60)
print("FUSION CODE")
print("=" * 60)

try:
    print(model.fusion.code)
except Exception as e:
    print("Could not inspect fusion:", e)