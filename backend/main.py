from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any

from PIL import Image, ImageFilter

import io
import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, models


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(
    BASE_DIR
)

MODEL_DIR = os.path.join(
    PROJECT_DIR,
    "models"
)

# Model selection: V3 is the robust production model, V2 remains available
USE_V2 = os.getenv("AIDETECT_USE_V2", "false").lower() == "true"

if USE_V2:
    SPATIAL_MODEL_PATH = os.path.join(MODEL_DIR, "spatial_resnet50_v2.pth")
    FREQUENCY_MODEL_PATH = os.path.join(MODEL_DIR, "frequency_resnet50_v2.pth")
    HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_resnet50_fft_v2.pth")
    MODEL_VERSION = "V2"
else:
    SPATIAL_MODEL_PATH = os.path.join(MODEL_DIR, "spatial_resnet50_v3.pth")
    FREQUENCY_MODEL_PATH = os.path.join(MODEL_DIR, "frequency_resnet50_v3.pth")
    HYBRID_MODEL_PATH = os.path.join(MODEL_DIR, "hybrid_resnet50_fft_v3.pth")
    MODEL_VERSION = "V3"

# V5-D Gated Residual Model Checkpoint
V5_D_MODEL_PATH = os.path.join(BASE_DIR, "models", "v5", "gated_residual_resnet50_v5_best.pth")
if not os.path.exists(V5_D_MODEL_PATH):
    V5_D_MODEL_PATH = os.path.join(PROJECT_DIR, "models", "v5", "gated_residual_resnet50_v5_best.pth")



# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print(f"AIDetect {MODEL_VERSION} Backend")
print("=" * 70)

print()
print("Device:", DEVICE)

if torch.cuda.is_available():

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

image_transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# FFT TRANSFORM
# ============================================================

class FFTTransform:

    def __call__(
        self,
        image
    ):

        # ----------------------------------------------------
        # 1. PIL -> Tensor at original image resolution
        # ----------------------------------------------------

        if not isinstance(
            image,
            Image.Image
        ):
            image = Image.fromarray(
                image
            )

        image = image.convert(
            "RGB"
        )

        x = transforms.ToTensor()(
            image
        )

        # ----------------------------------------------------
        # 2. FFT at original image resolution
        # ----------------------------------------------------

        fft = torch.fft.fft2(
            x
        )

        # ----------------------------------------------------
        # 3. Shift low frequencies to center
        # ----------------------------------------------------

        fft = torch.fft.fftshift(
            fft,
            dim=(-2, -1)
        )

        # ----------------------------------------------------
        # 4. Magnitude
        # ----------------------------------------------------

        magnitude = torch.abs(
            fft
        )

        # ----------------------------------------------------
        # 5. Log scaling
        # ----------------------------------------------------

        magnitude = torch.log1p(
            magnitude
        )

        # ----------------------------------------------------
        # 6. Per-channel min-max normalization
        # ----------------------------------------------------

        for c in range(
            magnitude.shape[0]
        ):

            channel = magnitude[c]

            min_val = channel.min()
            max_val = channel.max()

            magnitude[c] = (
                (channel - min_val)
                /
                (
                    max_val - min_val
                    + 1e-8
                )
            )

        # ----------------------------------------------------
        # 7. Interpolate frequency representation to 224 x 224
        # ----------------------------------------------------

        magnitude = F.interpolate(
            magnitude.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        # ----------------------------------------------------
        # 8. Channel-wise ImageNet normalization
        # ----------------------------------------------------
        norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        return norm(magnitude)



frequency_transform = FFTTransform()


# ============================================================
# V5-D PREPROCESSING TRANSFORMS (AUTHORITATIVE)
# ============================================================
import sys as _sys
for _p in [PROJECT_DIR, BASE_DIR, os.path.join(BASE_DIR, "v5")]:
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

try:
    from backend.v5.v5_models import GatedResidualResNet50
    from backend.v5.v5_dataset import build_v5_spatial_val_transform, AuthoritativeNativeFFTTransform
    from backend.v5.v5_calibrator import V5DCalibrator
except ImportError:
    try:
        from v5.v5_models import GatedResidualResNet50
        from v5.v5_dataset import build_v5_spatial_val_transform, AuthoritativeNativeFFTTransform
        from v5.v5_calibrator import V5DCalibrator
    except ImportError:
        from v5_models import GatedResidualResNet50
        from v5_dataset import build_v5_spatial_val_transform, AuthoritativeNativeFFTTransform
        from v5_calibrator import V5DCalibrator

v5_spatial_transform = build_v5_spatial_val_transform()
v5_fft_transform = AuthoritativeNativeFFTTransform()
v5_d_calibrator = V5DCalibrator()

def create_v5_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3):
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
    x = coords.unsqueeze(0)
    y = coords.unsqueeze(1)
    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel

V5_GAUSSIAN_KERNEL = create_v5_gaussian_kernel(kernel_size=5, sigma=1.0, channels=3).to(DEVICE)

def extract_noise_residual_v5(spatial_tensor: torch.Tensor) -> torch.Tensor:
    blurred = F.conv2d(spatial_tensor, V5_GAUSSIAN_KERNEL, padding=2, groups=3)
    return spatial_tensor - blurred


# ============================================================
# SPATIAL MODEL
# ============================================================

class SpatialResNet50(
    nn.Module
):

    def __init__(self, use_v3_head: bool = True):

        super().__init__()

        self.model = models.resnet50(
            weights=None
        )

        num_features = (
            self.model.fc.in_features
        )

        if use_v3_head:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=0.30),
                nn.Linear(
                    num_features,
                    2
                )
            )
        else:
            self.model.fc = nn.Linear(
                num_features,
                2
            )

    def forward(
        self,
        x
    ):

        return self.model(x)


# ============================================================
# FREQUENCY MODEL
# ============================================================

class FrequencyResNet50(
    nn.Module
):

    def __init__(self):

        super().__init__()

        self.model = models.resnet50(
            weights=None
        )

        num_features = (
            self.model.fc.in_features
        )

        self.model.fc = nn.Linear(
            num_features,
            2
        )

    def forward(
        self,
        x
    ):

        return self.model(x)


# ============================================================
# HYBRID MODEL
# ============================================================

class HybridResNet50FFT(
    nn.Module
):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # SPATIAL BRANCH
        # ----------------------------------------------------

        spatial_model = models.resnet50(
            weights=None
        )

        self.spatial_features = nn.Sequential(
            *list(
                spatial_model.children()
            )[:-1]
        )

        self.spatial_dim = 2048

        # ----------------------------------------------------
        # FREQUENCY BRANCH
        # ----------------------------------------------------

        self.frequency_features = nn.Sequential(

            nn.Conv2d(
                3,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                32
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                64
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                128
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.MaxPool2d(
                2
            ),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(
                256
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.frequency_dim = 256

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        combined_dim = (
            self.spatial_dim
            +
            self.frequency_dim
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                combined_dim,
                512
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                512,
                2
            )
        )

    def forward(
        self,
        spatial_input,
        frequency_input
    ):

        # ----------------------------------------------------
        # Spatial features
        # ----------------------------------------------------

        spatial_features = (
            self.spatial_features(
                spatial_input
            )
        )

        spatial_features = (
            spatial_features.view(
                spatial_features.size(0),
                -1
            )
        )

        # ----------------------------------------------------
        # Frequency features
        # ----------------------------------------------------

        frequency_features = (
            self.frequency_features(
                frequency_input
            )
        )

        frequency_features = (
            frequency_features.view(
                frequency_features.size(0),
                -1
            )
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        combined = torch.cat(
            [
                spatial_features,
                frequency_features
            ],
            dim=1
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        return self.classifier(
            combined
        )


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    model_name
):

    print()
    print("-" * 70)
    print(
        f"Loading {model_name}"
    )
    print("-" * 70)

    print(
        checkpoint_path
    )

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            f"Model file not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    # --------------------------------------------------------
    # Extract state dictionary
    # --------------------------------------------------------

    if isinstance(
        checkpoint,
        dict
    ):

        if (
            "model_state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "model_state_dict"
                ]
            )

        elif (
            "state_dict"
            in checkpoint
        ):

            state_dict = (
                checkpoint[
                    "state_dict"
                ]
            )

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    # --------------------------------------------------------
    # Remove DataParallel prefix
    # --------------------------------------------------------

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[7:]

        cleaned_state_dict[
            key
        ] = value

    # --------------------------------------------------------
    # Spatial / Frequency raw ResNet checkpoint
    # --------------------------------------------------------

    if hasattr(model, "model"):
        target_model = model.model
    else:
        target_model = model

    # Ensure fc head matches checkpoint head structure
    if hasattr(target_model, "fc"):
        if "fc.1.weight" in cleaned_state_dict and isinstance(target_model.fc, nn.Linear):
            target_model.fc = nn.Sequential(
                nn.Dropout(p=0.30),
                nn.Linear(target_model.fc.in_features, 2)
            )
        elif "fc.weight" in cleaned_state_dict and isinstance(target_model.fc, nn.Sequential):
            target_model.fc = nn.Linear(2048, 2)

    # --------------------------------------------------------
    # STRICT LOAD
    # --------------------------------------------------------

    target_model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(
        DEVICE
    )

    model.eval()

    print(
        f"[OK] {model_name} "
        f"loaded with STRICT matching."
    )

    return model


# ============================================================
# LOAD ALL MODELS
# ============================================================

print()
print(f"Loading {MODEL_VERSION} models...")

spatial_model = load_checkpoint(
    SpatialResNet50(),
    SPATIAL_MODEL_PATH,
    f"Spatial {MODEL_VERSION}"
)

frequency_model = load_checkpoint(
    FrequencyResNet50(),
    FREQUENCY_MODEL_PATH,
    f"Frequency {MODEL_VERSION}"
)

hybrid_model = load_checkpoint(
    HybridResNet50FFT(),
    HYBRID_MODEL_PATH,
    f"Hybrid {MODEL_VERSION}"
)

print()
print("=" * 70)
print(f"ALL {MODEL_VERSION} MODELS LOADED SUCCESSFULLY")
print("=" * 70)

# ============================================================
# LOAD FORENSIC DECISION PIPELINE (PHASE 14/15 PRODUCTION)
# ============================================================
print("\nLoading Forensic Decision Pipeline (Phase 14/15 Production)...")
try:
    from backend.forensic_inference import ForensicInferencePipeline
except ImportError:
    from forensic_inference import ForensicInferencePipeline

forensic_pipeline = ForensicInferencePipeline(device=DEVICE, strategy="calibrated")
print(f"[OK] Forensic Decision Pipeline loaded with default strategy: '{forensic_pipeline.engine.default_strategy}'\n")

# ============================================================
# LOAD V5-D GATED RESIDUAL MODEL
# ============================================================
print("Loading V5-D Gated Residual Model...")
v5_d_model = None
if os.path.exists(V5_D_MODEL_PATH):
    try:
        v5_d_model = load_checkpoint(
            GatedResidualResNet50(),
            V5_D_MODEL_PATH,
            "V5-D Gated Residual"
        )
        print("[OK] V5-D Gated Residual model loaded successfully.\n")
    except Exception as e:
        print(f"[WARNING] Could not load V5-D model from {V5_D_MODEL_PATH}: {e}\n")
else:
    print(f"[WARNING] V5-D checkpoint not found at {V5_D_MODEL_PATH}\n")

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title=f"AIDetect {MODEL_VERSION} API",
    version="4.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=[
        "*"
    ],

    allow_headers=[
        "*"
    ],
)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            f"AIDetect {MODEL_VERSION} backend is running",

        "device":
            str(DEVICE),

        "model_version":
            MODEL_VERSION,

        "models_loaded": {
            f"spatial_{MODEL_VERSION.lower()}": spatial_model is not None,
            f"frequency_{MODEL_VERSION.lower()}": frequency_model is not None,
            f"hybrid_{MODEL_VERSION.lower()}": hybrid_model is not None,
            "v5_d_gated_residual": v5_d_model is not None,
        },
        "v5_d": {
            "loaded": v5_d_model is not None,
            "checkpoint": "gated_residual_resnet50_v5_best.pth",
            "model_name": "V5-D Gated Residual Calibrated",
            "calibrated": v5_d_calibrator.is_loaded,
            "calibration_method": v5_d_calibrator.version,
        },
        "forensic_pipeline": {
            "loaded": forensic_pipeline is not None,
            "default_strategy": forensic_pipeline.engine.default_strategy,
            "models": {
                "generation": "frequency_resnet50_v4.pth",
                "manipulation": "manipulation_frequency_resnet50_v1.pth",
            }
        }
    }


# ============================================================
# SOFTMAX RESULT
# ============================================================

def result_from_logits(
    logits,
    model_name="Model"
):

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    logit_real = float(logits[0, 0].item())
    logit_ai = float(logits[0, 1].item())

    real_probability = (
        probabilities[0, 0].item()
    )

    ai_probability = (
        probabilities[0, 1].item()
    )

    if ai_probability >= 0.5:

        prediction = (
            "AI-GENERATED"
        )

        confidence = (
            ai_probability
        )

        predicted_class = 1

    else:

        prediction = "REAL"

        confidence = (
            real_probability
        )

        predicted_class = 0

    return {

        "prediction":
            prediction,

        "confidence":
            round(
                confidence * 100,
                2
            ),

        "ai_probability":
            round(
                ai_probability * 100,
                2
            ),

        "real_probability":
            round(
                real_probability * 100,
                2
            ),

        "debug": {
            "model": model_name,
            "logit_real": round(logit_real, 4),
            "logit_ai": round(logit_ai, 4),
            "real_probability": round(real_probability * 100, 4),
            "ai_probability": round(ai_probability * 100, 4),
            "predicted_class": predicted_class,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2)
        }
    }


# ============================================================
# RUN SPATIAL MODEL
# ============================================================

def run_spatial(
    image
):

    input_tensor = (
        image_transform(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        logits = spatial_model(
            input_tensor
        )

    return result_from_logits(
        logits,
        f"Spatial {MODEL_VERSION}"
    )


# ============================================================
# RUN FREQUENCY MODEL
# ============================================================

def run_frequency(
    image
):

    input_tensor = (
        frequency_transform(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        logits = frequency_model(
            input_tensor
        )

    return result_from_logits(
        logits,
        f"Frequency {MODEL_VERSION}"
    )


# ============================================================
# RUN HYBRID MODEL
# ============================================================

def run_hybrid(
    image
):

    spatial_input = (
        image_transform(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    frequency_input = (
        frequency_transform(
            image
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    with torch.no_grad():

        logits = hybrid_model(
            spatial_input,
            frequency_input
        )

    return result_from_logits(
        logits,
        f"Hybrid {MODEL_VERSION}"
    )


# ============================================================
# RUN ALL THREE MODELS
# ============================================================

def run_all_models(
    image
):

    spatial_result = run_spatial(
        image
    )

    frequency_result = run_frequency(
        image
    )

    hybrid_result = run_hybrid(
        image
    )

    return {

        "spatial": spatial_result,

        "frequency": frequency_result,

        "hybrid": hybrid_result
    }


# ============================================================
# RUN V5-D GATED RESIDUAL MODEL
# ============================================================

def run_v5_d(
    image
) -> Dict[str, Any]:
    """
    Execute inference on the V5-D Gated Residual model (Spatial + Frequency + Noise Residual).
    Under torch.inference_mode().
    Deterministic labels: 0 = REAL, 1 = AI.
    """
    if v5_d_model is None:
        raise RuntimeError("V5-D Gated Residual model is not loaded.")

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("RGB")

    # 1. Authoritative V5 Spatial preprocessing (256 bilinear resize -> 224 center crop -> ImageNet normalized)
    spatial_tensor = v5_spatial_transform(image).unsqueeze(0).to(DEVICE)

    # 2. Authoritative V5 Native-Resolution FFT preprocessing -> 224 bilinear -> min-max -> ImageNet normalized
    freq_tensor = v5_fft_transform(image).unsqueeze(0).to(DEVICE)

    # 3. Authoritative V5 Noise Residual preprocessing (Spatial - 5x5 Gaussian blur with sigma=1.0)
    residual_tensor = extract_noise_residual_v5(spatial_tensor)

    with torch.inference_mode():
        logits, gate = v5_d_model.forward_with_gates(
            spatial_tensor,
            freq_tensor,
            residual_tensor
        )
        probs = F.softmax(logits, dim=1)[0]
        prob_real = float(probs[0].item())
        prob_ai = float(probs[1].item())

        # Gating distribution across 3 encoders (2048 dims each)
        spatial_gate_mean = float(gate[0, :2048].mean().item())
        frequency_gate_mean = float(gate[0, 2048:4096].mean().item())
        residual_gate_mean = float(gate[0, 4096:].mean().item())

    raw_real_pct = round(prob_real * 100, 2)
    raw_ai_pct = round(prob_ai * 100, 2)

    # Post-Hoc Probability Calibration (Phases 7 & 8)
    logits_tuple = (float(logits[0, 0].item()), float(logits[0, 1].item()))
    cal_res = v5_d_calibrator.calibrate(
        raw_real_prob=raw_real_pct,
        raw_ai_prob=raw_ai_pct,
        logits=logits_tuple
    )

    cal_ai_pct = cal_res["calibrated_ai_probability"]
    cal_real_pct = cal_res["calibrated_real_probability"]
    cal_conf = cal_res["confidence"]
    cal_method = cal_res["calibration_method"]

    prediction = "AI-GENERATED" if cal_ai_pct >= 50.0 else "REAL"

    return {
        "prediction": prediction,
        "confidence": cal_conf,
        "raw_ai_probability": raw_ai_pct,
        "raw_real_probability": raw_real_pct,
        "calibrated_ai_probability": cal_ai_pct,
        "calibrated_real_probability": cal_real_pct,
        "ai_probability": cal_ai_pct,
        "real_probability": cal_real_pct,
        "calibration_method": cal_method,
        "model_name": "V5-D Gated Residual Calibrated",
        "gate_weights": {
            "spatial": round(spatial_gate_mean, 4),
            "frequency": round(frequency_gate_mean, 4),
            "residual": round(residual_gate_mean, 4)
        }
    }


# ============================================================
# ROBUSTNESS TRANSFORMATIONS
# ============================================================

def jpeg_compression(
    image
):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=35
    )

    buffer.seek(0)

    return Image.open(
        buffer
    ).convert("RGB")


def resize_image(
    image
):

    original_width, original_height = (
        image.size
    )

    small_width = max(
        32,
        original_width // 2
    )

    small_height = max(
        32,
        original_height // 2
    )

    resized = image.resize(
        (
            small_width,
            small_height
        ),
        Image.Resampling.LANCZOS
    )

    restored = resized.resize(
        (
            original_width,
            original_height
        ),
        Image.Resampling.LANCZOS
    )

    return restored


def blur_image(
    image
):

    return image.filter(
        ImageFilter.GaussianBlur(
            radius=2
        )
    )


def add_noise(
    image
):

    image_array = np.asarray(
        image
    ).astype(
        np.float32
    )

    noise = np.random.normal(
        loc=0,
        scale=12,
        size=image_array.shape
    )

    noisy_array = (
        image_array
        +
        noise
    )

    noisy_array = np.clip(
        noisy_array,
        0,
        255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        noisy_array
    ).convert("RGB")


# ============================================================
# ANALYZE ENDPOINT
# ============================================================

@app.post(
    "/analyze"
)
async def analyze_image(
    file: UploadFile = File(...),
    strategy: Optional[str] = Query(None)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if (
        not file.content_type
        or
        not file.content_type.startswith(
            "image/"
        )
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload a valid "
                "image file."
            )
        )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    try:

        contents = await file.read()
        if len(contents) == 0:
            raise ValueError("Uploaded file is empty.")

        original_image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=(
                "Could not read the "
                "uploaded image."
            )
        )

    # --------------------------------------------------------
    # Forensic inference (Phase 14/15 Production Integration)
    # --------------------------------------------------------

    try:
        forensic_result = forensic_pipeline.predict(
            original_image,
            strategy=strategy
        )
    except Exception as e:
        forensic_result = {
            "error": str(e)
        }

    # --------------------------------------------------------
    # Original image - all models
    # --------------------------------------------------------

    try:

        model_results = run_all_models(
            original_image
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Model inference failed: "
                f"{str(e)}"
            )
        )

    # --------------------------------------------------------
    # V5-D Gated Residual Inference (Primary Classifier)
    # --------------------------------------------------------

    v5_d_result = None
    if v5_d_model is not None:
        try:
            v5_d_result = run_v5_d(
                original_image
            )
        except Exception as e:
            print(f"[WARNING] V5-D inference failed: {e}")
            v5_d_result = {
                "error": str(e),
                "model_name": "V5-D Gated Residual"
            }

    # --------------------------------------------------------
    # Primary result selection (V5-D Gated Residual Calibrated)
    # --------------------------------------------------------

    if v5_d_result is not None and "error" not in v5_d_result:
        primary_result = v5_d_result
        primary_verdict = v5_d_result["prediction"]  # "AI-GENERATED" or "REAL"
        primary_confidence = v5_d_result["confidence"]
    else:
        primary_result = model_results["hybrid"]
        primary_verdict = primary_result["prediction"]
        primary_confidence = primary_result["confidence"]

    # --------------------------------------------------------
    # Supporting Forensic Cross-Check (Strategy E)
    # Strategy E internals and calculations remain 100% intact.
    # --------------------------------------------------------
    forensic_cross_check = None
    if forensic_result and "final" in forensic_result:
        strategy_e_final = forensic_result["final"]
        strategy_e_label = strategy_e_final.get("label", "UNCERTAIN")
        strategy_e_conf = strategy_e_final.get("confidence", 0.0)

        # Check alignment between primary V5-D and Strategy E
        if strategy_e_label == "UNCERTAIN":
            cross_check_status = "CONFLICTING"
            cross_check_explanation = (
                "Legacy forensic signals show intra-model disagreement or domain shift; "
                "calibrated V5-D serves as the authoritative primary classifier."
            )
        elif (primary_verdict == "REAL" and strategy_e_label == "REAL_ORIGINAL") or \
             (primary_verdict == "AI-GENERATED" and strategy_e_label in ("AI_GENERATED", "AI_MANIPULATED")):
            cross_check_status = "CONSISTENT"
            cross_check_explanation = (
                "Forensic cross-check confirms primary classification evidence."
            )
        else:
            cross_check_status = "CONFLICTING"
            cross_check_explanation = (
                "Legacy forensic signals disagree with the primary classification; "
                "calibrated V5-D prediction takes precedence."
            )

        forensic_cross_check = {
            "status": cross_check_status,
            "strategy_e_verdict": strategy_e_label,
            "strategy_e_confidence": round(strategy_e_conf * 100 if strategy_e_conf <= 1.0 else strategy_e_conf, 2),
            "strategy_e_reason": strategy_e_final.get("reason", ""),
            "explanation": cross_check_explanation
        }

    # ========================================================
    # ROBUSTNESS ANALYSIS
    # ========================================================

    try:

        # ----------------------------------------------------
        # JPEG
        # ----------------------------------------------------

        jpeg_image = jpeg_compression(
            original_image
        )

        jpeg_results = run_all_models(
            jpeg_image
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        resized_image = resize_image(
            original_image
        )

        resize_results = run_all_models(
            resized_image
        )

        # ----------------------------------------------------
        # Blur
        # ----------------------------------------------------

        blurred_image = blur_image(
            original_image
        )

        blur_results = run_all_models(
            blurred_image
        )

        # ----------------------------------------------------
        # Noise
        # ----------------------------------------------------

        noisy_image = add_noise(
            original_image
        )

        noise_results = run_all_models(
            noisy_image
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Robustness analysis failed: "
                f"{str(e)}"
            )
        )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "filename":
            file.filename,

        # ----------------------------------------------------
        # PRIMARY RESULT (V5-D Gated Residual Calibrated)
        # ----------------------------------------------------

        "prediction":
            primary_verdict,

        "confidence":
            primary_confidence,

        "primary_verdict":
            primary_verdict,

        "primary_confidence":
            primary_confidence,

        "ai_probability":
            primary_result.get("calibrated_ai_probability", primary_result["ai_probability"]),

        "real_probability":
            primary_result.get("calibrated_real_probability", primary_result["real_probability"]),

        "forensic_cross_check":
            forensic_cross_check,

        # ----------------------------------------------------
        # MODEL COMPARISON
        # ----------------------------------------------------

        "models": {

            "spatial":
                model_results[
                    "spatial"
                ],

            "frequency":
                model_results[
                    "frequency"
                ],

            "hybrid":
                model_results[
                    "hybrid"
                ]
        },

        # ----------------------------------------------------
        # DEBUG INFORMATION (Spatial, Frequency, Hybrid)
        # ----------------------------------------------------

        "debug": {

            "spatial":
                model_results[
                    "spatial"
                ][
                    "debug"
                ],

            "frequency":
                model_results[
                    "frequency"
                ][
                    "debug"
                ],

            "hybrid":
                model_results[
                    "hybrid"
                ][
                    "debug"
                ]
        },

        # ----------------------------------------------------
        # ROBUSTNESS
        #
        # Primary robustness values use Hybrid model.
        # ----------------------------------------------------

        "robustness": {

            "original":
                model_results[
                    "hybrid"
                ],

            "jpeg_compression":
                jpeg_results[
                    "hybrid"
                ],

            "resize":
                resize_results[
                    "hybrid"
                ],

            "blur":
                blur_results[
                    "hybrid"
                ],

            "noise":
                noise_results[
                    "hybrid"
                ]
        },

        # ----------------------------------------------------
        # Detailed robustness comparison
        # ----------------------------------------------------

        "robustness_models": {

            "original":
                model_results,

            "jpeg_compression":
                jpeg_results,

            "resize":
                resize_results,

            "blur":
                blur_results,

            "noise":
                noise_results
        },

        # ----------------------------------------------------
        # Integrated forensic pipeline (Phase 14/15)
        # ----------------------------------------------------

        "forensic":
            forensic_result,

        # ----------------------------------------------------
        # V5-D Gated Residual model result
        # ----------------------------------------------------

        "v5_d":
            v5_d_result,

        "message":
            "Image analyzed successfully"
    }


# ============================================================
# DEDICATED FORENSIC ANALYZE ENDPOINTS (PHASE 14/15)
# ============================================================

@app.post("/forensic/analyze")
@app.post("/analyze/forensic")
async def analyze_forensic(
    file: UploadFile = File(...),
    strategy: Optional[str] = Query(None, description="Decision strategy: 'calibrated' (default) or 'baseline'")
):
    """
    Dedicated forensic analysis endpoint integrating frozen Generation (V4)
    and frozen Manipulation (Phase 7 Frequency) detectors with the Strategy E
    calibrated decision engine.
    """
    if (
        not file.content_type
        or not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise ValueError("Uploaded file is empty.")

        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read the uploaded image: {str(e)}"
        )

    try:
        forensic_result = forensic_pipeline.predict(
            image,
            strategy=strategy
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Forensic inference failed: {str(e)}"
        )

    return {
        "filename": file.filename,
        "forensic": forensic_result,
        "final_label": forensic_result["final"]["label"],
        "final_confidence": forensic_result["final"]["confidence"],
        "strategy": forensic_result["final"]["strategy"],
        "decision_case": forensic_result["final"]["decision_case"],
        "message": "Forensic analysis completed successfully"
    }