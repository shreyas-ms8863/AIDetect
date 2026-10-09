"""
AIDetect Domain-Risk & Visual Domain Analyzer
=============================================
Provides lightweight, generic visual-domain analysis to detect when an input image
departs substantially from the detector's validated natural-image operating domain
(e.g., documents, scans, forms, certificates, screenshots).

CRITICAL CONSTRAINTS:
- No hardcoded document names (no Aadhaar, PAN, passport, marks card, etc.)
- No hardcoded filenames, individuals, or user exceptions.
- Purely generic visual domain characteristics:
    1. Flat Paper / Canvas Fraction (neutral, low-variance light background)
    2. Text Glyph Density (connected dark character strokes on canvas)
    3. Rectilinear Line Energy (long continuous horizontal/vertical dividers, tables, borders)
    4. Monochromatic / Low-Saturation Dominance (absence of natural scene chrominance)
- Returns:
    domain_type:       NATURAL_PHOTO | DOCUMENT_LIKE | SCREENSHOT_LIKE | UNKNOWN
    domain_risk:       LOW | MEDIUM | HIGH
    domain_risk_score: float in [0.0, 1.0]
    domain_signals:    dict of underlying diagnostic metrics
"""

from typing import Dict, Any, Union
import numpy as np
from PIL import Image
from scipy import ndimage


# Thresholds for Domain Risk Classification
DOMAIN_RISK_HIGH_THRESHOLD   = 0.50
DOMAIN_RISK_MEDIUM_THRESHOLD = 0.28


def analyze_image_domain(
    image_input: Union[Image.Image, np.ndarray, str]
) -> Dict[str, Any]:
    """
    Analyze generic visual-domain characteristics of an image.

    Args:
        image_input: PIL Image, numpy array (H, W, 3) in [0, 255], or path string.

    Returns:
        Dict containing:
            - domain_type: str ("NATURAL_PHOTO", "DOCUMENT_LIKE", "SCREENSHOT_LIKE", "UNKNOWN")
            - domain_risk: str ("LOW", "MEDIUM", "HIGH")
            - domain_risk_score: float in [0.0, 1.0]
            - domain_signals: dict with individual normalized visual metrics
            - summary: human-readable domain summary
    """
    if isinstance(image_input, str):
        with Image.open(image_input) as raw:
            img = raw.convert("RGB")
    elif isinstance(image_input, Image.Image):
        img = image_input.convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(np.uint8(image_input)).convert("RGB")
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # Standardize scale for fast, scale-invariant feature extraction
    # 512x512 preserves text lines and borders while executing in < 15ms
    img_thumb = img.resize((512, 512), Image.Resampling.BILINEAR)
    arr = np.array(img_thumb, dtype=np.float32) / 255.0  # (512, 512, 3) in [0, 1]

    # -----------------------------------------------------------------------
    # 1. Color Saturation & Monochromatic Ratio
    # Natural photos typically have continuous chrominance diversity.
    # Documents, scans, and forms are largely achromatic (black ink, white paper).
    # -----------------------------------------------------------------------
    c_max = np.max(arr, axis=2)
    c_min = np.min(arr, axis=2)
    c_diff = c_max - c_min
    sat = np.where(c_max > 0.05, c_diff / (c_max + 1e-6), 0.0)
    mono_fraction = float(np.mean(sat < 0.14))  # fraction of nearly grayscale pixels

    # -----------------------------------------------------------------------
    # 2. Grayscale & Local Gradients
    # -----------------------------------------------------------------------
    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])
    grad_mag = np.pad(dx, ((0, 0), (0, 1))) + np.pad(dy, ((0, 1), (0, 0)))

    # -----------------------------------------------------------------------
    # 3. Paper / Canvas Mask
    # Paper is light background with very low local gradient variance and neutral color
    # -----------------------------------------------------------------------
    paper_mask = (gray > 0.68) & (grad_mag < 0.07) & (sat < 0.20)
    paper_fraction = float(np.mean(paper_mask))

    # -----------------------------------------------------------------------
    # 4. Text Glyphs on Canvas
    # Connected components of dark strokes (gray < 0.55) adjacent to the paper canvas
    # -----------------------------------------------------------------------
    struct_dilate = np.ones((7, 7), dtype=bool)
    near_paper = ndimage.binary_dilation(paper_mask, structure=struct_dilate)
    dark_on_paper = (gray < 0.55) & near_paper & (grad_mag > 0.08)

    labeled, num_features = ndimage.label(dark_on_paper)
    if num_features > 0:
        sizes = ndimage.sum(np.ones_like(dark_on_paper), labeled, range(1, num_features + 1))
        # Text glyph size range: 4 to 350 pixels on a 512x512 thumbnail
        text_glyphs = int(np.sum((sizes >= 4) & (sizes <= 350)))
    else:
        text_glyphs = 0

    # -----------------------------------------------------------------------
    # 5. Rectilinear Line Energy (Borders, Tables, Dividers)
    # Documents, tables, and forms feature long continuous horizontal &
    # vertical lines spanning >= 31px.
    # -----------------------------------------------------------------------
    h_lines = ndimage.uniform_filter1d(dx, size=31, axis=1)
    v_lines = ndimage.uniform_filter1d(dy, size=31, axis=0)
    line_energy = float(np.mean(h_lines > 0.09) + np.mean(v_lines > 0.09))

    # -----------------------------------------------------------------------
    # 6. Normalized Domain Risk Indicators
    # -----------------------------------------------------------------------
    # Canvas presence: ramps up when paper occupies > 35% of image
    s_paper = float(np.clip((paper_fraction - 0.35) / 0.25, 0.0, 1.0))
    # Text glyph count: ramps up from 50 to 200 glyphs
    s_glyphs = float(np.clip((text_glyphs - 50) / 150.0, 0.0, 1.0))
    # Line structure: forms, divider lines, table grids
    s_lines = float(np.clip(line_energy / 0.12, 0.0, 1.0))
    # Monochromatic ratio
    s_mono = float(np.clip((mono_fraction - 0.45) / 0.40, 0.0, 1.0))

    # Text on canvas interaction: glyphs inside a paper canvas
    text_on_canvas = s_paper * s_glyphs

    # Composite continuous risk score [0, 1]
    base_score = float(
        0.40 * text_on_canvas +
        0.25 * s_paper * s_mono +
        0.25 * s_lines +
        0.10 * s_mono
    )

    # Nonlinear boost when strong multimodal document indicators co-occur
    if text_on_canvas > 0.40 and s_lines > 0.40:
        base_score = min(1.0, base_score * 1.30)
    elif s_lines > 0.70 and s_mono > 0.70:
        # High-contrast scanner / photocopier line dominance
        base_score = min(1.0, base_score * 1.25)

    domain_risk_score = round(float(np.clip(base_score, 0.0, 1.0)), 4)

    # -----------------------------------------------------------------------
    # 7. Classify Domain Risk & Type
    # -----------------------------------------------------------------------
    if domain_risk_score >= DOMAIN_RISK_HIGH_THRESHOLD:
        domain_risk = "HIGH"
        if s_lines > 0.60 and mono_fraction < 0.60:
            domain_type = "SCREENSHOT_LIKE"
        else:
            domain_type = "DOCUMENT_LIKE"
    elif domain_risk_score >= DOMAIN_RISK_MEDIUM_THRESHOLD:
        domain_risk = "MEDIUM"
        if text_on_canvas > 0.25 or s_paper > 0.35:
            domain_type = "DOCUMENT_LIKE"
        else:
            domain_type = "UNKNOWN"
    else:
        domain_risk = "LOW"
        domain_type = "NATURAL_PHOTO"

    # Human-readable summary
    if domain_risk == "HIGH":
        summary = (
            f"Image exhibits strong structured {domain_type.lower().replace('_', ' ')} visual characteristics "
            f"(paper canvas: {paper_fraction:.1%}, glyphs: {text_glyphs}, lines: {line_energy:.4f}). "
            f"Domain characteristics fall outside the detector's validated natural-photograph operating distribution."
        )
    elif domain_risk == "MEDIUM":
        summary = (
            f"Image exhibits moderate out-of-distribution visual traits "
            f"(domain risk score: {domain_risk_score:.2f}). Operating reliability is moderate."
        )
    else:
        summary = (
            f"Image visual characteristics are consistent with natural photographic scenes "
            f"(domain risk score: {domain_risk_score:.2f}, risk: LOW)."
        )

    return {
        "domain_type": domain_type,
        "domain_risk": domain_risk,
        "domain_risk_score": domain_risk_score,
        "domain_signals": {
            "paper_canvas_fraction": round(paper_fraction, 4),
            "text_glyph_count": text_glyphs,
            "rectilinear_line_energy": round(line_energy, 4),
            "monochromatic_fraction": round(mono_fraction, 4),
        },
        "summary": summary,
    }

