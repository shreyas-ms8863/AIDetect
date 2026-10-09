"""Read-only diagnostics for V5-D predictions on supplied images.

Run from the repository root. Importing backend.main loads the same production
models, transforms, checkpoint, and calibrator used by the API.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
CLASS_THRESHOLD_PERCENT = 50.0
BRANCH_SLICES = {
    "spatial": slice(0, 2048),
    "frequency": slice(2048, 4096),
    "residual": slice(4096, 6144),
}


def load_production_module():
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from backend import main as production

    if production.v5_d_model is None:
        raise RuntimeError(
            "Production V5-D did not load. Refusing to diagnose with the legacy fallback."
        )
    return production


def probability_result(production: Any, logits: Any) -> dict[str, Any]:
    probabilities = production.F.softmax(logits, dim=1)[0]
    raw_real = float(probabilities[0].item())
    raw_ai = float(probabilities[1].item())
    calibrated = production.v5_d_calibrator.calibrate(
        raw_real_prob=raw_real * 100.0,
        raw_ai_prob=raw_ai * 100.0,
        logits=(float(logits[0, 0].item()), float(logits[0, 1].item())),
    )
    calibrated_ai = float(calibrated["calibrated_ai_probability"])
    calibrated_real = float(calibrated["calibrated_real_probability"])
    prediction = "AI-GENERATED" if calibrated_ai >= CLASS_THRESHOLD_PERCENT else "REAL"
    return {
        "raw_real": raw_real * 100.0,
        "raw_ai": raw_ai * 100.0,
        "calibrated_real": calibrated_real,
        "calibrated_ai": calibrated_ai,
        "calibration_method": calibrated["calibration_method"],
        "confidence": max(calibrated_real, calibrated_ai),
        "prediction": prediction,
    }


def attach_capture_hooks(model: Any, captured: dict[str, Any]) -> list[Any]:
    handles = []
    for name in ("spatial", "frequency", "residual"):
        module = getattr(model, f"{name}_encoder")
        handles.append(
            module.register_forward_pre_hook(
                lambda _module, inputs, key=name: captured.__setitem__(
                    f"{key}_input", inputs[0]
                )
            )
        )
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, key=name: captured.__setitem__(
                    f"{key}_features", output
                )
            )
        )
    handles.append(
        model.gate.register_forward_hook(
            lambda _module, _inputs, output: captured.__setitem__("gate", output)
        )
    )
    return handles


def run_branch_ablations(production: Any, captured: dict[str, Any]) -> dict[str, Any]:
    torch = production.torch
    model = production.v5_d_model
    combined = torch.cat(
        [
            captured["spatial_features"],
            captured["frequency_features"],
            captured["residual_features"],
        ],
        dim=1,
    )
    gated_features = combined * captured["gate"]
    results = {}

    with torch.inference_mode():
        for branch, feature_slice in BRANCH_SLICES.items():
            ablated_features = gated_features.clone()
            ablated_features[:, feature_slice] = 0
            logits = model.classifier(ablated_features)
            result = probability_result(production, logits)
            result["delta_calibrated_ai"] = (
                result["calibrated_ai"]
                - float(captured["production_calibrated_ai"])
            )
            results[branch] = result

    return results


def get_image_metadata(path: Path) -> tuple[Image.Image, dict[str, Any]]:
    with Image.open(path) as opened:
        opened.load()
        width, height = opened.size
        image_format = opened.format
        image_mode = opened.mode
        try:
            orientation = opened.getexif().get(274)
        except Exception:
            orientation = None
        rgb_image = opened.convert("RGB")

    metadata = {
        "filename": path.name,
        "path": str(path.resolve()),
        "format": image_format,
        "width": width,
        "height": height,
        "aspect_ratio": width / height if height else None,
        "file_size_bytes": path.stat().st_size,
        "image_mode": image_mode,
        "exif_orientation": orientation,
        "decoded_rgb_width": rgb_image.width,
        "decoded_rgb_height": rgb_image.height,
    }
    return rgb_image, metadata


def describe_preprocessing(production: Any, rgb_image: Image.Image, captured: dict[str, Any]) -> dict[str, Any]:
    transform_steps = production.v5_spatial_transform.transforms
    resize_transform = transform_steps[0]
    resized = resize_transform(rgb_image)
    crop_size = 224
    crop_left = int(round((resized.width - crop_size) / 2.0))
    crop_top = int(round((resized.height - crop_size) / 2.0))

    return {
        "decoded_rgb_dimensions": [rgb_image.width, rgb_image.height],
        "spatial_input_dimensions": list(captured["spatial_input"].shape),
        "frequency_input_dimensions": list(captured["frequency_input"].shape),
        "residual_input_dimensions": list(captured["residual_input"].shape),
        "spatial_resize": "Resize shorter edge to 256, bilinear interpolation",
        "spatial_crop": "CenterCrop(224, 224)",
        "resized_dimensions_before_crop": [resized.width, resized.height],
        "center_crop_box_xyxy": [
            crop_left,
            crop_top,
            crop_left + crop_size,
            crop_top + crop_size,
        ],
        "frequency_operation": (
            "RGB ToTensor at native resolution -> fft2 -> fftshift -> abs -> log1p -> "
            "per-channel min-max -> bilinear resize to 224x224 -> ImageNet normalization"
        ),
        "residual_operation": "normalized spatial tensor - 5x5 depthwise Gaussian blur (sigma=1.0)",
    }


def run_crop_diagnostics(production: Any, rgb_image: Image.Image, baseline: dict[str, Any]) -> dict[str, Any]:
    torch = production.torch
    transforms = production.transforms
    transform_steps = production.v5_spatial_transform.transforms
    resized = transform_steps[0](rgb_image)
    spatial_tail = transforms.Compose(transform_steps[2:])
    crop_size = 224
    max_left = resized.width - crop_size
    max_top = resized.height - crop_size
    center_left = int(round(max_left / 2.0))
    center_top = int(round(max_top / 2.0))
    crop_boxes = {
        "top": (center_left, 0, center_left + crop_size, crop_size),
        "bottom": (center_left, max_top, center_left + crop_size, max_top + crop_size),
        "left": (0, center_top, crop_size, center_top + crop_size),
        "right": (max_left, center_top, max_left + crop_size, center_top + crop_size),
    }
    frequency = production.v5_fft_transform(rgb_image).unsqueeze(0).to(production.DEVICE)
    results = {
        "production_center": {
            "raw_ai": baseline["raw_ai_probability"],
            "calibrated_ai": baseline["calibrated_ai_probability"],
            "prediction": baseline["prediction"],
        }
    }

    with torch.inference_mode():
        for crop_name, box in crop_boxes.items():
            crop = resized.crop(box)
            spatial = spatial_tail(crop).unsqueeze(0).to(production.DEVICE)
            residual = production.extract_noise_residual_v5(spatial)
            logits = production.v5_d_model(spatial, frequency, residual)
            result = probability_result(production, logits)
            results[crop_name] = {
                "raw_ai": result["raw_ai"],
                "calibrated_ai": result["calibrated_ai"],
                "prediction": result["prediction"],
            }

    calibrated_values = [item["calibrated_ai"] for item in results.values()]
    results["summary"] = {
        "calibrated_ai_range": max(calibrated_values) - min(calibrated_values),
        "prediction_changes_from_production_center": sum(
            item["prediction"] != baseline["prediction"]
            for name, item in results.items()
            if name != "summary"
        ),
        "note": "Diagnostic crop variants only; production preprocessing is unchanged.",
    }
    return results


def analyze_image(
    production: Any,
    path: Path,
    known_real: bool,
    include_crops: bool,
) -> dict[str, Any]:
    rgb_image, metadata = get_image_metadata(path)
    captured: dict[str, Any] = {}
    handles = attach_capture_hooks(production.v5_d_model, captured)
    try:
        v5_result = production.run_v5_d(rgb_image)
    finally:
        for handle in handles:
            handle.remove()

    captured["production_calibrated_ai"] = v5_result["calibrated_ai_probability"]
    captured["production_raw_ai"] = v5_result["raw_ai_probability"]
    ablations = run_branch_ablations(production, captured)
    preprocessing = describe_preprocessing(production, rgb_image, captured)

    strategy_e: dict[str, Any]
    try:
        strategy_e = production.forensic_pipeline.predict(rgb_image)
    except Exception as error:
        strategy_e = {"error": f"{type(error).__name__}: {error}"}

    calibrator = production.v5_d_calibrator
    calibration_path = str(calibrator.config_path.resolve()) if calibrator.config_path else None
    raw_ai = float(v5_result["raw_ai_probability"])
    calibrated_ai = float(v5_result["calibrated_ai_probability"])
    raw_label = "AI-GENERATED" if raw_ai >= CLASS_THRESHOLD_PERCENT else "REAL"
    calibrated_label = "AI-GENERATED" if calibrated_ai >= CLASS_THRESHOLD_PERCENT else "REAL"

    input_notes = []
    if metadata["image_mode"] != "RGB":
        input_notes.append(
            f"Source mode is {metadata['image_mode']}; production converts it to RGB."
        )
    if metadata["exif_orientation"] not in (None, 1):
        input_notes.append(
            "Non-default EXIF orientation is present; production uses Pillow RGB conversion without explicit EXIF transpose."
        )
    if preprocessing["resized_dimensions_before_crop"] != [224, 224]:
        kept_fraction = (224 * 224) / math.prod(preprocessing["resized_dimensions_before_crop"])
        input_notes.append(
            f"Production center crop retains {kept_fraction:.1%} of the resized canvas."
        )
    if not input_notes:
        input_notes.append("No mode/orientation/crop-geometry anomaly was flagged by these checks.")

    branch_effects = {
        name: float(result["delta_calibrated_ai"])
        for name, result in ablations.items()
    }
    largest_effect = max(branch_effects, key=lambda name: abs(branch_effects[name]))
    branch_interpretations = {}
    for branch, delta in branch_effects.items():
        if delta < 0:
            description = "Masking this branch lowered fused calibrated AI probability in this counterfactual."
        elif delta > 0:
            description = "Masking this branch raised fused calibrated AI probability in this counterfactual."
        else:
            description = "Masking this branch did not change the rounded calibrated AI probability."
        branch_interpretations[branch] = description

    all_branches_support_ai = all(delta < 0 for delta in branch_effects.values())
    interpretation = {
        "known_real_false_positive": bool(known_real and v5_result["prediction"] == "AI-GENERATED"),
        "largest_counterfactual_change": {
            "branch_ablated": largest_effect,
            "calibrated_ai_probability_change_percentage_points": branch_effects[largest_effect],
            "meaning": "Sensitivity of this fused classifier output to masking that branch's gated feature block; not causal proof.",
        },
        "branch_ablation_interpretations": branch_interpretations,
        "all_three_ablation_effects_lower_ai_probability": all_branches_support_ai,
        "branch_prediction_disagreement": (
            "UNKNOWN: V5-D has no independent branch classifiers; ablation outputs are not branch votes."
        ),
        "calibration_classification": (
            "preserved" if raw_label == calibrated_label else "different after percentage rounding; inspect logits near 50%"
        ),
        "calibration_reduces_confidence": max(calibrated_ai, 100.0 - calibrated_ai) < max(raw_ai, 100.0 - raw_ai),
        "input_preprocessing_observations": input_notes,
        "domain_shift_assessment": (
            "LIKELY COMPATIBLE with the documented source/domain generalization limitation; this image alone cannot prove domain shift."
            if known_real and v5_result["prediction"] == "AI-GENERATED"
            else "UNKNOWN from one image; the code diagnostic cannot establish domain shift."
        ),
        "causality_warning": "Single-image branch ablations and crop variants are diagnostic counterfactuals, not causal evidence.",
    }

    result = {
        "image": metadata,
        "preprocessing": preprocessing,
        "v5_d": {
            "model_name": v5_result["model_name"],
            "checkpoint_path_loaded": str(Path(production.V5_D_MODEL_PATH).resolve()),
            "strict_loading_succeeded": production.v5_d_model is not None,
            "device": str(production.DEVICE),
            "raw_real_probability": v5_result["raw_real_probability"],
            "raw_ai_probability": v5_result["raw_ai_probability"],
            "calibrated_real_probability": v5_result["calibrated_real_probability"],
            "calibrated_ai_probability": v5_result["calibrated_ai_probability"],
            "calibration_file_loaded": calibration_path,
            "calibration_loaded": calibrator.is_loaded,
            "calibration_method": v5_result["calibration_method"],
            "temperature": calibrator.T,
            "prediction": v5_result["prediction"],
            "confidence": v5_result["confidence"],
            "gate_statistics": v5_result["gate_weights"],
            "gate_statistics_note": "Feature-wise sigmoid gate means; not normalized branch contributions or percentages.",
        },
        "branch_ablation_diagnostics": ablations,
        "strategy_e_supporting_only": strategy_e,
        "interpretation": interpretation,
    }

    if include_crops:
        result["multi_crop_diagnostics"] = run_crop_diagnostics(production, rgb_image, v5_result)
    return result


def csv_row(result: dict[str, Any]) -> dict[str, Any]:
    image = result["image"]
    v5_result = result["v5_d"]
    row = {
        "filename": image["filename"],
        "width": image["width"],
        "height": image["height"],
        "aspect_ratio": image["aspect_ratio"],
        "format": image["format"],
        "spatial_gate": v5_result["gate_statistics"]["spatial"],
        "frequency_gate": v5_result["gate_statistics"]["frequency"],
        "residual_gate": v5_result["gate_statistics"]["residual"],
        "raw_ai": v5_result["raw_ai_probability"],
        "calibrated_ai": v5_result["calibrated_ai_probability"],
        "prediction": v5_result["prediction"],
        "confidence": v5_result["confidence"],
        "spatial_ablation_ai": result["branch_ablation_diagnostics"]["spatial"]["calibrated_ai"],
        "frequency_ablation_ai": result["branch_ablation_diagnostics"]["frequency"]["calibrated_ai"],
        "residual_ablation_ai": result["branch_ablation_diagnostics"]["residual"]["calibrated_ai"],
        "exif_orientation": image["exif_orientation"],
        "image_mode": image["image_mode"],
        "file_size_bytes": image["file_size_bytes"],
        "strategy_e_label": result["strategy_e_supporting_only"].get("final", {}).get("label"),
        "known_real_false_positive": result["interpretation"]["known_real_false_positive"],
    }
    crops = result.get("multi_crop_diagnostics")
    if crops:
        for name in ("production_center", "top", "bottom", "left", "right"):
            row[f"crop_{name}_raw_ai"] = crops[name]["raw_ai"]
            row[f"crop_{name}_calibrated_ai"] = crops[name]["calibrated_ai"]
            row[f"crop_{name}_prediction"] = crops[name]["prediction"]
        row["crop_calibrated_ai_range"] = crops["summary"]["calibrated_ai_range"]
    return row


def summarize(results: list[dict[str, Any]], known_real: bool) -> dict[str, Any]:
    probabilities = [float(item["v5_d"]["calibrated_ai_probability"]) for item in results]
    summary = {
        "images": len(results),
        "mean_calibrated_ai_probability": statistics.mean(probabilities) if probabilities else None,
        "median_calibrated_ai_probability": statistics.median(probabilities) if probabilities else None,
        "percentage_above_50": 100.0 * sum(value >= 50 for value in probabilities) / len(probabilities) if probabilities else None,
        "percentage_above_70": 100.0 * sum(value >= 70 for value in probabilities) / len(probabilities) if probabilities else None,
        "percentage_above_90": 100.0 * sum(value >= 90 for value in probabilities) / len(probabilities) if probabilities else None,
    }
    if known_real:
        summary["false_positive_rate_percent"] = summary["percentage_above_50"]
    else:
        summary["ai_positive_rate_percent_not_fpr"] = summary["percentage_above_50"]

    for branch in BRANCH_SLICES:
        deltas = [
            float(item["branch_ablation_diagnostics"][branch]["delta_calibrated_ai"])
            for item in results
        ]
        summary[f"{branch}_ablation_mean_delta_percentage_points"] = (
            statistics.mean(deltas) if deltas else None
        )
        summary[f"{branch}_ablation_decreased_ai_count"] = sum(delta < 0 for delta in deltas)
        summary[f"{branch}_ablation_increased_ai_count"] = sum(delta > 0 for delta in deltas)

    crop_ranges = [
        item["multi_crop_diagnostics"]["summary"]["calibrated_ai_range"]
        for item in results
        if "multi_crop_diagnostics" in item
    ]
    if crop_ranges:
        summary["mean_multi_crop_calibrated_ai_range"] = statistics.mean(crop_ranges)
        summary["images_with_multi_crop_prediction_change"] = sum(
            item["multi_crop_diagnostics"]["summary"]["prediction_changes_from_production_center"] > 0
            for item in results
        )
    return summary


def write_csv_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No successful image diagnostics are available to write.")
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def collect_images(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        return [source]
    pattern = "**/*" if recursive else "*"
    return sorted(
        path for path in source.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="One image or a directory of images")
    parser.add_argument("--csv", type=Path, help="New CSV output path; existing files are never overwritten")
    parser.add_argument("--recursive", action="store_true", help="Search image files recursively in directory mode")
    parser.add_argument("--known-real", action="store_true", help="Treat every supplied image as known genuine for FPR summaries")
    parser.add_argument("--multi-crop", action="store_true", help="Run diagnostic-only center/top/bottom/left/right crop comparisons")
    args = parser.parse_args()

    if not args.source.exists():
        parser.error(f"Source does not exist: {args.source}")
    if args.source.is_dir() and args.csv is None:
        parser.error("Directory mode requires --csv PATH so results are saved explicitly.")

    image_paths = collect_images(args.source, args.recursive)
    if not image_paths:
        parser.error("No supported image files found.")

    production = load_production_module()
    results = []
    failures = []
    for image_path in image_paths:
        try:
            results.append(
                analyze_image(production, image_path, args.known_real, args.multi_crop)
            )
        except Exception as error:
            failures.append({
                "filename": image_path.name,
                "error": f"{type(error).__name__}: {error}",
            })

    if args.csv:
        write_csv_exclusive(args.csv, [csv_row(result) for result in results])

    if args.source.is_file() and not args.csv and results:
        print(json.dumps(results[0], indent=2, ensure_ascii=True))
    else:
        print(json.dumps({
            "summary": summarize(results, args.known_real),
            "csv_path": str(args.csv.expanduser().resolve()) if args.csv else None,
            "failures": failures,
        }, indent=2, ensure_ascii=True))

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())