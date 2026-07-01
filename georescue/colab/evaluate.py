#!/usr/bin/env python3
"""Evaluate a Qwen2-VL + LoRA adapter on flood annotation tiles.

This script:
1. Loads `Qwen/Qwen2-VL-7B-Instruct`
2. Applies the LoRA adapter from `georescue/colab/adapter`
3. Runs inference on images from `georescue/colab/satellite_images_data`
4. Saves annotated overlays, raw predictions, and summary reports
5. Computes timing and label-based metrics when ground truth exists

Ground-truth discovery:
- Metadata columns: `mask`, `mask_path`, `label`, `label_path`, `geojson`,
  `geojson_path`, `flood_polygon_path`, `ground_truth`, `gt_path`
- Sidecar files beside each image:
  `<stem>.geojson`, `<stem>.json`, `<stem>_flood.geojson`, `<stem>_flood.json`,
  `<stem>_mask.png`, `<stem>_mask.jpg`, `<stem>_mask.jpeg`

Current repository note:
- `metadata.csv` in `satellite_images_data` contains image bounds and filenames,
  but no flood ground truth files were found during implementation.
- The script still produces predictions, overlays, timings, and reports.
- Accuracy / F1 / IoU / Dice / Precision / Recall become available automatically
  once GT mask or polygon files are added.
"""

from __future__ import annotations

import argparse
import csv
import html
import importlib.metadata
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from peft import PeftModel
from PIL import Image, ImageColor, ImageDraw, ImageFont
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

try:
    from qwen_vl_utils import process_vision_info
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit(
        "Missing dependency `qwen-vl-utils`. Install requirements from "
        "`requirements.txt` in this `colab` folder before running evaluate.py."
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent if len(SCRIPT_DIR.parents) >= 2 else SCRIPT_DIR
DEFAULT_IMAGES_DIR = SCRIPT_DIR / "satellite_images_data"
DEFAULT_METADATA = DEFAULT_IMAGES_DIR / "metadata.csv"
DEFAULT_ADAPTER_DIR = SCRIPT_DIR / "adapter"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "output"
DEFAULT_MODEL_NAME = "Qwen/Qwen2-VL-7B-Instruct"
DEFAULT_MAX_IMAGE_SIZE = 768
DEFAULT_MIN_PIXELS = 256 * 28 * 28
DEFAULT_MAX_PIXELS = 640 * 28 * 28
LOW_MEMORY_RETRY_IMAGE_SIZES = (512, 448, 384, 336)
LOW_MEMORY_RETRY_MAX_TOKENS = (256, 192, 128, 96)

PROMPT = """Analyze this satellite image for flooding.
Return only valid JSON with this exact schema:
{
  "severity": "low|medium|high|critical",
  "findings": "short plain-English summary",
  "affected_zones": [
    [[lon, lat], [lon, lat], [lon, lat], [lon, lat]]
  ]
}

Rules:
- Focus on flooded areas visible in the image.
- `affected_zones` must contain polygons in longitude/latitude order.
- If no flooded area is visible, return an empty list for `affected_zones`.
- Do not include markdown fences or extra commentary.
"""

GT_METADATA_COLUMNS = (
    "mask",
    "mask_path",
    "label",
    "label_path",
    "geojson",
    "geojson_path",
    "flood_polygon_path",
    "ground_truth",
    "gt_path",
)

SIDE_CAR_GT_PATTERNS = (
    "{stem}.geojson",
    "{stem}.json",
    "{stem}_flood.geojson",
    "{stem}_flood.json",
    "{stem}_mask.png",
    "{stem}_mask.jpg",
    "{stem}_mask.jpeg",
)

@dataclass
class ImageRecord:
    idx: int
    image_path: Path
    bbox: tuple[float, float, float, float]  # lon_min, lat_min, lon_max, lat_max
    metadata: dict[str, Any]
    ground_truth_path: Path | None


@dataclass
class PredictionResult:
    idx: int
    image_name: str
    image_path: str
    severity: str
    findings: str
    affected_zone_count: int
    predicted_flood_pixels: int
    predicted_flood_ratio: float
    inference_time_sec: float
    parse_ok: bool
    raw_output: str
    prediction_json: dict[str, Any]
    ground_truth_available: bool
    ground_truth_path: str | None
    gt_flood_pixels: int | None
    pixel_accuracy: float | None
    precision: float | None
    recall: float | None
    specificity: float | None
    f1_score: float | None
    iou: float | None
    dice: float | None
    confusion: dict[str, int] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate flood annotations with Qwen2-VL + LoRA adapter.")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR, help="Directory containing input images.")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="CSV metadata with image file and bbox columns.")
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR, help="LoRA adapter directory.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to store outputs.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="Base Hugging Face model name.")
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images processed.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Generation length.")
    parser.add_argument("--device-map", type=str, default="auto", help="Transformers device_map value.")
    parser.add_argument("--dtype", type=str, default="auto", choices=("auto", "float16", "bfloat16", "float32"), help="Torch dtype for model load.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip images with an existing prediction JSON output.")
    parser.add_argument("--max-image-size", type=int, default=DEFAULT_MAX_IMAGE_SIZE, help="Resize images so the largest side is at most this many pixels.")
    parser.add_argument("--min-pixels", type=int, default=DEFAULT_MIN_PIXELS, help="Processor min_pixels hint for Qwen2-VL.")
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Processor max_pixels hint for Qwen2-VL.")
    parser.add_argument("--load-in-4bit", action="store_true", help="Load the base model in 4-bit quantized mode when bitsandbytes is available.")
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> str | torch.dtype:
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def load_model_and_processor(
    model_name: str,
    adapter_dir: Path,
    device_map: str,
    dtype_name: str,
    min_pixels: int,
    max_pixels: int,
    load_in_4bit: bool,
):
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")
    offload_dir = SCRIPT_DIR / "offload"
    offload_dir.mkdir(parents=True, exist_ok=True)

    quantization_config = None
    if load_in_4bit:
        ensure_compatible_bitsandbytes()
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    print(f"[evaluate] Loading base model: {model_name}")
    model_load_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "offload_folder": str(offload_dir),
    }
    if quantization_config is not None:
        model_load_kwargs["quantization_config"] = quantization_config
        print("[evaluate] Using 4-bit quantized loading for lower GPU memory usage.")
    else:
        model_load_kwargs["torch_dtype"] = resolve_dtype(dtype_name)

    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        **model_load_kwargs,
    )

    print(f"[evaluate] Applying adapter: {adapter_dir}")
    disable_incompatible_torchao()
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        offload_folder=str(offload_dir),
    )
    if quantization_config is None:
        try:
            model = model.merge_and_unload()
            print("[evaluate] Adapter merged into base model.")
        except torch.OutOfMemoryError:
            print(
                "[evaluate] merge_and_unload() ran out of GPU memory. "
                "Continuing with the LoRA adapter attached for inference."
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        print("[evaluate] Keeping LoRA adapter attached because merge is not used with 4-bit loading.")
    model.eval()

    processor_source = str(adapter_dir) if (adapter_dir / "tokenizer.json").exists() else model_name
    processor = AutoProcessor.from_pretrained(
        processor_source,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    return model, processor


def ensure_compatible_bitsandbytes() -> None:
    try:
        version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit(
            "4-bit loading requires bitsandbytes>=0.46.1. "
            "Run `pip install -U bitsandbytes>=0.46.1` in Colab, then restart the runtime."
        ) from exc

    def _version_tuple(raw: str) -> tuple[int, ...]:
        parts = []
        for piece in raw.split("."):
            digits = "".join(ch for ch in piece if ch.isdigit())
            if not digits:
                break
            parts.append(int(digits))
        return tuple(parts)

    if _version_tuple(version) < (0, 46, 1):
        raise SystemExit(
            f"Found bitsandbytes {version}, but 4-bit loading requires >=0.46.1. "
            "Run `pip install -U bitsandbytes>=0.46.1` in Colab, then restart the runtime."
        )


def disable_incompatible_torchao() -> None:
    """Work around Colab environments that ship an unsupported torchao version.

    PEFT checks torchao availability while creating LoRA layers. Some Colab
    runtimes include `torchao==0.10.x`, which causes PEFT to raise an import
    error even though torchao is optional for this evaluation script.
    """
    try:
        version = importlib.metadata.version("torchao")
    except importlib.metadata.PackageNotFoundError:
        return

    def _version_tuple(raw: str) -> tuple[int, ...]:
        parts = []
        for piece in raw.split("."):
            digits = "".join(ch for ch in piece if ch.isdigit())
            if not digits:
                break
            parts.append(int(digits))
        return tuple(parts)

    if _version_tuple(version) >= (0, 16, 0):
        return

    print(
        f"[evaluate] Detected incompatible torchao {version}; "
        "disabling torchao integration for PEFT."
    )
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.torchao as peft_torchao

        peft_import_utils.is_torchao_available = lambda: False
        peft_torchao.is_torchao_available = lambda: False
    except Exception as exc:
        print(f"[evaluate] Warning: failed to patch torchao integration: {exc}")


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    dirs = {
        "root": output_dir,
        "annotated": output_dir / "annotated_images",
        "predictions": output_dir / "predictions",
        "masks": output_dir / "predicted_masks",
        "reports": output_dir / "reports",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_records(images_dir: Path, metadata_path: Path, max_images: int | None) -> list[ImageRecord]:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    records: list[ImageRecord] = []
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader):
            image_name = row.get("file")
            if not image_name:
                continue

            image_path = images_dir / image_name
            if not image_path.exists():
                print(f"[evaluate] Skipping missing image: {image_path}")
                continue

            bbox = (
                float(row["lon_min"]),
                float(row["lat_min"]),
                float(row["lon_max"]),
                float(row["lat_max"]),
            )
            gt_path = discover_ground_truth_path(row, images_dir, image_path)
            records.append(
                ImageRecord(
                    idx=int(row.get("idx", row_idx)),
                    image_path=image_path,
                    bbox=bbox,
                    metadata=row,
                    ground_truth_path=gt_path,
                )
            )
            if max_images is not None and len(records) >= max_images:
                break

    if not records:
        raise RuntimeError(f"No valid image records found in {metadata_path}")
    return records


def discover_ground_truth_path(row: dict[str, Any], images_dir: Path, image_path: Path) -> Path | None:
    for column in GT_METADATA_COLUMNS:
        raw_value = row.get(column)
        if not raw_value:
            continue
        candidate = Path(raw_value)
        if not candidate.is_absolute():
            candidate = images_dir / raw_value
        if candidate.exists():
            return candidate

    for pattern in SIDE_CAR_GT_PATTERNS:
        candidate = image_path.with_name(pattern.format(stem=image_path.stem))
        if candidate.exists():
            return candidate
    return None


def load_rgb_image(image_path: Path, max_image_size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_image_size:
        image.thumbnail((max_image_size, max_image_size), Image.Resampling.LANCZOS)
    return image


def run_inference(
    model,
    processor,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> tuple[str, float]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - start

    generated = output_ids[:, inputs.input_ids.shape[1]:]
    output_text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return output_text, elapsed


def clear_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def build_retry_plan(initial_image_size: int, initial_max_new_tokens: int) -> list[tuple[int, int]]:
    plan: list[tuple[int, int]] = [(initial_image_size, initial_max_new_tokens)]
    for size, tokens in zip(LOW_MEMORY_RETRY_IMAGE_SIZES, LOW_MEMORY_RETRY_MAX_TOKENS):
        candidate = (min(initial_image_size, size), min(initial_max_new_tokens, tokens))
        if candidate not in plan:
            plan.append(candidate)
    return plan


def parse_prediction(raw_text: str) -> tuple[dict[str, Any], bool]:
    cleaned = raw_text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif cleaned.startswith("```") and cleaned.count("```") >= 2:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            raise ValueError("Prediction is not a JSON object")
    except Exception:
        parsed = {"severity": "unknown", "findings": "", "affected_zones": []}
        return parsed, False

    parsed.setdefault("severity", "unknown")
    parsed.setdefault("findings", "")
    parsed.setdefault("affected_zones", [])
    if not isinstance(parsed["affected_zones"], list):
        parsed["affected_zones"] = []
    return parsed, True


def normalize_polygon(zone: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(zone, list):
        return points

    for item in zone:
        if (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and _is_number(item[0])
            and _is_number(item[1])
        ):
            points.append((float(item[0]), float(item[1])))

    if len(points) >= 3 and points[0] != points[-1]:
        points.append(points[0])
    return points


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def geojson_to_polygons(payload: Any) -> list[list[tuple[float, float]]]:
    polygons: list[list[tuple[float, float]]] = []
    if not isinstance(payload, dict):
        return polygons

    payload_type = payload.get("type")
    if payload_type == "FeatureCollection":
        for feature in payload.get("features", []):
            polygons.extend(geojson_to_polygons(feature))
        return polygons
    if payload_type == "Feature":
        return geojson_to_polygons(payload.get("geometry", {}))
    if payload_type == "Polygon":
        coords = payload.get("coordinates", [])
        if coords and isinstance(coords[0], list):
            poly = normalize_polygon(coords[0])
            if len(poly) >= 4:
                polygons.append(poly)
        return polygons
    if payload_type == "MultiPolygon":
        for poly_coords in payload.get("coordinates", []):
            if poly_coords and isinstance(poly_coords[0], list):
                poly = normalize_polygon(poly_coords[0])
                if len(poly) >= 4:
                    polygons.append(poly)
    return polygons


def polygons_to_mask(
    polygons: Iterable[list[tuple[float, float]]],
    width: int,
    height: int,
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    lon_min, lat_min, lon_max, lat_max = bbox
    lon_span = max(lon_max - lon_min, 1e-12)
    lat_span = max(lat_max - lat_min, 1e-12)

    for polygon in polygons:
        pixel_points: list[tuple[float, float]] = []
        for lon, lat in polygon:
            x = ((lon - lon_min) / lon_span) * (width - 1)
            y = ((lat_max - lat) / lat_span) * (height - 1)
            pixel_points.append((x, y))
        if len(pixel_points) >= 3:
            draw.polygon(pixel_points, fill=255)

    return np.array(mask, dtype=np.uint8) > 0


def load_ground_truth_mask(record: ImageRecord, image_size: tuple[int, int]) -> np.ndarray | None:
    if record.ground_truth_path is None:
        return None

    gt_path = record.ground_truth_path
    if gt_path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        mask_img = Image.open(gt_path).convert("L").resize(image_size, Image.Resampling.NEAREST)
        return np.array(mask_img, dtype=np.uint8) > 127

    if gt_path.suffix.lower() in {".json", ".geojson"}:
        data = json.loads(gt_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "affected_zones" in data:
            polygons = [normalize_polygon(zone) for zone in data.get("affected_zones", [])]
        else:
            polygons = geojson_to_polygons(data)
        polygons = [poly for poly in polygons if len(poly) >= 4]
        if not polygons:
            return np.zeros((image_size[1], image_size[0]), dtype=bool)
        return polygons_to_mask(polygons, image_size[0], image_size[1], record.bbox)

    return None


def compute_binary_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[dict[str, int], dict[str, float | None]]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    tn = int(np.logical_and(~pred, ~gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    def safe_div(num: float, den: float) -> float | None:
        return None if den == 0 else num / den

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1_score = safe_div(2 * tp, (2 * tp) + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    dice = safe_div(2 * tp, (2 * tp) + fp + fn)

    metrics = {
        "pixel_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "iou": iou,
        "dice": dice,
    }
    confusion = {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return confusion, metrics


def render_overlay(
    image: Image.Image,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None,
    title_lines: list[str],
) -> Image.Image:
    base = image.convert("RGBA")
    pred_overlay = colored_overlay(pred_mask, base.size, "#ff3b30", alpha=90)
    blended = Image.alpha_composite(base, pred_overlay)

    if gt_mask is not None:
        gt_overlay = colored_overlay(gt_mask, base.size, "#34c759", alpha=70)
        blended = Image.alpha_composite(blended, gt_overlay)

    canvas = Image.new("RGBA", (blended.width, blended.height + 80), (255, 255, 255, 255))
    canvas.alpha_composite(blended, (0, 80))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for i, line in enumerate(title_lines):
        draw.text((10, 10 + i * 16), line, fill=(0, 0, 0, 255), font=font)
    return canvas.convert("RGB")


def colored_overlay(mask: np.ndarray, size: tuple[int, int], color: str, alpha: int) -> Image.Image:
    rgba = Image.new("RGBA", size, ImageColor.getrgb(color) + (0,))
    overlay_mask = Image.fromarray(np.where(mask, alpha, 0).astype(np.uint8), mode="L")
    rgba.putalpha(overlay_mask)
    return rgba


def save_prediction_artifacts(
    result: PredictionResult,
    prediction_json: dict[str, Any],
    raw_text: str,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None,
    image: Image.Image,
    dirs: dict[str, Path],
) -> None:
    stem = Path(result.image_name).stem

    prediction_payload = {
        "image_name": result.image_name,
        "image_path": result.image_path,
        "severity": result.severity,
        "findings": result.findings,
        "affected_zone_count": result.affected_zone_count,
        "predicted_flood_pixels": result.predicted_flood_pixels,
        "predicted_flood_ratio": result.predicted_flood_ratio,
        "inference_time_sec": result.inference_time_sec,
        "parse_ok": result.parse_ok,
        "ground_truth_available": result.ground_truth_available,
        "ground_truth_path": result.ground_truth_path,
        "prediction": prediction_json,
        "raw_output": raw_text,
        "metrics": {
            "pixel_accuracy": result.pixel_accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "specificity": result.specificity,
            "f1_score": result.f1_score,
            "iou": result.iou,
            "dice": result.dice,
            "confusion": result.confusion,
        },
    }
    (dirs["predictions"] / f"{stem}.json").write_text(json.dumps(prediction_payload, indent=2), encoding="utf-8")

    Image.fromarray((pred_mask.astype(np.uint8) * 255), mode="L").save(dirs["masks"] / f"{stem}_pred_mask.png")
    if gt_mask is not None:
        Image.fromarray((gt_mask.astype(np.uint8) * 255), mode="L").save(dirs["masks"] / f"{stem}_gt_mask.png")

    overlay = render_overlay(
        image=image,
        pred_mask=pred_mask,
        gt_mask=gt_mask,
        title_lines=[
            f"Image: {result.image_name}",
            f"Severity: {result.severity} | Zones: {result.affected_zone_count} | Parse OK: {result.parse_ok}",
            f"Inference: {result.inference_time_sec:.3f}s | Pred flood ratio: {result.predicted_flood_ratio:.4f}",
            "Legend: red=prediction, green=ground truth" if gt_mask is not None else "Legend: red=prediction",
        ],
    )
    overlay.save(dirs["annotated"] / f"{stem}_annotated.png")


def aggregate_results(results: list[PredictionResult], report_generation_time_sec: float, total_runtime_sec: float) -> dict[str, Any]:
    metrics_available = [r for r in results if r.ground_truth_available and r.pixel_accuracy is not None]
    parse_success_count = sum(1 for r in results if r.parse_ok)
    zone_positive_count = sum(1 for r in results if r.affected_zone_count > 0)

    summary: dict[str, Any] = {
        "dataset": {
            "total_images": len(results),
            "images_with_ground_truth": len(metrics_available),
            "images_without_ground_truth": len(results) - len(metrics_available),
        },
        "runtime": {
            "total_runtime_sec": total_runtime_sec,
            "report_generation_time_sec": report_generation_time_sec,
            "total_inference_time_sec": sum(r.inference_time_sec for r in results),
            "avg_inference_time_sec": _mean([r.inference_time_sec for r in results]),
            "median_inference_time_sec": _median([r.inference_time_sec for r in results]),
            "min_inference_time_sec": _min_or_none([r.inference_time_sec for r in results]),
            "max_inference_time_sec": _max_or_none([r.inference_time_sec for r in results]),
            "throughput_images_per_sec": (len(results) / sum(r.inference_time_sec for r in results)) if results and sum(r.inference_time_sec for r in results) > 0 else None,
        },
        "prediction_quality": {
            "parse_success_rate": safe_ratio(parse_success_count, len(results)),
            "images_with_predicted_flood": zone_positive_count,
            "predicted_flood_rate": safe_ratio(zone_positive_count, len(results)),
            "avg_predicted_zone_count": _mean([r.affected_zone_count for r in results]),
            "avg_predicted_flood_ratio": _mean([r.predicted_flood_ratio for r in results]),
        },
        "label_based_metrics": {},
    }

    if metrics_available:
        summary["label_based_metrics"] = {
            "accuracy": _mean([r.pixel_accuracy for r in metrics_available]),
            "precision": _mean([r.precision for r in metrics_available]),
            "recall": _mean([r.recall for r in metrics_available]),
            "specificity": _mean([r.specificity for r in metrics_available]),
            "f1_score": _mean([r.f1_score for r in metrics_available]),
            "iou": _mean([r.iou for r in metrics_available]),
            "dice": _mean([r.dice for r in metrics_available]),
            "avg_gt_flood_pixels": _mean([float(r.gt_flood_pixels or 0) for r in metrics_available]),
            "aggregated_confusion": {
                "tp": sum((r.confusion or {}).get("tp", 0) for r in metrics_available),
                "tn": sum((r.confusion or {}).get("tn", 0) for r in metrics_available),
                "fp": sum((r.confusion or {}).get("fp", 0) for r in metrics_available),
                "fn": sum((r.confusion or {}).get("fn", 0) for r in metrics_available),
            },
        }
    else:
        summary["label_based_metrics"] = {
            "accuracy": None,
            "precision": None,
            "recall": None,
            "specificity": None,
            "f1_score": None,
            "iou": None,
            "dice": None,
            "note": "Ground-truth masks/polygons were not found, so label-based metrics could not be computed.",
        }

    severity_distribution: dict[str, int] = {}
    for item in results:
        severity_distribution[item.severity] = severity_distribution.get(item.severity, 0) + 1
    summary["prediction_quality"]["severity_distribution"] = severity_distribution

    return summary


def write_reports(results: list[PredictionResult], summary: dict[str, Any], dirs: dict[str, Path]) -> None:
    summary_path = dirs["reports"] / "metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows = []
    for r in results:
        row = {
            "idx": r.idx,
            "image_name": r.image_name,
            "severity": r.severity,
            "zone_count": r.affected_zone_count,
            "parse_ok": r.parse_ok,
            "ground_truth_available": r.ground_truth_available,
            "predicted_flood_pixels": r.predicted_flood_pixels,
            "predicted_flood_ratio": round(r.predicted_flood_ratio, 6),
            "inference_time_sec": round(r.inference_time_sec, 6),
            "pixel_accuracy": _round_or_none(r.pixel_accuracy),
            "precision": _round_or_none(r.precision),
            "recall": _round_or_none(r.recall),
            "specificity": _round_or_none(r.specificity),
            "f1_score": _round_or_none(r.f1_score),
            "iou": _round_or_none(r.iou),
            "dice": _round_or_none(r.dice),
            "findings": r.findings,
        }
        rows.append(row)

    import pandas as pd

    pd.DataFrame(rows).to_csv(dirs["reports"] / "per_image_metrics.csv", index=False)

    md_lines = [
        "# Flood Annotation Evaluation Report",
        "",
        f"- Total images: {summary['dataset']['total_images']}",
        f"- Images with ground truth: {summary['dataset']['images_with_ground_truth']}",
        f"- Images without ground truth: {summary['dataset']['images_without_ground_truth']}",
        f"- Average inference time: {_fmt(summary['runtime']['avg_inference_time_sec'])} sec",
        f"- Median inference time: {_fmt(summary['runtime']['median_inference_time_sec'])} sec",
        f"- Report generation time: {_fmt(summary['runtime']['report_generation_time_sec'])} sec",
        f"- Parse success rate: {_fmt(summary['prediction_quality']['parse_success_rate'])}",
        "",
        "## Label-Based Metrics",
        "",
        f"- Accuracy: {_fmt(summary['label_based_metrics'].get('accuracy'))}",
        f"- Precision: {_fmt(summary['label_based_metrics'].get('precision'))}",
        f"- Recall: {_fmt(summary['label_based_metrics'].get('recall'))}",
        f"- Specificity: {_fmt(summary['label_based_metrics'].get('specificity'))}",
        f"- F1 Score: {_fmt(summary['label_based_metrics'].get('f1_score'))}",
        f"- IoU: {_fmt(summary['label_based_metrics'].get('iou'))}",
        f"- Dice: {_fmt(summary['label_based_metrics'].get('dice'))}",
        "",
        "## Prediction Distribution",
        "",
    ]
    for severity, count in summary["prediction_quality"]["severity_distribution"].items():
        md_lines.append(f"- {severity}: {count}")
    note = summary["label_based_metrics"].get("note")
    if note:
        md_lines.extend(["", f"> {note}"])

    (dirs["reports"] / "report.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (dirs["reports"] / "report.html").write_text(build_html_report(results, summary), encoding="utf-8")


def evaluate_record(
    record: ImageRecord,
    model,
    processor,
    max_new_tokens: int,
    max_image_size: int,
    dirs: dict[str, Path],
    skip_existing: bool,
) -> PredictionResult:
    stem = record.image_path.stem
    prediction_path = dirs["predictions"] / f"{stem}.json"
    if skip_existing and prediction_path.exists():
        cached = json.loads(prediction_path.read_text(encoding="utf-8"))
        metrics = cached.get("metrics", {})
        return PredictionResult(
            idx=record.idx,
            image_name=record.image_path.name,
            image_path=str(record.image_path),
            severity=cached.get("severity", "unknown"),
            findings=cached.get("findings", ""),
            affected_zone_count=int(cached.get("affected_zone_count", 0)),
            predicted_flood_pixels=int(cached.get("predicted_flood_pixels", 0)),
            predicted_flood_ratio=float(cached.get("predicted_flood_ratio", 0.0)),
            inference_time_sec=float(cached.get("inference_time_sec", 0.0)),
            parse_ok=bool(cached.get("parse_ok", False)),
            raw_output=cached.get("raw_output", ""),
            prediction_json=cached.get("prediction", {}),
            ground_truth_available=bool(cached.get("ground_truth_available", False)),
            ground_truth_path=cached.get("ground_truth_path"),
            gt_flood_pixels=None,
            pixel_accuracy=metrics.get("pixel_accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            specificity=metrics.get("specificity"),
            f1_score=metrics.get("f1_score"),
            iou=metrics.get("iou"),
            dice=metrics.get("dice"),
            confusion=metrics.get("confusion"),
        )

    raw_output = ""
    inference_time_sec = 0.0
    image = None
    last_oom: torch.OutOfMemoryError | None = None

    for retry_image_size, retry_max_new_tokens in build_retry_plan(max_image_size, max_new_tokens):
        try:
            clear_cuda_memory()
            image = load_rgb_image(record.image_path, max_image_size=retry_image_size)
            if retry_image_size != max_image_size or retry_max_new_tokens != max_new_tokens:
                print(
                    f"[evaluate] Retrying {record.image_path.name} with "
                    f"max_image_size={retry_image_size}, max_new_tokens={retry_max_new_tokens}"
                )
            raw_output, inference_time_sec = run_inference(
                model,
                processor,
                image,
                PROMPT,
                retry_max_new_tokens,
            )
            last_oom = None
            break
        except torch.OutOfMemoryError as exc:
            last_oom = exc
            print(
                f"[evaluate] CUDA OOM on {record.image_path.name} "
                f"(max_image_size={retry_image_size}, max_new_tokens={retry_max_new_tokens})."
            )
            clear_cuda_memory()

    if last_oom is not None or image is None:
        raise RuntimeError(
            f"Unable to run inference for {record.image_path.name} even after low-memory retries."
        ) from last_oom

    parsed, parse_ok = parse_prediction(raw_output)
    severity = str(parsed.get("severity", "unknown")).strip().lower()
    findings = str(parsed.get("findings", "")).strip()
    polygons = [normalize_polygon(zone) for zone in parsed.get("affected_zones", [])]
    polygons = [poly for poly in polygons if len(poly) >= 4]
    pred_mask = polygons_to_mask(polygons, image.width, image.height, record.bbox)
    gt_mask = load_ground_truth_mask(record, image.size)

    confusion = None
    metric_values = {
        "pixel_accuracy": None,
        "precision": None,
        "recall": None,
        "specificity": None,
        "f1_score": None,
        "iou": None,
        "dice": None,
    }
    gt_flood_pixels = None
    if gt_mask is not None:
        confusion, metric_values = compute_binary_metrics(pred_mask, gt_mask)
        gt_flood_pixels = int(gt_mask.sum())

    result = PredictionResult(
        idx=record.idx,
        image_name=record.image_path.name,
        image_path=str(record.image_path),
        severity=severity,
        findings=findings,
        affected_zone_count=len(polygons),
        predicted_flood_pixels=int(pred_mask.sum()),
        predicted_flood_ratio=float(pred_mask.mean()),
        inference_time_sec=inference_time_sec,
        parse_ok=parse_ok,
        raw_output=raw_output,
        prediction_json=parsed,
        ground_truth_available=gt_mask is not None,
        ground_truth_path=str(record.ground_truth_path) if record.ground_truth_path else None,
        gt_flood_pixels=gt_flood_pixels,
        pixel_accuracy=metric_values["pixel_accuracy"],
        precision=metric_values["precision"],
        recall=metric_values["recall"],
        specificity=metric_values["specificity"],
        f1_score=metric_values["f1_score"],
        iou=metric_values["iou"],
        dice=metric_values["dice"],
        confusion=confusion,
    )

    save_prediction_artifacts(result, parsed, raw_output, pred_mask, gt_mask, image, dirs)
    return result


def safe_ratio(num: float, den: float) -> float | None:
    return None if den == 0 else num / den


def _mean(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    return None if not filtered else statistics.mean(filtered)


def _median(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    return None if not filtered else statistics.median(filtered)


def _min_or_none(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    return None if not filtered else min(filtered)


def _max_or_none(values: list[float | None]) -> float | None:
    filtered = [float(v) for v in values if v is not None]
    return None if not filtered else max(filtered)


def _round_or_none(value: float | None, ndigits: int = 6) -> float | None:
    return None if value is None else round(float(value), ndigits)


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def build_html_report(results: list[PredictionResult], summary: dict[str, Any]) -> str:
    def h(value: Any) -> str:
        return html.escape(str(value))

    rows = []
    for r in results:
        rows.append(
            "<tr>"
            f"<td>{h(r.idx)}</td>"
            f"<td>{h(r.image_name)}</td>"
            f"<td>{h(r.severity)}</td>"
            f"<td>{h(r.affected_zone_count)}</td>"
            f"<td>{h(_fmt(r.inference_time_sec))}</td>"
            f"<td>{h(_fmt(r.pixel_accuracy))}</td>"
            f"<td>{h(_fmt(r.f1_score))}</td>"
            f"<td>{h(_fmt(r.iou))}</td>"
            f"<td>{h('yes' if r.ground_truth_available else 'no')}</td>"
            "</tr>"
        )

    label_metrics = summary["label_based_metrics"]
    return (
        "<!DOCTYPE html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <title>Flood Annotation Evaluation Report</title>\n"
        "  <style>\n"
        "    body { font-family: Arial, sans-serif; margin: 24px; color: #222; }\n"
        "    h1, h2 { color: #0b5394; }\n"
        "    table { border-collapse: collapse; width: 100%; margin-top: 12px; }\n"
        "    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 14px; }\n"
        "    th { background: #f3f6fa; }\n"
        "    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0; }\n"
        "    .card { border: 1px solid #d8e1ea; border-radius: 8px; padding: 12px; background: #fbfdff; }\n"
        "    .muted { color: #666; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>Flood Annotation Evaluation Report</h1>\n"
        "  <div class=\"cards\">\n"
        f"    <div class=\"card\"><strong>Total images</strong><br>{h(summary['dataset']['total_images'])}</div>\n"
        f"    <div class=\"card\"><strong>Avg inference time</strong><br>{h(_fmt(summary['runtime']['avg_inference_time_sec']))} sec</div>\n"
        f"    <div class=\"card\"><strong>F1 Score</strong><br>{h(_fmt(label_metrics.get('f1_score')))}</div>\n"
        f"    <div class=\"card\"><strong>IoU</strong><br>{h(_fmt(label_metrics.get('iou')))}</div>\n"
        f"    <div class=\"card\"><strong>Accuracy</strong><br>{h(_fmt(label_metrics.get('accuracy')))}</div>\n"
        f"    <div class=\"card\"><strong>Report generation time</strong><br>{h(_fmt(summary['runtime']['report_generation_time_sec']))} sec</div>\n"
        "  </div>\n"
        f"  <p class=\"muted\">{h(label_metrics.get('note', ''))}</p>\n"
        "  <h2>Per-image results</h2>\n"
        "  <table>\n"
        "    <thead>\n"
        "      <tr>\n"
        "        <th>Idx</th>\n"
        "        <th>Image</th>\n"
        "        <th>Severity</th>\n"
        "        <th>Zones</th>\n"
        "        <th>Inference (sec)</th>\n"
        "        <th>Accuracy</th>\n"
        "        <th>F1</th>\n"
        "        <th>IoU</th>\n"
        "        <th>GT</th>\n"
        "      </tr>\n"
        "    </thead>\n"
        "    <tbody>\n"
        f"      {''.join(rows)}\n"
        "    </tbody>\n"
        "  </table>\n"
        "</body>\n"
        "</html>\n"
    )


def main() -> None:
    args = parse_args()
    start_total = time.perf_counter()

    print(f"[evaluate] Repo root: {REPO_ROOT}")
    print(f"[evaluate] Images dir: {args.images_dir}")
    print(f"[evaluate] Metadata CSV: {args.metadata}")
    print(f"[evaluate] Adapter dir: {args.adapter_dir}")
    print(f"[evaluate] Output dir: {args.output_dir}")

    dirs = ensure_output_dirs(args.output_dir)
    records = load_records(args.images_dir, args.metadata, args.max_images)
    model, processor = load_model_and_processor(
        args.model_name,
        args.adapter_dir,
        args.device_map,
        args.dtype,
        args.min_pixels,
        args.max_pixels,
        args.load_in_4bit,
    )

    results: list[PredictionResult] = []
    for i, record in enumerate(records, start=1):
        print(f"[evaluate] ({i}/{len(records)}) Processing {record.image_path.name}")
        result = evaluate_record(
            record=record,
            model=model,
            processor=processor,
            max_new_tokens=args.max_new_tokens,
            max_image_size=args.max_image_size,
            dirs=dirs,
            skip_existing=args.skip_existing,
        )
        results.append(result)

    report_start = time.perf_counter()
    total_runtime_sec = time.perf_counter() - start_total
    summary = aggregate_results(results, report_generation_time_sec=0.0, total_runtime_sec=total_runtime_sec)
    write_reports(results, summary, dirs)
    report_generation_time_sec = time.perf_counter() - report_start

    final_runtime_sec = time.perf_counter() - start_total
    summary = aggregate_results(results, report_generation_time_sec=report_generation_time_sec, total_runtime_sec=final_runtime_sec)
    write_reports(results, summary, dirs)

    print("[evaluate] Done.")
    print(f"[evaluate] Annotated images: {dirs['annotated']}")
    print(f"[evaluate] Prediction JSONs: {dirs['predictions']}")
    print(f"[evaluate] Reports: {dirs['reports']}")
    print("[evaluate] Key metrics:")
    print(f"  - Accuracy: {_fmt(summary['label_based_metrics'].get('accuracy'))}")
    print(f"  - F1 Score: {_fmt(summary['label_based_metrics'].get('f1_score'))}")
    print(f"  - IoU: {_fmt(summary['label_based_metrics'].get('iou'))}")
    print(f"  - Avg inference time: {_fmt(summary['runtime']['avg_inference_time_sec'])} sec")
    print(f"  - Report generation time: {_fmt(summary['runtime']['report_generation_time_sec'])} sec")


if __name__ == "__main__":
    main()
