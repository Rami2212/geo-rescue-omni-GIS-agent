from __future__ import annotations

import io
import json
import logging
import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
from shapely.geometry import MultiPoint, Polygon, mapping

from .hf_client import hf_token
from .schemas import BackendRunRequest

logger = logging.getLogger(__name__)

DEFAULT_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


class FloodVisionAgent:
    """Extract flooded-area polygons from satellite imagery."""

    def __init__(self) -> None:
        package_root = Path(__file__).resolve().parents[1]
        self.base_model_name = os.getenv("GEORESCUE_VISION_BASE_MODEL", DEFAULT_VISION_MODEL)
        self.model_name = os.getenv("GEORESCUE_VISION_MODEL", self.base_model_name)
        self.adapter_dir = Path(
            os.getenv("GEORESCUE_VISION_ADAPTER_DIR", str(package_root / "adapter"))
        )
        self.mode = os.getenv("GEORESCUE_VISION_MODE", "local_lora").lower()
        self._model = None
        self._processor = None

    def analyze(self, request: BackendRunRequest) -> dict:
        image_bytes = request.uploaded_image_bytes or self._latest_image_bytes(
            request.realtime_image_dir
        )
        if not image_bytes:
            return self._empty_detection(request, "No satellite image was provided.")

        model_result = self._try_qwen_lora_detection(image_bytes, request)
        if model_result:
            return model_result

        return self._heuristic_detection(image_bytes, request)

    def _latest_image_bytes(self, image_dir: Optional[str]) -> Optional[bytes]:
        search_dir = Path(image_dir) if image_dir else self._default_realtime_dir()
        if not search_dir.exists() or not search_dir.is_dir():
            return None

        candidates = [
            p
            for p in search_dir.iterdir()
            if p.is_file() and p.suffix.lower() in DEFAULT_IMAGE_EXTENSIONS
        ]
        if not candidates:
            return None
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        try:
            return latest.read_bytes()
        except OSError as exc:
            logger.warning("Could not read realtime image %s: %s", latest, exc)
            return None

    def _default_realtime_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "ml_serving" / "data" / "raw"

    def _try_qwen_lora_detection(
        self, image_bytes: bytes, request: BackendRunRequest
    ) -> Optional[dict]:
        if self.mode in {"off", "heuristic"}:
            return None

        try:
            if not self._adapter_available():
                raise FileNotFoundError(
                    f"LoRA adapter files were not found in {self.adapter_dir}"
                )
            prompt = self._vision_prompt()
            raw_text = self._run_local_qwen_lora(image_bytes, prompt)
            parsed = self._parse_model_json(raw_text)
            if not parsed:
                return None
            return self._model_output_to_geojson(parsed, request, raw_text)
        except Exception as exc:
            logger.warning("Qwen2.5-VL + local LoRA flood detection unavailable: %s", exc)
            if self.mode == "require":
                return self._empty_detection(
                    request, f"Qwen2.5-VL + local LoRA inference failed: {exc}"
                )
            return None

    def _adapter_available(self) -> bool:
        if not self.adapter_dir.exists() or not self.adapter_dir.is_dir():
            return False
        expected = {"adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"}
        return any((self.adapter_dir / name).exists() for name in expected)

    def _load_local_qwen_lora(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        from peft import PeftModel
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        token = hf_token()
        local_files_only = os.getenv("GEORESCUE_VISION_LOCAL_FILES_ONLY", "false").lower() == "true"
        model_kwargs = {
            "device_map": os.getenv("GEORESCUE_VISION_DEVICE_MAP", "auto"),
            "torch_dtype": os.getenv("GEORESCUE_VISION_TORCH_DTYPE", "auto"),
            "local_files_only": local_files_only,
        }
        if token:
            model_kwargs["token"] = token

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.base_model_name,
            **model_kwargs,
        )
        model = PeftModel.from_pretrained(model, str(self.adapter_dir))

        processor_kwargs = {"local_files_only": local_files_only}
        if token:
            processor_kwargs["token"] = token
        self._processor = AutoProcessor.from_pretrained(
            self.base_model_name,
            **processor_kwargs,
        )
        self._model = model

    def _run_local_qwen_lora(self, image_bytes: bytes, prompt: str) -> str:
        from PIL import Image

        self._load_local_qwen_lora()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], padding=True, return_tensors="pt"
        )
        inputs = inputs.to(self._model.device)
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=int(os.getenv("GEORESCUE_VISION_MAX_TOKENS", "768")),
            do_sample=False,
        )
        generated = output_ids[:, inputs.input_ids.shape[1] :]
        return self._processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def _vision_prompt(self) -> str:
        return (
            "Detect visible flooded water in this satellite/aerial image. "
            "Return only JSON. Use this schema: "
            '{"severity":"low|moderate|high|extreme","confidence":0.0,'
            '"findings":"short operational description",'
            '"polygons":[[[x,y],[x,y],...]]}. '
            "Coordinates must be normalized image coordinates from 0 to 1, "
            "ordered around the flooded boundary. Use polygons, not rectangles."
        )

    def _parse_model_json(self, raw_text: str) -> Optional[dict]:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _model_output_to_geojson(
        self, parsed: dict, request: BackendRunRequest, raw_text: str
    ) -> dict:
        polygons = parsed.get("polygons") or parsed.get("flood_polygons") or []
        features = []
        for idx, polygon in enumerate(polygons, start=1):
            geo_poly = self._normalized_polygon_to_geo(polygon, request)
            if geo_poly:
                features.append(self._feature(idx, geo_poly, parsed, "qwen2.5-vl-lora"))

        if not features:
            return self._empty_detection(
                request, "Qwen2.5-VL returned no flood polygon.", raw_text=raw_text
            )

        return {
            "severity": parsed.get("severity", "unknown"),
            "confidence": parsed.get("confidence"),
            "findings": parsed.get("findings", "Flooded area polygon extracted."),
            "model": self.model_name,
            "base_model": self.base_model_name,
            "adapter_dir": str(self.adapter_dir),
            "method": "qwen2.5-vl-local-lora",
            "georeferencing": self._georef_note(request),
            "flood_geojson": {"type": "FeatureCollection", "features": features},
            "raw_model_response": raw_text,
        }

    def _heuristic_detection(self, image_bytes: bytes, request: BackendRunRequest) -> dict:
        try:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            image.thumbnail((256, 256))
            arr = np.asarray(image).astype(np.float32)
        except Exception as exc:
            return self._empty_detection(request, f"Could not inspect image: {exc}")

        red = arr[:, :, 0]
        green = arr[:, :, 1]
        blue = arr[:, :, 2]
        brightness = (red + green + blue) / 3
        water_like = ((blue + green) > (red * 1.55)) & (brightness < 185)
        dark_water = (brightness < 75) & (blue >= red * 0.8) & (green >= red * 0.8)
        mask = water_like | dark_water

        coverage = float(mask.mean())
        if coverage < 0.01:
            return self._empty_detection(
                request,
                "No strong flooded-water signal found; routing will use a conservative center polygon.",
            )

        points = np.argwhere(mask)
        normalized = self._mask_outline(points, arr.shape[0], arr.shape[1])
        geo_poly = self._normalized_polygon_to_geo(normalized, request)
        if not geo_poly:
            return self._empty_detection(request, "Detected flood mask was too small.")

        severity = self._severity_from_coverage(coverage)
        parsed = {
            "severity": severity,
            "confidence": round(min(0.35 + coverage * 2.0, 0.82), 2),
            "findings": (
                f"Detected an irregular flooded area covering about "
                f"{coverage * 100:.1f}% of the uploaded image."
            ),
        }
        return {
            **parsed,
            "model": self.model_name,
            "base_model": self.base_model_name,
            "adapter_dir": str(self.adapter_dir),
            "method": "image-color-polygon-fallback",
            "georeferencing": self._georef_note(request),
            "flood_geojson": {
                "type": "FeatureCollection",
                "features": [self._feature(1, geo_poly, parsed, "heuristic")],
            },
        }

    def _mask_outline(self, points: np.ndarray, height: int, width: int) -> list[list[float]]:
        center = points.mean(axis=0)
        buckets: dict[int, tuple[float, np.ndarray]] = {}
        for point in points[:: max(1, len(points) // 3000)]:
            dy = float(point[0] - center[0])
            dx = float(point[1] - center[1])
            angle = (math.atan2(dy, dx) + math.tau) % math.tau
            bucket = int(angle / math.tau * 28)
            radius = dx * dx + dy * dy
            if bucket not in buckets or radius > buckets[bucket][0]:
                buckets[bucket] = (radius, point)

        outline = []
        for _, point in sorted(buckets.values(), key=lambda item: math.atan2(item[1][0] - center[0], item[1][1] - center[1])):
            outline.append([float(point[1] / max(width - 1, 1)), float(point[0] / max(height - 1, 1))])

        if len(outline) < 3:
            hull = MultiPoint(
                [(float(p[1] / max(width - 1, 1)), float(p[0] / max(height - 1, 1))) for p in points]
            ).convex_hull
            if isinstance(hull, Polygon):
                outline = [[float(x), float(y)] for x, y in hull.exterior.coords[:-1]]
        return outline

    def _normalized_polygon_to_geo(
        self, polygon: list, request: BackendRunRequest
    ) -> Optional[Polygon]:
        if not polygon or len(polygon) < 3:
            return None

        center_lat, center_lon = request.map_center or request.start
        radius_km = float(request.graph_radius_km or 12.0)
        lat_span = max(radius_km * 0.45 / 111.0, 0.01)
        lon_span = max(
            radius_km * 0.45 / (111.0 * max(math.cos(math.radians(center_lat)), 0.2)),
            0.01,
        )

        coords = []
        for point in polygon:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            x = min(max(float(point[0]), 0.0), 1.0)
            y = min(max(float(point[1]), 0.0), 1.0)
            lon = center_lon + (x - 0.5) * lon_span
            lat = center_lat - (y - 0.5) * lat_span
            coords.append((lon, lat))

        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area <= 0:
            return None
        return poly.simplify(0.00005, preserve_topology=True)

    def _feature(self, idx: int, polygon: Polygon, parsed: dict, method: str) -> dict:
        return {
            "type": "Feature",
            "properties": {
                "name": f"Detected Flood Polygon {idx}",
                "severity": parsed.get("severity", "unknown"),
                "confidence": parsed.get("confidence"),
                "source": method,
            },
            "geometry": mapping(polygon),
        }

    def _empty_detection(
        self, request: BackendRunRequest, message: str, raw_text: str | None = None
    ) -> dict:
        center_lat, center_lon = request.map_center or request.start
        radius_km = float(request.graph_radius_km or 12.0)
        lat_delta = max(radius_km * 0.08 / 111.0, 0.003)
        lon_delta = max(
            radius_km * 0.08 / (111.0 * max(math.cos(math.radians(center_lat)), 0.2)),
            0.003,
        )
        coords = [
            (center_lon - lon_delta * 0.5, center_lat - lat_delta),
            (center_lon + lon_delta * 0.7, center_lat - lat_delta * 0.6),
            (center_lon + lon_delta, center_lat + lat_delta * 0.2),
            (center_lon + lon_delta * 0.2, center_lat + lat_delta),
            (center_lon - lon_delta * 0.8, center_lat + lat_delta * 0.5),
            (center_lon - lon_delta, center_lat - lat_delta * 0.3),
        ]
        polygon = Polygon(coords)
        parsed = {"severity": "unknown", "confidence": 0.0, "findings": message}
        result = {
            **parsed,
            "model": self.model_name,
            "base_model": self.base_model_name,
            "adapter_dir": str(self.adapter_dir),
            "method": "conservative-placeholder-polygon",
            "georeferencing": self._georef_note(request),
            "flood_geojson": {
                "type": "FeatureCollection",
                "features": [self._feature(1, polygon, parsed, "fallback")],
            },
        }
        if raw_text:
            result["raw_model_response"] = raw_text
        return result

    def _severity_from_coverage(self, coverage: float) -> str:
        if coverage >= 0.35:
            return "extreme"
        if coverage >= 0.18:
            return "high"
        if coverage >= 0.06:
            return "moderate"
        return "low"

    def _georef_note(self, request: BackendRunRequest) -> str:
        if request.map_center:
            return (
                "Image-space flood polygon projected around the current map center. "
                "Provide georeferenced imagery bounds for survey-grade placement."
            )
        return (
            "Image-space flood polygon projected around the start point because "
            "no map center or image georeference was provided."
        )
