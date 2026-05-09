"""PyTorch Dataset that loads (satellite image, GIS label) pairs for Qwen2-VL fine-tuning."""
import json
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

SEVERITY_TO_INT = {"low": 0, "moderate": 1, "high": 2, "extreme": 3}

FLOOD_ANALYSIS_PROMPT = (
    "Analyze this satellite image for flood and disaster impact. "
    "Identify flooded areas, damaged roads, and affected infrastructure. "
    "Return a JSON object with keys: "
    "severity (low/medium/high/critical), "
    "findings (text description of observed damage), "
    "affected_zones (list of polygon coordinate arrays [[lon, lat], ...])."
)


class FloodAnalysisDataset(Dataset):
    def __init__(
        self,
        dataset_index: "str | Path",
        image_size: int = 1024,
        require_image: bool = False,
    ) -> None:
        index_path = Path(dataset_index)
        self.samples = json.loads(index_path.read_text())
        if require_image:
            self.samples = [s for s in self.samples if s.get("satellite_image_path")]
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> "dict[str, Any]":
        sample = self.samples[idx]

        training_label = sample.get("training_label") or self._build_label(sample)
        item: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "prompt": FLOOD_ANALYSIS_PROMPT,
            "expected_output": json.dumps(training_label),
            "severity_int": SEVERITY_TO_INT.get(sample.get("severity", "low"), 0),
            "metadata": {
                "severity": sample.get("severity"),
                "affected_roads": sample.get("affected_roads"),
                "affected_length_m": sample.get("affected_length_m"),
                "max_precip_mm": sample.get("max_precip_mm"),
            },
        }

        img_path = sample.get("satellite_image_path")
        if img_path and Path(img_path).exists():
            img = Image.open(img_path).convert("RGB")
            if max(img.size) > self.image_size:
                img.thumbnail((self.image_size, self.image_size), Image.LANCZOS)
            item["image"] = img

        return item

    def _build_label(self, sample: dict) -> dict:
        severity_map = {"low": "low", "moderate": "medium", "high": "high", "extreme": "critical"}
        return {
            "severity": severity_map.get(sample.get("severity", "low"), "low"),
            "findings": (
                f"{sample.get('affected_roads', 0)} roads affected by flooding "
                f"({sample.get('affected_length_m', 0):.0f}m total length). "
                f"Precipitation: {sample.get('max_precip_mm', 0):.1f}mm. "
                f"Flood radius: {sample.get('radius_km', 0):.1f}km."
            ),
            "affected_zones": [],
        }
