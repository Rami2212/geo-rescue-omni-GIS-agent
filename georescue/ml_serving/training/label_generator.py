"""Convert GIS analysis outputs into structured training labels for Qwen2-VL."""
import json
from pathlib import Path

import geopandas as gpd

SEVERITY_MAP = {
    "low": "low",
    "moderate": "medium",
    "high": "high",
    "extreme": "critical",
}


def geojson_to_zone_coordinates(geojson_path: "str | Path") -> "list[list[list[float]]]":
    gdf = gpd.read_file(str(geojson_path))
    if gdf.crs and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    zones = []
    for geom in gdf.geometry:
        if geom is None:
            continue
        if geom.geom_type == "Polygon":
            zones.append([list(c) for c in geom.exterior.coords])
        elif geom.geom_type == "MultiPolygon":
            for poly in geom.geoms:
                zones.append([list(c) for c in poly.exterior.coords])
    return zones


def build_training_label(sample: dict, base_dir: "Path | None" = None) -> dict:
    label = {
        "severity": SEVERITY_MAP.get(sample.get("severity", "low"), "low"),
        "findings": (
            f"{sample.get('affected_roads', 0)} roads affected by flooding "
            f"({sample.get('affected_length_m', 0):.0f}m total length). "
            f"Precipitation: {sample.get('max_precip_mm', 0):.1f}mm. "
            f"Flood radius: {sample.get('radius_km', 0):.1f}km."
        ),
        "affected_zones": [],
    }

    flood_path = sample.get("flood_polygon_path")
    if flood_path:
        resolved = Path(flood_path)
        if not resolved.is_absolute() and base_dir:
            resolved = base_dir / resolved
        if resolved.exists():
            label["affected_zones"] = geojson_to_zone_coordinates(resolved)

    return label


def enrich_dataset_labels(dataset_index: "str | Path") -> None:
    """Load dataset_index.json, add training_label field to each sample, and re-save."""
    index_path = Path(dataset_index)
    base_dir = index_path.parent
    samples = json.loads(index_path.read_text())

    for sample in samples:
        sample["training_label"] = build_training_label(sample, base_dir=base_dir)

    index_path.write_text(json.dumps(samples, indent=2))
    print(f"Enriched {len(samples)} samples with training labels → {index_path}")
