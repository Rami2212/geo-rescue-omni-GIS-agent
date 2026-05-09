"""Generate a near-real-time flood polygon from live weather data."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import geopandas as gpd
import requests
from shapely.geometry import Point


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

LIVE_FLOOD_POLYGON_FILE = "live_flood_polygon.geojson"
LIVE_WEATHER_SNAPSHOT_FILE = "live_weather_snapshot.json"

# Colombo defaults
DEFAULT_LAT = 6.9271
DEFAULT_LON = 79.8612


def fetch_live_precipitation(lat: float, lon: float) -> Dict:
    """Fetch latest hourly precipitation values from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "past_hours": 6,
        "forecast_hours": 3,
        "timezone": "UTC",
    }

    logger.info("Fetching live weather data from Open-Meteo...")
    response = requests.get(url, params=params, timeout=25)
    response.raise_for_status()

    payload = response.json()
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    precip = hourly.get("precipitation", [])

    if not times or not precip:
        raise ValueError("Open-Meteo response did not include hourly precipitation data.")

    return {
        "lat": lat,
        "lon": lon,
        "source": "open-meteo",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "hourly_time_utc": times,
        "hourly_precip_mm": precip,
    }


def precipitation_to_radius_km(precip_mm_values: list[float]) -> Tuple[float, str]:
    """Map recent precipitation intensity to a flood impact radius."""
    recent_window = precip_mm_values[-6:] if len(precip_mm_values) >= 6 else precip_mm_values
    mm_6h = float(sum(recent_window))

    if mm_6h < 5:
        return 0.8, "low"
    if mm_6h < 20:
        return 1.5, "moderate"
    if mm_6h < 40:
        return 2.5, "high"
    return 3.5, "extreme"


def build_flood_polygon_geojson(
    lat: float,
    lon: float,
    radius_km: float,
    severity: str,
    weather_snapshot: Dict,
    output_path: Path,
) -> Path:
    """Create and save a flood polygon in EPSG:4326 based on computed radius."""
    center = gpd.GeoDataFrame({"id": [1]}, geometry=[Point(lon, lat)], crs="EPSG:4326")
    center_m = center.to_crs("EPSG:32644")

    radius_m = radius_km * 1000.0
    polygon_m = center_m.geometry.buffer(radius_m)

    flood_gdf = gpd.GeoDataFrame(
        {
            "id": [1],
            "source": ["open-meteo"],
            "severity": [severity],
            "radius_km": [radius_km],
            "fetched_at_utc": [weather_snapshot["fetched_at_utc"]],
        },
        geometry=polygon_m,
        crs="EPSG:32644",
    ).to_crs("EPSG:4326")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    flood_gdf.to_file(output_path, driver="GeoJSON")
    return output_path


def save_weather_snapshot(snapshot: Dict, output_path: Path) -> Path:
    """Save raw weather response subset for traceability."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    return output_path


def generate_live_flood_polygon(lat: float = DEFAULT_LAT, lon: float = DEFAULT_LON) -> Tuple[Path, Dict]:
    """Fetch live weather and produce a flood polygon file for overlay analysis."""
    snapshot = fetch_live_precipitation(lat=lat, lon=lon)
    radius_km, severity = precipitation_to_radius_km(snapshot["hourly_precip_mm"])
    snapshot["derived_radius_km"] = radius_km
    snapshot["derived_severity"] = severity

    flood_polygon_path = build_flood_polygon_geojson(
        lat=lat,
        lon=lon,
        radius_km=radius_km,
        severity=severity,
        weather_snapshot=snapshot,
        output_path=PROCESSED_DIR / LIVE_FLOOD_POLYGON_FILE,
    )
    weather_snapshot_path = save_weather_snapshot(snapshot, RAW_DIR / LIVE_WEATHER_SNAPSHOT_FILE)

    logger.info("Live flood input generated.")
    logger.info("  - Severity: %s", severity)
    logger.info("  - Radius: %.2f km", radius_km)
    logger.info("  - Flood polygon: %s", flood_polygon_path)
    logger.info("  - Snapshot: %s", weather_snapshot_path)

    return flood_polygon_path, snapshot


if __name__ == "__main__":
    generate_live_flood_polygon()