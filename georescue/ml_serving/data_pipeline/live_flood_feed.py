"""Fetch live precipitation from Open-Meteo and convert to a flood polygon GeoDataFrame."""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import requests
import shapely.geometry as geom

from .config import COLOMBO_CENTER_LAT, COLOMBO_CENTER_LON, PROCESSED_DIR, RAW_DIR

PRECIPITATION_THRESHOLDS = {
    "low":      (0.1,  2.0),
    "moderate": (2.0,  10.0),
    "high":     (10.0, 30.0),
    "extreme":  (30.0, float("inf")),
}

SEVERITY_RADIUS_KM = {
    "low": 1.0,
    "moderate": 2.5,
    "high": 5.0,
    "extreme": 10.0,
}


def fetch_precipitation(lat: float, lon: float) -> dict:
    resp = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "precipitation",
            "past_hours": 6,
            "forecast_hours": 3,
            "timezone": "UTC",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def classify_severity(precip_mm: float) -> str:
    for severity, (lo, hi) in PRECIPITATION_THRESHOLDS.items():
        if lo <= precip_mm < hi:
            return severity
    return "extreme"


def build_flood_polygon(lat: float, lon: float, radius_km: float) -> geom.Polygon:
    deg_lat = radius_km / 111.0
    deg_lon = radius_km / (111.0 * math.cos(math.radians(lat)))
    return geom.Point(lon, lat).buffer(max(deg_lat, deg_lon), resolution=16)


def fetch_live_flood(
    lat: float = COLOMBO_CENTER_LAT,
    lon: float = COLOMBO_CENTER_LON,
    save_dir: "Path | None" = None,
) -> "tuple[str, dict]":
    out_dir = Path(save_dir) if save_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    data = fetch_precipitation(lat, lon)
    precip_values = data.get("hourly", {}).get("precipitation", [0.0])
    max_precip = max(precip_values) if precip_values else 0.0
    severity = classify_severity(max_precip)
    radius_km = SEVERITY_RADIUS_KM[severity]

    polygon = build_flood_polygon(lat, lon, radius_km)
    gdf = gpd.GeoDataFrame(
        [{"severity": severity, "radius_km": radius_km, "max_precip_mm": max_precip}],
        geometry=[polygon],
        crs="EPSG:4326",
    )
    polygon_path = str(out_dir / "live_flood_polygon.geojson")
    gdf.to_file(polygon_path, driver="GeoJSON")

    snapshot = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "latitude": lat,
        "longitude": lon,
        "max_precip_mm": max_precip,
        "severity": severity,
        "radius_km": radius_km,
    }
    (RAW_DIR / "live_weather_snapshot.json").write_text(json.dumps(snapshot, indent=2))
    return polygon_path, snapshot
