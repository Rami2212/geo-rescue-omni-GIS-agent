"""Orchestrates one full flood-analysis cycle."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .live_flood_feed import generate_live_flood_polygon
from .flood_overlay import analyze_flood_impact
from .routing import plan_safe_route, save_route_geojson

logger = logging.getLogger(__name__)

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
STATUS_FILE = PROCESSED_DIR / "latest_status.json"


def run_cycle() -> dict:
    """Run one live flood-analysis cycle and return a status summary dict."""
    flood_polygon_path, snapshot = generate_live_flood_polygon()

    blocked_roads_path, saved_polygon_path, stats = analyze_flood_impact(
        flood_polygon_path=flood_polygon_path
    )

    route_path, route_gdf = plan_safe_route(
        remove_blocked=True,
        blocked_roads_path=blocked_roads_path,
    )

    latest_route_path = PROCESSED_DIR / "latest_route.geojson"
    save_route_geojson(route_gdf, latest_route_path)

    route_length = float(route_gdf["length_m"].sum()) if not route_gdf.empty else None

    status = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "severity": snapshot.get("derived_severity"),
        "affected_roads": stats.get("total_affected_roads"),
        "total_affected_length_m": stats.get("total_affected_length_m"),
        "route_length_m": route_length,
        "route_path": str(route_path),
        "latest_route_path": str(latest_route_path),
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with STATUS_FILE.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=2)

    logger.info("[GeoRescue] Cycle complete — severity=%s, affected_roads=%s",
                status["severity"], status["affected_roads"])
    return status
