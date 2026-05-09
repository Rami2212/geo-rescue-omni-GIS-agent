"""GeoJSON export and utility helpers."""

import json
from typing import Optional


def export_results(
    route_geojson: Optional[dict],
    damage_geojson: Optional[dict],
    flood_geojson: Optional[dict] = None,
    blocked_geojson: Optional[dict] = None,
) -> dict:
    """Bundle all spatial results into a single exportable dict."""
    return {
        "drawn_damage_zones": damage_geojson,
        "local_safe_route": route_geojson,
        "live_flood_zone": flood_geojson,
        "blocked_roads": blocked_geojson,
    }


def geojson_feature_count(geojson: Optional[dict]) -> int:
    """Return the number of features in a GeoJSON FeatureCollection."""
    if not geojson:
        return 0
    return len(geojson.get("features", []))


def geojson_to_str(geojson: Optional[dict]) -> str:
    """Pretty-print a GeoJSON dict for display in the UI."""
    if not geojson:
        return "{}"
    return json.dumps(geojson, indent=2)
