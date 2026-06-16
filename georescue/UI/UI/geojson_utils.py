from typing import Optional


def export_geojson(
    route_geojson: Optional[dict],
    damage_geojson: Optional[dict],
    flood_geojson: Optional[dict] = None,
    blocked_geojson: Optional[dict] = None,
    alternative_routes_geojson: Optional[list] = None,
    realtime_exports: Optional[dict] = None,
) -> dict:
    return {
        "damage_zones": damage_geojson,
        "flooded_areas": flood_geojson,
        "blocked_roads": blocked_geojson,
        "safe_route": route_geojson,
        "alternative_routes": alternative_routes_geojson or [],
        "realtime_exports": realtime_exports or {},
    }

