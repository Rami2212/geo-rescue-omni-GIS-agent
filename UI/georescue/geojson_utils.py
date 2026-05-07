from typing import Optional


def export_geojson(route_geojson: Optional[dict], damage_geojson: Optional[dict]) -> dict:
    return {
        "damage_zones": damage_geojson,
        "safe_route": route_geojson,
    }

