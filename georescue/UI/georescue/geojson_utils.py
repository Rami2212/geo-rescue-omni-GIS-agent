from typing import Optional, List

from shapely.geometry import Polygon, MultiPolygon, shape


def export_geojson(route_geojson: Optional[dict], damage_geojson: Optional[dict]) -> dict:
    return {
        "damage_zones": damage_geojson,
        "safe_route": route_geojson,
    }


def geojson_to_polygons(geojson: Optional[dict]) -> List[Polygon]:
    if not geojson:
        return []

    polygons: List[Polygon] = []
    for feature in geojson.get("features", []):
        try:
            geom = shape(feature.get("geometry", {}))
        except Exception:
            continue
        if isinstance(geom, Polygon):
            polygons.append(geom)
        elif isinstance(geom, MultiPolygon):
            polygons.extend(list(geom.geoms))
    return polygons


def extract_route_stats(route_geojson: Optional[dict]) -> dict:
    if not route_geojson:
        return {}

    feature = None
    for item in route_geojson.get("features", []):
        feature = item
        break
    if feature is None:
        return {}

    return feature.get("properties", {})
