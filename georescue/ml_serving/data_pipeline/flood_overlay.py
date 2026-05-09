"""Overlay a flood polygon on a road network to identify affected/blocked roads."""
from pathlib import Path

from .config import PROCESSED_DIR
from .export_geojson import load_geojson, save_geojson


def analyze_flood_impact(
    road_network_path: "str | Path",
    flood_polygon_path: "str | Path",
    save_dir: "Path | None" = None,
) -> dict:
    out_dir = Path(save_dir) if save_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    roads = load_geojson(road_network_path)
    flood = load_geojson(flood_polygon_path)

    if roads.crs != flood.crs:
        flood = flood.to_crs(roads.crs)

    flood_union = flood.geometry.union_all()
    intersecting = roads[roads.geometry.intersects(flood_union)].copy()

    blocked_path = str(out_dir / "blocked_roads_flood.geojson")
    if not intersecting.empty:
        save_geojson(intersecting, blocked_path)

    roads_m = roads.to_crs("EPSG:3857")
    intersecting_m = intersecting.to_crs("EPSG:3857") if not intersecting.empty else intersecting

    return {
        "total_roads": len(roads),
        "affected_roads": len(intersecting),
        "total_length_m": float(roads_m.geometry.length.sum()),
        "affected_length_m": float(intersecting_m.geometry.length.sum()) if not intersecting.empty else 0.0,
        "blocked_roads_path": blocked_path,
    }
