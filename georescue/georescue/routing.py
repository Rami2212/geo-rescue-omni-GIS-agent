"""
Local OSMnx + NetworkX routing for GeoRescue.

Provides a cached road-graph fetch and a hazard-aware shortest-path
calculator that works entirely offline (no GIS API required).
"""

import logging
from typing import List, Optional, Tuple

import networkx as nx
import osmnx as ox
import streamlit as st
from shapely.geometry import LineString, Polygon, mapping, shape
from shapely.ops import linemerge, unary_union

from georescue.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OSMnx helpers — version-safe wrappers
# ---------------------------------------------------------------------------

def _add_edge_lengths(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    for attr in ("add_edge_lengths", "distance.add_edge_lengths"):
        try:
            parts = attr.split(".")
            obj = ox
            for part in parts:
                obj = getattr(obj, part)
            return obj(graph)
        except AttributeError:
            continue
    logger.warning("osmnx add_edge_lengths not found; using raw graph.")
    return graph


def _route_to_gdf(graph: nx.MultiDiGraph, route_nodes: List[int]):
    for attr in ("routing.route_to_gdf", "utils_graph.route_to_gdf"):
        try:
            parts = attr.split(".")
            obj = ox
            for part in parts:
                obj = getattr(obj, part)
            return obj(graph, route_nodes)
        except AttributeError:
            continue
    raise AttributeError("osmnx route_to_gdf not available in this version.")


# ---------------------------------------------------------------------------
# Cached road-graph fetch
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Fetching road network from OpenStreetMap…")
def fetch_road_graph(
    center: Tuple[float, float],
    radius_km: float,
) -> nx.MultiDiGraph:
    """
    Download and cache the driving road graph centred on `center`.

    The graph is cached for the lifetime of the Streamlit server process;
    changing center or radius_km produces a new cache entry.
    """
    dist_m = int(radius_km * 1000)
    logger.info("Fetching OSMnx graph: centre=%s radius=%dm", center, dist_m)
    graph = ox.graph_from_point(
        center, dist=dist_m, network_type="drive", simplify=True
    )
    return _add_edge_lengths(graph)


# ---------------------------------------------------------------------------
# Polygon parsing from Folium draw events
# ---------------------------------------------------------------------------

def parse_drawn_polygons(drawings: List[dict]) -> List[Polygon]:
    """Convert raw Folium draw GeoJSON features to Shapely Polygon objects."""
    polygons: List[Polygon] = []
    for feature in drawings:
        try:
            geom = shape(feature.get("geometry", {}))
            if isinstance(geom, Polygon) and geom.is_valid and not geom.is_empty:
                polygons.append(geom)
        except Exception as exc:
            logger.debug("Skipping invalid drawn polygon: %s", exc)
    return polygons


# ---------------------------------------------------------------------------
# Route calculation
# ---------------------------------------------------------------------------

def calculate_safe_route(
    graph: nx.MultiDiGraph,
    start: Tuple[float, float],
    dest: Tuple[float, float],
    hazard_polygons: Optional[List[Polygon]] = None,
) -> Tuple[Optional[dict], Optional[LineString], dict, Optional[str]]:
    """
    Compute the shortest drive route from `start` to `dest` avoiding hazards.

    Args:
        graph:           OSMnx MultiDiGraph.
        start:           (lat, lon) of route origin.
        dest:            (lat, lon) of route destination.
        hazard_polygons: Optional list of Shapely Polygons to avoid.

    Returns:
        (route_geojson, route_line, stats_dict, error_message)
        On failure: (None, None, {}, error_message)
    """
    cfg = get_settings()
    working_graph = graph
    blocked_edges = 0

    if hazard_polygons:
        edges_gdf = ox.graph_to_gdfs(
            graph, nodes=False, edges=True, fill_edge_geometry=True
        )
        hazard_union = unary_union(hazard_polygons)
        blocked = edges_gdf[edges_gdf.intersects(hazard_union)]
        blocked_edges = len(blocked)
        if blocked_edges:
            working_graph = graph.copy()
            working_graph.remove_edges_from(blocked.index)
            logger.info("Removed %d edges inside hazard zone", blocked_edges)

    try:
        start_node = ox.distance.nearest_nodes(working_graph, start[1], start[0])
        dest_node = ox.distance.nearest_nodes(working_graph, dest[1], dest[0])
    except Exception as exc:
        logger.error("nearest_nodes failed: %s", exc)
        return None, None, {}, "Could not locate nearby roads for the selected points."

    try:
        route_nodes = nx.shortest_path(working_graph, start_node, dest_node, weight="length")
    except nx.NetworkXNoPath:
        return None, None, {}, "No safe route found — all paths may be blocked."
    except nx.NodeNotFound as exc:
        return None, None, {}, f"Routing node error: {exc}"
    except Exception as exc:
        logger.error("Routing error: %s", exc)
        return None, None, {}, f"Routing failed: {exc}"

    route_gdf = _route_to_gdf(working_graph, route_nodes)
    if route_gdf.empty:
        return None, None, {}, "Route calculation returned no segments."

    merged = linemerge(route_gdf.geometry.unary_union)
    if isinstance(merged, LineString):
        route_line = merged
    else:
        route_line = LineString(
            [coord for line in merged.geoms for coord in line.coords]
        )

    distance_m = float(route_gdf["length"].sum())
    speed_mps = (cfg.default_speed_kmh * 1000) / 3600
    travel_time_min = round(distance_m / speed_mps / 60, 1)

    stats = {
        "distance_km": round(distance_m / 1000, 2),
        "travel_time_min": travel_time_min,
        "blocked_edges": blocked_edges,
        "route_segments": len(route_nodes) - 1,
    }

    route_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Local Safe Route",
                    **stats,
                },
                "geometry": mapping(route_line),
            }
        ],
    }

    return route_geojson, route_line, stats, None
