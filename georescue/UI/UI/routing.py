from typing import List, Optional, Tuple

import networkx as nx
import osmnx as ox
import streamlit as st
from shapely.geometry import LineString, Polygon, shape, mapping
from shapely.ops import unary_union, linemerge

from .config import DEFAULT_SPEED_KMH
from .logging_setup import setup_logging

logger = setup_logging()


def add_edge_lengths_safe(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if hasattr(ox, "add_edge_lengths"):
        return ox.add_edge_lengths(graph)
    if hasattr(ox, "distance") and hasattr(ox.distance, "add_edge_lengths"):
        return ox.distance.add_edge_lengths(graph)
    logger.warning("osmnx edge-length helper not available; using raw graph.")
    return graph


def route_to_gdf_safe(graph: nx.MultiDiGraph, route_nodes: List[int]):
    if hasattr(ox, "routing") and hasattr(ox.routing, "route_to_gdf"):
        return ox.routing.route_to_gdf(graph, route_nodes)
    if hasattr(ox, "utils_graph") and hasattr(ox.utils_graph, "route_to_gdf"):
        return ox.utils_graph.route_to_gdf(graph, route_nodes)
    raise AttributeError("osmnx route_to_gdf helper not available.")


@st.cache_resource(show_spinner=True)
def fetch_sri_lanka_graph(center: Tuple[float, float], radius_km: float):
    logger.info("Fetching road network for Colombo, Sri Lanka")
    dist_m = int(radius_km * 1000)
    graph = ox.graph_from_point(center, dist=dist_m, network_type="drive", simplify=True)
    graph = add_edge_lengths_safe(graph)
    return graph


def parse_drawn_polygons(drawings: List[dict]) -> List[Polygon]:
    polygons: List[Polygon] = []
    for feature in drawings:
        try:
            geom = shape(feature.get("geometry", {}))
            if isinstance(geom, Polygon) and geom.is_valid and not geom.is_empty:
                polygons.append(geom)
        except Exception as exc:
            logger.warning("Invalid polygon skipped: %s", exc)
    return polygons


def calculate_safe_route(
    graph: nx.MultiDiGraph,
    start: Tuple[float, float],
    dest: Tuple[float, float],
    hazard_polygons: List[Polygon],
) -> Tuple[Optional[dict], Optional[LineString], dict, Optional[str]]:
    if not start or not dest:
        return None, None, {}, "Start and destination points are required."

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

    try:
        start_node = ox.distance.nearest_nodes(working_graph, start[1], start[0])
        dest_node = ox.distance.nearest_nodes(working_graph, dest[1], dest[0])
    except Exception as exc:
        logger.error("Failed to find nearest nodes: %s", exc)
        return None, None, {}, "Could not find nearby roads for the selected points."

    try:
        route_nodes = nx.shortest_path(
            working_graph, start_node, dest_node, weight="length"
        )
    except nx.NetworkXNoPath:
        return None, None, {}, "No safe route found (roads may be blocked)."
    except Exception as exc:
        logger.error("Routing error: %s", exc)
        return None, None, {}, "Routing failed. Please adjust inputs and retry."

    route_gdf = route_to_gdf_safe(working_graph, route_nodes)
    if route_gdf.empty:
        return None, None, {}, "Route calculation returned no segments."

    merged = linemerge(route_gdf.geometry.unary_union)
    if isinstance(merged, LineString):
        route_line = merged
    else:
        route_line = LineString([coord for line in merged.geoms for coord in line.coords])

    distance_m = float(route_gdf["length"].sum())
    speed_mps = (DEFAULT_SPEED_KMH * 1000) / 3600
    travel_time_min = distance_m / speed_mps / 60

    stats = {
        "distance_km": round(distance_m / 1000, 2),
        "travel_time_min": round(travel_time_min, 1),
        "blocked_edges": blocked_edges,
    }

    route_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Safe Route"},
                "geometry": mapping(route_line),
            }
        ],
    }

    return route_geojson, route_line, stats, None


