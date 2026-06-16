from __future__ import annotations

import logging
import math
from typing import Iterable, Optional

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, mapping, shape
from shapely.ops import linemerge, unary_union

from .schemas import BackendRunRequest

logger = logging.getLogger(__name__)

DEFAULT_SPEED_KMH = 30.0


class RouteAgent:
    def build_routes(self, request: BackendRunRequest, flood_geojson: dict) -> dict:
        flood_polygons = self._extract_polygons(flood_geojson)
        try:
            graph = self._fetch_graph(request)
            return self._graph_routes(request, graph, flood_polygons)
        except Exception as exc:
            logger.warning("OSMnx routing failed, using direct route fallback: %s", exc)
            return self._fallback_routes(request, str(exc))

    def _fetch_graph(self, request: BackendRunRequest) -> nx.MultiDiGraph:
        center = request.map_center or request.start
        radius_km = float(request.graph_radius_km or 12.0)
        graph = ox.graph_from_point(
            center, dist=int(radius_km * 1000), network_type="drive", simplify=True
        )
        if hasattr(ox, "add_edge_lengths"):
            return ox.add_edge_lengths(graph)
        return ox.distance.add_edge_lengths(graph)

    def _graph_routes(
        self,
        request: BackendRunRequest,
        graph: nx.MultiDiGraph,
        flood_polygons: list,
    ) -> dict:
        working_graph = graph.copy()
        blocked_geojson = self._blocked_roads_geojson(graph, flood_polygons)
        blocked_edges = [
            tuple(feature["properties"]["edge_key"])
            for feature in blocked_geojson.get("features", [])
            if "edge_key" in feature.get("properties", {})
        ]
        if blocked_edges:
            working_graph.remove_edges_from(blocked_edges)

        start_node = ox.distance.nearest_nodes(
            working_graph, request.start[1], request.start[0]
        )
        dest_node = ox.distance.nearest_nodes(working_graph, request.dest[1], request.dest[0])

        primary_nodes = nx.shortest_path(
            working_graph, start_node, dest_node, weight="length"
        )
        primary_geojson, stats = self._route_geojson(
            working_graph, primary_nodes, "Primary Safe Route", blocked_edges
        )

        alternatives = []
        alt_graph = working_graph.copy()
        for idx in range(1, 3):
            self._remove_middle_route_edge(alt_graph, primary_nodes)
            try:
                alt_nodes = nx.shortest_path(alt_graph, start_node, dest_node, weight="length")
            except Exception:
                break
            alt_geojson, _ = self._route_geojson(
                alt_graph, alt_nodes, f"Alternative Safe Route {idx}", blocked_edges
            )
            alternatives.append(alt_geojson)
            primary_nodes = alt_nodes

        return {
            "primary_route_geojson": primary_geojson,
            "alternative_routes_geojson": alternatives,
            "blocked_geojson": blocked_geojson,
            "route_stats": stats,
        }

    def _blocked_roads_geojson(self, graph: nx.MultiDiGraph, flood_polygons: list) -> dict:
        if not flood_polygons:
            return {"type": "FeatureCollection", "features": []}

        edges_gdf = ox.graph_to_gdfs(
            graph, nodes=False, edges=True, fill_edge_geometry=True
        )
        flood_union = unary_union(flood_polygons)
        blocked = edges_gdf[edges_gdf.intersects(flood_union)]
        features = []
        for edge_key, row in blocked.iterrows():
            edge_key_json = [self._json_scalar(value) for value in edge_key]
            properties = {
                "name": row.get("name", "Unnamed Road"),
                "highway": row.get("highway", "unknown"),
                "length_m": round(float(row.get("length", 0.0)), 1),
                "impact_level": "blocked_by_detected_flood",
                "edge_key": edge_key_json,
            }
            features.append(
                {
                    "type": "Feature",
                    "properties": properties,
                    "geometry": mapping(row.geometry),
                }
            )
        return {"type": "FeatureCollection", "features": features}

    def _route_geojson(
        self,
        graph: nx.MultiDiGraph,
        route_nodes: list[int],
        name: str,
        blocked_edges: Iterable[tuple],
    ) -> tuple[dict, dict]:
        route_gdf = self._route_to_gdf(graph, route_nodes)
        geometry_union = (
            route_gdf.geometry.union_all()
            if hasattr(route_gdf.geometry, "union_all")
            else route_gdf.geometry.unary_union
        )
        merged = linemerge(geometry_union)
        if isinstance(merged, LineString):
            route_line = merged
        else:
            route_line = LineString(
                [coord for line in merged.geoms for coord in line.coords]
            )

        distance_m = float(route_gdf["length"].sum())
        travel_time_min = distance_m / ((DEFAULT_SPEED_KMH * 1000) / 3600) / 60
        stats = {
            "distance_km": round(distance_m / 1000, 2),
            "travel_time_min": round(travel_time_min, 1),
            "blocked_edges": len(list(blocked_edges)),
        }
        feature = {
            "type": "Feature",
            "properties": {"name": name, **stats},
            "geometry": mapping(route_line),
        }
        return {"type": "FeatureCollection", "features": [feature]}, stats

    def _route_to_gdf(self, graph: nx.MultiDiGraph, route_nodes: list[int]):
        if hasattr(ox, "routing") and hasattr(ox.routing, "route_to_gdf"):
            return ox.routing.route_to_gdf(graph, route_nodes)
        return ox.utils_graph.route_to_gdf(graph, route_nodes)

    def _remove_middle_route_edge(
        self, graph: nx.MultiDiGraph, route_nodes: list[int]
    ) -> None:
        if len(route_nodes) < 3:
            return
        middle = max(0, len(route_nodes) // 2 - 1)
        u = route_nodes[middle]
        v = route_nodes[middle + 1]
        if graph.has_edge(u, v):
            keys = list(graph[u][v].keys())
            graph.remove_edges_from((u, v, key) for key in keys)

    def _fallback_routes(self, request: BackendRunRequest, reason: str) -> dict:
        line = LineString(
            [(request.start[1], request.start[0]), (request.dest[1], request.dest[0])]
        )
        distance_km = self._haversine_km(request.start, request.dest)
        stats = {
            "distance_km": round(distance_km, 2),
            "travel_time_min": round(distance_km / DEFAULT_SPEED_KMH * 60, 1),
            "blocked_edges": 0,
            "routing_mode": "direct_fallback",
            "fallback_reason": reason[:180],
        }
        primary = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {"name": "Direct Route Fallback", **stats},
                    "geometry": mapping(line),
                }
            ],
        }
        return {
            "primary_route_geojson": primary,
            "alternative_routes_geojson": [],
            "blocked_geojson": {"type": "FeatureCollection", "features": []},
            "route_stats": stats,
        }

    def _extract_polygons(self, flood_geojson: Optional[dict]) -> list:
        polygons = []
        for feature in (flood_geojson or {}).get("features", []):
            try:
                geom = shape(feature.get("geometry", {}))
                if geom.is_valid and not geom.is_empty:
                    polygons.append(geom)
            except Exception:
                continue
        return polygons

    def _haversine_km(self, start: tuple[float, float], dest: tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, start)
        lat2, lon2 = map(math.radians, dest)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def _json_scalar(self, value):
        return value.item() if hasattr(value, "item") else value
