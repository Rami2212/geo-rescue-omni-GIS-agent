"""Download and cache the OSM road network for Colombo, Sri Lanka."""
from pathlib import Path

import networkx as nx
import osmnx as ox

from .config import PROCESSED_DIR
from .export_geojson import save_geojson


def download_road_network(
    location: str = "Colombo, Sri Lanka",
    network_type: str = "drive",
    save_dir: "Path | None" = None,
) -> tuple:
    out_dir = Path(save_dir) if save_dir else PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    G = ox.graph_from_place(location, network_type=network_type)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)

    save_geojson(edges_gdf.reset_index(), out_dir / "colombo_road_network.geojson")
    save_geojson(nodes_gdf.reset_index(), out_dir / "colombo_road_nodes.geojson")
    nx.write_graphml(G, str(out_dir / "colombo_road_network_graph.graphml"))

    print(f"Road network: {len(edges_gdf)} edges, {len(nodes_gdf)} nodes → {out_dir}")
    return nodes_gdf, edges_gdf, G


def load_road_graph(graph_path: "str | Path") -> nx.MultiDiGraph:
    return nx.read_graphml(str(graph_path))
