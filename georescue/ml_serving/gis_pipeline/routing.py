"""
OSMnx and NetworkX routing for safe path navigation.

Loads road network, removes blocked roads, and generates shortest routes
avoiding flood-affected areas. Exports routes as GeoJSON.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List, Dict

import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString, Point
import pandas as pd


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Constants
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_FILE = "colombo_road_network_graph.graphml"
ROUTE_FILE = "optimal_route_{}.geojson"

# Default route endpoints in Colombo (in lat, lon format for coordinates)
DEFAULT_START_COORD = (6.9271, 80.7789)  # Colombo Fort
DEFAULT_END_COORD = (6.8520, 80.8197)    # Mount Lavinia


def load_graph(graph_path: Optional[Path] = None) -> object:
    """
    Load road network graph from GraphML file.

    Args:
        graph_path: Path to GraphML file. If None, uses default.

    Returns:
        networkx.MultiDiGraph: Road network graph.

    Raises:
        Exception: If graph file not found or invalid.
    """
    try:
        if graph_path is None:
            graph_path = PROCESSED_DIR / GRAPH_FILE

        logger.info(f"Loading graph from {graph_path}...")
        graph = ox.load_graphml(graph_path)
        
        num_nodes = len(graph.nodes())
        num_edges = len(graph.edges())
        
        logger.info(f"  - Nodes: {num_nodes}")
        logger.info(f"  - Edges: {num_edges}")
        
        return graph

    except FileNotFoundError:
        logger.error(f"Graph file not found: {graph_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load graph: {str(e)}")
        raise


def remove_blocked_roads(
    graph: object,
    blocked_roads_path: Optional[Path] = None,
) -> object:
    """
    Remove blocked road segments from graph based on flood impact.

    Args:
        graph: networkx.MultiDiGraph road network graph.
        blocked_roads_path: Path to blocked roads GeoJSON. If None, skips removal.

    Returns:
        networkx.MultiDiGraph: Modified graph with blocked roads removed.

    Raises:
        Exception: If blocked roads file invalid.
    """
    try:
        if blocked_roads_path is None:
            blocked_roads_path = PROCESSED_DIR / "blocked_roads_flood.geojson"

        # Check if blocked roads file exists
        if not blocked_roads_path.exists():
            logger.warning(f"Blocked roads file not found: {blocked_roads_path}")
            logger.info("Proceeding with full graph (no roads removed)")
            return graph

        logger.info(f"Removing blocked roads from graph...")
        blocked_gdf = gpd.read_file(blocked_roads_path)
        
        initial_edges = len(graph.edges())
        
        # Create working copy
        graph_safe = graph.copy()
        
        # Extract blocked road nodes
        blocked_edges_removed = 0
        for idx, road in blocked_gdf.iterrows():
            geometry = road.geometry
            
            if geometry.geom_type == 'LineString':
                coords = list(geometry.coords)
                
                # Find and remove edges matching blocked roads
                for u, v, k in list(graph_safe.edges(keys=True)):
                    edge_data = graph_safe[u][v][k]
                    
                    # Check if edge geometry matches blocked road
                    if 'geometry' in edge_data:
                        edge_geom = edge_data['geometry']
                        # Check for overlap/intersection
                        if edge_geom.intersects(geometry):
                            try:
                                graph_safe.remove_edge(u, v, k)
                                blocked_edges_removed += 1
                            except nx.NetworkXError:
                                continue
        
        remaining_edges = len(graph_safe.edges())
        logger.info(f"  - Edges removed: {blocked_edges_removed}")
        logger.info(f"  - Remaining edges: {remaining_edges}")
        
        return graph_safe

    except Exception as e:
        logger.error(f"Failed to remove blocked roads: {str(e)}")
        raise


def get_nearest_nodes(
    graph: object,
    coordinates: Tuple[float, float],
) -> int:
    """
    Find nearest network node to given coordinates using OSMnx.

    Args:
        graph: networkx.MultiDiGraph road network graph.
        coordinates: Coordinates as (latitude, longitude) tuple.

    Returns:
        int: Nearest node ID.

    Raises:
        Exception: If no nearest node found.
    """
    try:
        lat, lon = coordinates
        logger.info(f"Finding nearest node for ({lat}, {lon})...")
        
        # Use nearest_nodes to find closest node
        nearest_node = ox.nearest_nodes(
            graph,
            X=lon,  # longitude
            Y=lat   # latitude
        )
        
        logger.info(f"  - Nearest node: {nearest_node}")
        
        # Get node coordinates
        node_lat = graph.nodes[nearest_node]['y']
        node_lon = graph.nodes[nearest_node]['x']
        logger.info(f"  - Node coordinates: ({node_lat:.6f}, {node_lon:.6f})")
        
        return nearest_node

    except Exception as e:
        logger.error(f"Failed to find nearest node: {str(e)}")
        raise


def calculate_route_length(
    graph: object,
    route: List[int],
) -> float:
    """
    Calculate total length of route in meters.

    Args:
        graph: networkx.MultiDiGraph road network graph.
        route: List of node IDs forming the route.

    Returns:
        float: Total route length in meters.
    """
    try:
        total_length = 0
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            
            # Get edge data (MultiDiGraph may have multiple edges)
            if graph.has_edge(u, v):
                edge_data = graph[u][v]
                
                # Sum lengths of all edges between nodes
                for key in edge_data:
                    if 'length' in edge_data[key]:
                        total_length += edge_data[key]['length']
        
        return total_length

    except Exception as e:
        logger.error(f"Failed to calculate route length: {str(e)}")
        return 0


def generate_shortest_route(
    graph: object,
    start_coords: Tuple[float, float],
    end_coords: Tuple[float, float],
    weight: str = "length",
) -> List[int]:
    """
    Generate shortest path between two coordinates using NetworkX.

    Args:
        graph: networkx.MultiDiGraph road network graph.
        start_coords: Start coordinates as (latitude, longitude).
        end_coords: End coordinates as (latitude, longitude).
        weight: Edge attribute to use for weight ('length' for distance).

    Returns:
        List[int]: List of node IDs forming the route.

    Raises:
        Exception: If route cannot be found.
    """
    try:
        logger.info(f"Generating shortest route...")
        logger.info(f"  - Start: {start_coords}")
        logger.info(f"  - End: {end_coords}")
        
        # Find nearest nodes to coordinates
        start_node = get_nearest_nodes(graph, start_coords)
        end_node = get_nearest_nodes(graph, end_coords)
        
        logger.info(f"Calculating shortest path using NetworkX...")
        
        # Use shortest_path to find optimal route
        route = nx.shortest_path(
            graph,
            source=start_node,
            target=end_node,
            weight=weight
        )
        
        route_length = calculate_route_length(graph, route)
        
        logger.info(f"  - Route found!")
        logger.info(f"  - Number of nodes: {len(route)}")
        logger.info(f"  - Total distance: {route_length:.2f} m ({route_length/1000:.2f} km)")
        
        return route

    except nx.NetworkXNoPath:
        logger.error("No route found between start and end points")
        raise
    except Exception as e:
        logger.error(f"Failed to generate route: {str(e)}")
        raise


def convert_route_to_geodataframe(
    graph: object,
    route: List[int],
) -> gpd.GeoDataFrame:
    """
    Convert route node list to GeoDataFrame with LineString geometries.

    Args:
        graph: networkx.MultiDiGraph road network graph.
        route: List of node IDs forming the route.

    Returns:
        gpd.GeoDataFrame: Route as GeoDataFrame with geometry.
    """
    try:
        logger.info("Converting route to GeoDataFrame...")
        
        route_geometries = []
        route_data = []
        
        for i in range(len(route) - 1):
            u, v = route[i], route[i + 1]
            
            # Get edge data
            edge_keys = graph[u][v].keys()
            for key in edge_keys:
                edge_data = graph[u][v][key]
                
                if 'geometry' in edge_data:
                    geom = edge_data['geometry']
                else:
                    # Create geometry from node coordinates
                    u_lat, u_lon = graph.nodes[u]['y'], graph.nodes[u]['x']
                    v_lat, v_lon = graph.nodes[v]['y'], graph.nodes[v]['x']
                    geom = LineString([(u_lon, u_lat), (v_lon, v_lat)])
                
                route_geometries.append(geom)
                
                # Collect edge attributes
                route_data.append({
                    'segment': i,
                    'node_from': u,
                    'node_to': v,
                    'length_m': edge_data.get('length', 0),
                    'street_name': edge_data.get('name', 'Unknown'),
                })
        
        # Create GeoDataFrame
        route_gdf = gpd.GeoDataFrame(
            route_data,
            geometry=route_geometries,
            crs="EPSG:4326"
        )
        
        logger.info(f"  - Route segments: {len(route_gdf)}")
        logger.info(f"  - Total distance: {route_gdf['length_m'].sum():.2f} m")
        
        return route_gdf

    except Exception as e:
        logger.error(f"Failed to convert route: {str(e)}")
        raise


def save_route_geojson(
    route_gdf: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """
    Save route as GeoJSON file.

    Args:
        route_gdf: GeoDataFrame of route segments.
        output_path: Path to save GeoJSON file.

    Raises:
        Exception: If save fails.
    """
    try:
        logger.info(f"Saving route GeoJSON...")
        route_gdf.to_file(output_path, driver="GeoJSON")
        logger.info(f"  - Saved to {output_path}")
        logger.info(f"  - File size: {output_path.stat().st_size / 1024:.2f} KB")

    except Exception as e:
        logger.error(f"Failed to save route: {str(e)}")
        raise


def plan_safe_route(
    start_coords: Tuple[float, float] = DEFAULT_START_COORD,
    end_coords: Tuple[float, float] = DEFAULT_END_COORD,
    remove_blocked: bool = True,
    blocked_roads_path: Optional[Path] = None,
    graph_path: Optional[Path] = None,
) -> Tuple[Path, gpd.GeoDataFrame]:
    """
    Main function to plan safe route avoiding blocked roads.

    Args:
        start_coords: Start coordinates as (latitude, longitude).
        end_coords: End coordinates as (latitude, longitude).
        remove_blocked: Whether to exclude blocked roads from routing.
        blocked_roads_path: Path to blocked roads GeoJSON.
        graph_path: Path to road network graph.

    Returns:
        Tuple[Path, gpd.GeoDataFrame]: Path to saved GeoJSON and route GeoDataFrame.

    Raises:
        Exception: If routing fails.
    """
    logger.info("=" * 60)
    logger.info("Safe Route Planning")
    logger.info("=" * 60)

    try:
        # Load graph
        graph = load_graph(graph_path)

        # Remove blocked roads if requested
        if remove_blocked:
            graph = remove_blocked_roads(graph, blocked_roads_path)

        # Generate route
        route = generate_shortest_route(graph, start_coords, end_coords)

        # Convert to GeoDataFrame
        route_gdf = convert_route_to_geodataframe(graph, route)

        # Save route
        route_filename = ROUTE_FILE.format("colombo")
        route_path = PROCESSED_DIR / route_filename
        save_route_geojson(route_gdf, route_path)

        logger.info("=" * 60)
        logger.info("SUCCESS: Route planned and saved")
        logger.info("=" * 60)

        return route_path, route_gdf

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"FAILED: {str(e)}")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    # Example usage
    route_path, route_gdf = plan_safe_route()
