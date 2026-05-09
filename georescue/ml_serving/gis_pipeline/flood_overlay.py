"""
Flood polygon overlay and road network analysis.

Loads road GeoJSON, creates flood polygons, identifies blocked roads,
and exports affected road segments.
"""

import logging
import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import geopandas as gpd
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
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
RAW_DIR = DATA_DIR / "raw"
BLOCKED_ROADS_FILE = "blocked_roads_flood.geojson"
FLOOD_POLYGON_FILE = "flood_polygon.geojson"


def load_road_geojson(geojson_path: Path) -> gpd.GeoDataFrame:
    """
    Load road network GeoJSON file using GeoPandas.

    Args:
        geojson_path: Path to GeoJSON file.

    Returns:
        gpd.GeoDataFrame: Road network data.

    Raises:
        Exception: If file not found or invalid.
    """
    try:
        logger.info(f"Loading road network from {geojson_path}...")
        roads_gdf = gpd.read_file(geojson_path)
        logger.info(f"  - Loaded {len(roads_gdf)} road segments")
        logger.info(f"  - CRS: {roads_gdf.crs}")
        logger.info(f"  - Bounds: {roads_gdf.total_bounds}")
        return roads_gdf

    except FileNotFoundError:
        logger.error(f"File not found: {geojson_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load GeoJSON: {str(e)}")
        raise


def create_sample_flood_polygon(
    roads_gdf: gpd.GeoDataFrame,
    flood_center_lon: float = 80.85,
    flood_center_lat: float = 6.92,
    flood_radius_km: float = 2.0,
) -> Polygon:
    """
    Create a sample flood polygon around Colombo.

    Args:
        roads_gdf: Road network GeoDataFrame (for CRS reference).
        flood_center_lon: Longitude of flood center (default: central Colombo).
        flood_center_lat: Latitude of flood center (default: central Colombo).
        flood_radius_km: Radius of flood area in kilometers.

    Returns:
        Polygon: Flood polygon geometry.
    """
    try:
        logger.info(f"Creating sample flood polygon...")
        logger.info(f"  - Center: ({flood_center_lon}, {flood_center_lat})")
        logger.info(f"  - Radius: {flood_radius_km} km")

        # Convert km to degrees (approximate: 1 degree ≈ 111 km)
        radius_degrees = flood_radius_km / 111.0

        # Create circular buffer as polygon (approximated with square)
        min_lon = flood_center_lon - radius_degrees
        min_lat = flood_center_lat - radius_degrees
        max_lon = flood_center_lon + radius_degrees
        max_lat = flood_center_lat + radius_degrees

        # Create bounding box polygon and buffer it
        flood_poly = box(min_lon, min_lat, max_lon, max_lat)
        
        # Better circular approximation: convert to projected CRS
        temp_gdf = gpd.GeoDataFrame(
            {"geometry": [flood_poly]},
            crs="EPSG:4326"
        )
        
        # Project to meter-based CRS for accurate buffer
        if roads_gdf.crs and roads_gdf.crs != "EPSG:4326":
            temp_gdf = temp_gdf.to_crs(roads_gdf.crs)
        else:
            # Use UTM for Sri Lanka (Zone 44N)
            temp_gdf = temp_gdf.to_crs("EPSG:32644")
        
        # Buffer by flood radius in meters
        flood_radius_m = flood_radius_km * 1000
        flood_poly_buffered = temp_gdf.geometry[0].buffer(flood_radius_m)
        
        # Convert back to original CRS
        result_gdf = gpd.GeoDataFrame(
            {"geometry": [flood_poly_buffered]},
            crs="EPSG:32644"
        )
        if roads_gdf.crs and roads_gdf.crs != "EPSG:4326":
            result_gdf = result_gdf.to_crs(roads_gdf.crs)
        else:
            result_gdf = result_gdf.to_crs("EPSG:4326")
        
        flood_polygon = result_gdf.geometry[0]
        logger.info(f"  - Flood polygon area: {flood_polygon.area:.6f} square degrees")

        return flood_polygon

    except Exception as e:
        logger.error(f"Failed to create flood polygon: {str(e)}")
        raise


def load_flood_polygon(flood_polygon_path: Path, roads_gdf: gpd.GeoDataFrame) -> Polygon:
    """Load a flood polygon from GeoJSON and align CRS with roads."""
    logger.info(f"Loading flood polygon from {flood_polygon_path}...")
    flood_gdf = gpd.read_file(flood_polygon_path)

    if flood_gdf.empty:
        raise ValueError(f"Flood polygon file is empty: {flood_polygon_path}")

    if roads_gdf.crs and flood_gdf.crs and roads_gdf.crs != flood_gdf.crs:
        flood_gdf = flood_gdf.to_crs(roads_gdf.crs)

    flood_polygon = unary_union(flood_gdf.geometry)
    logger.info("  - Loaded polygon geometry from live/external source")
    return flood_polygon


def identify_blocked_roads(
    roads_gdf: gpd.GeoDataFrame,
    flood_polygon: Polygon,
) -> gpd.GeoDataFrame:
    """
    Identify roads intersecting with flood polygon using overlay.

    Args:
        roads_gdf: GeoDataFrame of road segments.
        flood_polygon: Flood polygon geometry.

    Returns:
        gpd.GeoDataFrame: Road segments affected by flood.
    """
    try:
        logger.info("Identifying blocked roads using spatial overlay...")

        # Create flood GeoDataFrame
        flood_gdf = gpd.GeoDataFrame(
            {"id": [1], "flood_zone": ["active"]},
            geometry=[flood_polygon],
            crs=roads_gdf.crs
        )

        # Perform overlay operation
        intersection_gdf = gpd.overlay(
            roads_gdf,
            flood_gdf,
            how='intersection',
            keep_geom_type=False
        )

        # Filter only LineString geometries (roads)
        blocked_roads = intersection_gdf[
            intersection_gdf.geometry.geom_type == 'LineString'
        ].copy()

        logger.info(f"  - Found {len(blocked_roads)} blocked road segments")

        # Calculate lengths in a meter-based CRS to avoid geographic length warnings.
        blocked_roads_m = blocked_roads.to_crs("EPSG:32644")
        blocked_roads['affected_length_m'] = blocked_roads_m.geometry.length
        blocked_roads['flood_impact'] = 'partial'
        
        # Identify fully blocked roads (compare to original road length)
        roads_within = gpd.clip(roads_gdf, flood_polygon)
        blocked_roads.loc[
            blocked_roads.index.isin(roads_within.index),
            'flood_impact'
        ] = 'fully_blocked'

        return blocked_roads

    except Exception as e:
        logger.error(f"Failed to identify blocked roads: {str(e)}")
        raise


def calculate_impact_statistics(
    blocked_roads: gpd.GeoDataFrame,
) -> dict:
    """
    Calculate impact statistics for blocked roads.

    Args:
        blocked_roads: GeoDataFrame of affected roads.

    Returns:
        dict: Statistics on flood impact.
    """
    try:
        total_affected_length = blocked_roads['affected_length_m'].sum()
        num_fully_blocked = (blocked_roads['flood_impact'] == 'fully_blocked').sum()
        num_partial = (blocked_roads['flood_impact'] == 'partial').sum()

        # Highway classification impact
        highway_impact = {}
        if 'highway' in blocked_roads.columns:
            highway_values = blocked_roads['highway'].apply(
                lambda x: ",".join(x) if isinstance(x, (list, tuple)) else str(x)
            )
            highway_impact = (
                highway_values
                .value_counts()
                .to_dict()
            )

        stats = {
            'total_affected_roads': len(blocked_roads),
            'total_affected_length_m': total_affected_length,
            'fully_blocked': num_fully_blocked,
            'partially_blocked': num_partial,
            'affected_highway_types': highway_impact,
        }

        return stats

    except Exception as e:
        logger.error(f"Failed to calculate statistics: {str(e)}")
        return {}


def save_blocked_roads_geojson(
    blocked_roads: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """
    Save blocked roads as GeoJSON file.

    Args:
        blocked_roads: GeoDataFrame of affected roads.
        output_path: Path to save GeoJSON file.

    Raises:
        Exception: If save fails.
    """
    try:
        logger.info(f"Saving blocked roads GeoJSON...")
        blocked_roads.to_file(output_path, driver="GeoJSON")
        logger.info(f"  - Saved to {output_path}")
        logger.info(f"  - File size: {output_path.stat().st_size / 1024:.2f} KB")

    except Exception as e:
        logger.error(f"Failed to save GeoJSON: {str(e)}")
        raise


def save_flood_polygon_geojson(
    flood_polygon: Polygon,
    roads_crs,
    output_path: Path,
) -> None:
    """
    Save flood polygon as GeoJSON for reference.

    Args:
        flood_polygon: Flood polygon geometry.
        roads_crs: CRS of road network.
        output_path: Path to save GeoJSON file.

    Raises:
        Exception: If save fails.
    """
    try:
        logger.info(f"Saving flood polygon GeoJSON...")
        flood_gdf = gpd.GeoDataFrame(
            {"id": [1], "type": ["flood_zone"]},
            geometry=[flood_polygon],
            crs=roads_crs
        )
        flood_gdf.to_file(output_path, driver="GeoJSON")
        logger.info(f"  - Saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save flood polygon: {str(e)}")
        raise


def analyze_flood_impact(
    roads_geojson_path: Optional[Path] = None,
    flood_polygon_path: Optional[Path] = None,
) -> Tuple[Path, Path, dict]:
    """
    Main function to analyze flood impact on road network.

    Args:
        roads_geojson_path: Path to road GeoJSON file. If None, uses default.

    Returns:
        Tuple[Path, Path, dict]: Paths to blocked roads and flood polygon, plus statistics.

    Raises:
        Exception: If analysis fails.
    """
    logger.info("=" * 60)
    logger.info("Flood Impact Analysis on Road Network")
    logger.info("=" * 60)

    try:
        # Default path
        if roads_geojson_path is None:
            roads_geojson_path = PROCESSED_DIR / "colombo_road_network.geojson"

        # Load road network
        roads_gdf = load_road_geojson(roads_geojson_path)

        # Load external polygon or fallback to sample flood zone
        if flood_polygon_path and flood_polygon_path.exists():
            flood_polygon = load_flood_polygon(flood_polygon_path, roads_gdf)
        else:
            flood_polygon = create_sample_flood_polygon(roads_gdf)

        # Identify blocked roads
        blocked_roads = identify_blocked_roads(roads_gdf, flood_polygon)

        # Calculate statistics
        stats = calculate_impact_statistics(blocked_roads)
        logger.info("Flood Impact Statistics:")
        logger.info(f"  - Total affected roads: {stats.get('total_affected_roads', 'N/A')}")
        total_len = stats.get('total_affected_length_m')
        if total_len is not None:
            logger.info(f"  - Total affected length: {total_len:.2f} m")
        else:
            logger.info("  - Total affected length: N/A")
        logger.info(f"  - Fully blocked: {stats.get('fully_blocked', 'N/A')}")
        logger.info(f"  - Partially blocked: {stats.get('partially_blocked', 'N/A')}")

        # Save results
        blocked_roads_path = PROCESSED_DIR / BLOCKED_ROADS_FILE
        flood_polygon_path = PROCESSED_DIR / FLOOD_POLYGON_FILE
        
        save_blocked_roads_geojson(blocked_roads, blocked_roads_path)
        save_flood_polygon_geojson(flood_polygon, roads_gdf.crs, flood_polygon_path)

        logger.info("=" * 60)
        logger.info("SUCCESS: Flood impact analysis complete")
        logger.info("=" * 60)

        return blocked_roads_path, flood_polygon_path, stats

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"FAILED: {str(e)}")
        logger.error("=" * 60)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze flood impact on road network")
    parser.add_argument(
        "--flood-polygon",
        type=str,
        default=None,
        help="Optional path to flood polygon GeoJSON (live or external)",
    )
    args = parser.parse_args()

    polygon_path = Path(args.flood_polygon) if args.flood_polygon else None
    blocked_roads_path, flood_polygon_path, stats = analyze_flood_impact(
        flood_polygon_path=polygon_path
    )
