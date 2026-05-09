"""
Folium map layer builders for GeoRescue.

Each function takes a folium.Map and optional data, adds a named layer,
and returns nothing (mutation pattern — consistent with folium API).
"""

import logging
from typing import List, Optional, Tuple

import folium
import networkx as nx
import osmnx as ox
from folium.plugins import Draw, Fullscreen, MiniMap, MousePosition
from shapely.geometry import Polygon, mapping

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base map
# ---------------------------------------------------------------------------

def load_base_map(center: Tuple[float, float], zoom: int) -> folium.Map:
    """Create the base Folium map with tile layers and drawing controls."""
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer("CartoDB dark_matter", name="Dark Mode").add_to(m)
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        name="Satellite",
        attr="Esri World Imagery",
    ).add_to(m)

    Fullscreen(position="topleft").add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    MousePosition(position="bottomright").add_to(m)

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "rectangle": True,
            "polygon": True,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

    return m


# ---------------------------------------------------------------------------
# User-drawn damage / hazard zones
# ---------------------------------------------------------------------------

def render_damage_layers(
    m: folium.Map,
    polygons: List[Polygon],
    visible: bool = True,
) -> Optional[dict]:
    """
    Render user-drawn hazard polygons as red translucent overlays.
    Returns the GeoJSON FeatureCollection, or None if nothing to render.
    """
    if not polygons or not visible:
        return None

    features = [
        {
            "type": "Feature",
            "properties": {"name": f"Damage Zone {i}"},
            "geometry": mapping(poly),
        }
        for i, poly in enumerate(polygons, start=1)
    ]
    geojson = {"type": "FeatureCollection", "features": features}

    folium.GeoJson(
        geojson,
        name="Drawn Hazard Zones",
        style_function=lambda _: {
            "fillColor": "#ff4444",
            "color": "#cc0000",
            "weight": 2,
            "fillOpacity": 0.35,
        },
        tooltip=folium.GeoJsonTooltip(fields=["name"]),
    ).add_to(m)
    return geojson


# ---------------------------------------------------------------------------
# API-provided flood zone
# ---------------------------------------------------------------------------

def render_flood_zone(
    m: folium.Map,
    flood_geojson: Optional[dict],
    visible: bool = True,
) -> None:
    """
    Render the live flood zone polygon from the GIS API as a blue overlay.
    The fill uses a gradient-like opacity to convey water extent.
    """
    if not flood_geojson or not visible:
        return
    features = flood_geojson.get("features", [])
    if not features:
        return

    severity = features[0].get("properties", {}).get("severity", "low")
    opacity_map = {"low": 0.2, "moderate": 0.35, "high": 0.5, "extreme": 0.65}
    fill_opacity = opacity_map.get(severity, 0.3)

    folium.GeoJson(
        flood_geojson,
        name="Live Flood Zone",
        style_function=lambda _: {
            "fillColor": "#1a78c2",
            "color": "#0d47a1",
            "weight": 2,
            "fillOpacity": fill_opacity,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["severity", "timestamp"],
            aliases=["Severity:", "Updated:"],
            sticky=True,
        ),
    ).add_to(m)


# ---------------------------------------------------------------------------
# API-provided blocked roads
# ---------------------------------------------------------------------------

def render_blocked_roads(
    m: folium.Map,
    blocked_geojson: Optional[dict],
    visible: bool = True,
) -> None:
    """
    Render road segments blocked by flooding as bold red lines.
    Includes a tooltip showing road name and affected length.
    """
    if not blocked_geojson or not visible:
        return
    if not blocked_geojson.get("features"):
        return

    folium.GeoJson(
        blocked_geojson,
        name="Blocked Roads",
        style_function=lambda _: {
            "color": "#e53935",
            "weight": 4,
            "opacity": 0.9,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["name", "highway", "affected_length_m"],
            aliases=["Road:", "Type:", "Affected (m):"],
            sticky=False,
        ),
    ).add_to(m)


# ---------------------------------------------------------------------------
# Safe route (local OSMnx or API-provided)
# ---------------------------------------------------------------------------

def render_safe_route(
    m: folium.Map,
    route_geojson: Optional[dict],
    label: str = "Safe Route",
    visible: bool = True,
) -> None:
    """
    Render the safe evacuation route as a green line.
    Works with both the local OSMnx output and the API GeoJSON.
    """
    if not route_geojson or not visible:
        return

    folium.GeoJson(
        route_geojson,
        name=label,
        style_function=lambda _: {
            "color": "#00e676",
            "weight": 5,
            "opacity": 0.95,
        },
    ).add_to(m)


# ---------------------------------------------------------------------------
# Road network overlay (optional, expensive)
# ---------------------------------------------------------------------------

def render_road_network(m: folium.Map, graph: nx.MultiDiGraph) -> None:
    """Render the full OSMnx road graph as thin grey lines (for debugging)."""
    try:
        edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
        folium.GeoJson(
            edges_gdf[["geometry"]].to_json(),
            name="Road Network",
            style_function=lambda _: {"color": "#607d8b", "weight": 1, "opacity": 0.5},
        ).add_to(m)
    except Exception as exc:
        logger.warning("Could not render road network: %s", exc)


# ---------------------------------------------------------------------------
# Start / destination markers
# ---------------------------------------------------------------------------

def render_route_markers(
    m: folium.Map,
    start: Tuple[float, float],
    dest: Tuple[float, float],
) -> None:
    """Add start (green pin) and destination (red pin) markers to the map."""
    folium.Marker(
        location=list(start),
        popup="Start Point",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(m)
    folium.Marker(
        location=list(dest),
        popup="Destination",
        icon=folium.Icon(color="red", icon="flag", prefix="fa"),
    ).add_to(m)
