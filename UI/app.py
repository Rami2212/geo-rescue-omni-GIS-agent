import os
import logging
from typing import List, Optional, Tuple

import streamlit as st
import folium
from streamlit_folium import st_folium
import osmnx as ox
import networkx as nx
from shapely.geometry import LineString, Polygon, shape, mapping
from shapely.ops import unary_union, linemerge
from folium.plugins import Draw, Fullscreen, MiniMap, MousePosition


# ------------------------------
# Configuration and constants
# ------------------------------
DEFAULT_CENTER_LAT = float(os.getenv("MAP_CENTER_LAT", "6.9271"))
DEFAULT_CENTER_LON = float(os.getenv("MAP_CENTER_LON", "79.8612"))
DEFAULT_ZOOM = int(os.getenv("MAP_DEFAULT_ZOOM", "12"))
DEFAULT_GRAPH_RADIUS_KM = float(os.getenv("MAP_GRAPH_RADIUS_KM", "12"))
DEFAULT_SPEED_KMH = float(os.getenv("DEFAULT_SPEED_KMH", "30"))

SAMPLE_DAMAGE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Sample Damage Zone"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [79.8512, 6.9171],
                        [79.8712, 6.9171],
                        [79.8712, 6.9371],
                        [79.8512, 6.9371],
                        [79.8512, 6.9171],
                    ]
                ],
            },
        }
    ],
}

STATUS_TEMPLATE = [
    "Supervisor: parsing intent",
    "Data Agent: fetching imagery",
    "Vision Agent: extracting damage polygon",
    "Spatial Agent: computing safe route",
    "Reporting Agent: preparing GeoJSON",
]


# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("georescue-ui")


# ------------------------------
# Helper functions
# ------------------------------

def add_edge_lengths_safe(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    if hasattr(ox, "add_edge_lengths"):
        return ox.add_edge_lengths(graph)
    if hasattr(ox, "distance") and hasattr(ox.distance, "add_edge_lengths"):
        return ox.distance.add_edge_lengths(graph)
    logger.warning("osmnx edge-length helper not available; using raw graph.")
    return graph


def get_orchestrator_url() -> str:
    try:
        return st.secrets.get("ORCHESTRATOR_URL", "")
    except Exception:
        return os.getenv("ORCHESTRATOR_URL", "")


def initialize_session_state() -> None:
    if "damage_geojson" not in st.session_state:
        st.session_state.damage_geojson = None
    if "route_geojson" not in st.session_state:
        st.session_state.route_geojson = None
    if "status_log" not in st.session_state:
        st.session_state.status_log = []
    if "drawn_polygons" not in st.session_state:
        st.session_state.drawn_polygons = []
    if "start_lat" not in st.session_state:
        st.session_state.start_lat = DEFAULT_CENTER_LAT - 0.01
    if "start_lon" not in st.session_state:
        st.session_state.start_lon = DEFAULT_CENTER_LON - 0.01
    if "dest_lat" not in st.session_state:
        st.session_state.dest_lat = DEFAULT_CENTER_LAT + 0.015
    if "dest_lon" not in st.session_state:
        st.session_state.dest_lon = DEFAULT_CENTER_LON + 0.015
    if "last_click" not in st.session_state:
        st.session_state.last_click = None


def load_base_map(center: Tuple[float, float], zoom: int) -> folium.Map:
    base_map = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer("CartoDB dark_matter", name="Dark Mode").add_to(base_map)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Satellite",
        attr="Esri",
    ).add_to(base_map)

    Fullscreen().add_to(base_map)
    MiniMap(toggle_display=True).add_to(base_map)
    MousePosition(position="bottomright").add_to(base_map)

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
    ).add_to(base_map)

    return base_map


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

    route_gdf = ox.utils_graph.route_to_gdf(working_graph, route_nodes)
    if route_gdf.empty:
        return None, None, {}, "Route calculation returned no segments."

    merged = linemerge(route_gdf.geometry.values)
    if isinstance(merged, LineString):
        route_line = merged
    else:
        route_line = LineString([coord for line in merged for coord in line.coords])

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


def render_damage_layers(
    base_map: folium.Map,
    polygons: List[Polygon],
    show_damage: bool,
) -> Optional[dict]:
    if not polygons or not show_damage:
        return None

    features = []
    for idx, poly in enumerate(polygons, start=1):
        features.append(
            {
                "type": "Feature",
                "properties": {"name": f"Damage Zone {idx}"},
                "geometry": mapping(poly),
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    folium.GeoJson(
        geojson,
        name="Damage Zones",
        style_function=lambda _: {
            "fillColor": "#ff5f5f",
            "color": "#ff5f5f",
            "weight": 2,
            "fillOpacity": 0.35,
        },
    ).add_to(base_map)

    return geojson


def export_geojson(route_geojson: Optional[dict], damage_geojson: Optional[dict]) -> dict:
    return {
        "damage_zones": damage_geojson,
        "safe_route": route_geojson,
    }


def render_road_network(base_map: folium.Map, graph: nx.MultiDiGraph) -> None:
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    geojson = edges_gdf[["geometry"]].to_json()
    folium.GeoJson(
        geojson,
        name="Road Network",
        style_function=lambda _: {"color": "#6c7a89", "weight": 1, "opacity": 0.6},
    ).add_to(base_map)


# ------------------------------
# Streamlit App
# ------------------------------

st.set_page_config(page_title="GeoRescue Sri Lanka", page_icon="🛰️", layout="wide")
initialize_session_state()

st.title("GeoRescue Sri Lanka Command Center")
st.caption("Disaster routing and GIS safety planning for Colombo")

with st.sidebar:
    st.header("Mission Controls")

    orchestrator_url = get_orchestrator_url()
    if not orchestrator_url:
        st.warning("Orchestrator URL is not configured. Running in mock mode.")

    st.text_input(
        "Orchestrator API URL (optional)",
        value=orchestrator_url,
        placeholder="http://localhost:8000",
        help="Set ORCHESTRATOR_URL in secrets or environment for production.",
    )

    show_damage = st.checkbox("Show damage zones", value=True)
    show_route = st.checkbox("Show safe route", value=True)
    show_roads = st.checkbox("Show road network", value=False)

    zoom = st.slider("Map zoom", min_value=9, max_value=16, value=DEFAULT_ZOOM)
    center_lat = st.number_input(
        "Center latitude", value=DEFAULT_CENTER_LAT, format="%.4f"
    )
    center_lon = st.number_input(
        "Center longitude", value=DEFAULT_CENTER_LON, format="%.4f"
    )
    graph_radius_km = st.slider(
        "Routing graph radius (km)", min_value=4, max_value=30, value=12
    )

    st.markdown("---")
    st.subheader("Status Feed")
    if st.session_state.status_log:
        for item in st.session_state.status_log:
            st.write(f"- {item}")
    else:
        st.write("No agent activity yet.")

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.subheader("Mission Request")
    user_prompt = st.text_area(
        "Describe the incident and routing goal",
        value="Assess flood impact near Colombo Fort and compute a safe route.",
        height=140,
    )

    st.file_uploader(
        "Optional: upload a satellite image",
        type=["png", "jpg", "jpeg", "tif"],
        accept_multiple_files=False,
    )

    st.markdown("---")
    st.subheader("Routing Points")
    st.session_state.start_lat = st.number_input(
        "Start latitude", value=st.session_state.start_lat, format="%.5f"
    )
    st.session_state.start_lon = st.number_input(
        "Start longitude", value=st.session_state.start_lon, format="%.5f"
    )
    st.session_state.dest_lat = st.number_input(
        "Destination latitude", value=st.session_state.dest_lat, format="%.5f"
    )
    st.session_state.dest_lon = st.number_input(
        "Destination longitude", value=st.session_state.dest_lon, format="%.5f"
    )

    st.caption("Tip: click on the map, then use the buttons to set points.")
    set_start = st.button("Set start from last click")
    set_dest = st.button("Set destination from last click")

    run_col, mock_col = st.columns(2)
    run_clicked = run_col.button("Compute Safe Route")
    mock_clicked = mock_col.button("Load Sample Hazard")

    if mock_clicked:
        st.session_state.drawn_polygons = [shape(feature["geometry"]) for feature in SAMPLE_DAMAGE_GEOJSON["features"]]
        st.info("Loaded sample damage polygon.")

    if run_clicked:
        if user_prompt.strip():
            st.session_state.status_log = STATUS_TEMPLATE
        else:
            st.session_state.status_log = ["Mission prompt is empty; routing still allowed."]

    st.markdown("---")
    st.subheader("Route Statistics")
    if st.session_state.route_geojson:
        stats = st.session_state.route_geojson.get("properties", {})
        if stats:
            st.write(f"- Distance: {stats.get('distance_km', 'n/a')} km")
            st.write(f"- Travel time: {stats.get('travel_time_min', 'n/a')} min")
            st.write(f"- Blocked roads: {stats.get('blocked_edges', 'n/a')}")
        else:
            st.write("Stats will appear after routing.")
    else:
        st.write("Stats will appear after routing.")

    with st.expander("Raw GeoJSON Output"):
        st.write(export_geojson(st.session_state.route_geojson, st.session_state.damage_geojson))

    st.caption("AI agents can be integrated for imagery analysis and hazard detection.")

with right_col:
    st.subheader("Operational Map")

    map_center = (center_lat, center_lon)
    base_map = load_base_map(map_center, zoom)

    graph = fetch_sri_lanka_graph(map_center, graph_radius_km)

    if show_roads:
        render_road_network(base_map, graph)

    damage_geojson = render_damage_layers(
        base_map, st.session_state.drawn_polygons, show_damage
    )
    if damage_geojson:
        st.session_state.damage_geojson = damage_geojson

    if run_clicked:
        route_geojson, route_line, stats, error = calculate_safe_route(
            graph,
            (st.session_state.start_lat, st.session_state.start_lon),
            (st.session_state.dest_lat, st.session_state.dest_lon),
            st.session_state.drawn_polygons,
        )
        if error:
            st.warning(error)
            st.session_state.route_geojson = None
        else:
            route_geojson["properties"] = stats
            st.session_state.route_geojson = route_geojson

    if show_route and st.session_state.route_geojson:
        folium.GeoJson(
            st.session_state.route_geojson,
            name="Safe Route",
            style_function=lambda _: {"color": "#3ddc84", "weight": 4},
        ).add_to(base_map)

    folium.LayerControl().add_to(base_map)
    map_output = st_folium(base_map, use_container_width=True, height=560)

    last_clicked = map_output.get("last_clicked") if map_output else None
    if last_clicked:
        st.session_state.last_click = last_clicked

    drawings = map_output.get("all_drawings", []) if map_output else []
    drawn_polygons = parse_drawn_polygons(drawings) if drawings else []
    if drawn_polygons:
        st.session_state.drawn_polygons = drawn_polygons

    if set_start and st.session_state.last_click:
        st.session_state.start_lat = st.session_state.last_click["lat"]
        st.session_state.start_lon = st.session_state.last_click["lng"]
    if set_dest and st.session_state.last_click:
        st.session_state.dest_lat = st.session_state.last_click["lat"]
        st.session_state.dest_lon = st.session_state.last_click["lng"]

    if orchestrator_url:
        st.caption(f"Planned API target: {orchestrator_url}")
