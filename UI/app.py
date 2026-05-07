import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape

from georescue import (
    DEFAULT_CENTER_LAT,
    DEFAULT_CENTER_LON,
    DEFAULT_ZOOM,
    DEFAULT_GRAPH_RADIUS_KM,
    SAMPLE_DAMAGE_GEOJSON,
    STATUS_TEMPLATE,
    SAMPLE_ORCHESTRATOR_ROUTE_GEOJSON,
    setup_logging,
    get_orchestrator_url,
    initialize_session_state,
    load_base_map,
    render_damage_layers,
    render_road_network,
    fetch_sri_lanka_graph,
    parse_drawn_polygons,
    calculate_safe_route,
    export_geojson,
)

logger = setup_logging()

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
    mock_route_clicked = st.button("Load Sample Route (Orchestrator)")

    if mock_clicked:
        st.session_state.drawn_polygons = [shape(feature["geometry"]) for feature in SAMPLE_DAMAGE_GEOJSON["features"]]
        st.info("Loaded sample damage polygon.")

    if mock_route_clicked:
        st.session_state.route_geojson = SAMPLE_ORCHESTRATOR_ROUTE_GEOJSON
        st.info("Loaded sample route from orchestrator.")

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
