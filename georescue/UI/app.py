import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape

from georescue.UI.UI import (
    DEFAULT_CENTER_LAT,
    DEFAULT_CENTER_LON,
    DEFAULT_ZOOM,
    SAMPLE_DAMAGE_GEOJSON,
    STATUS_TEMPLATE,
    SAMPLE_ORCHESTRATOR_ROUTE_GEOJSON,
    setup_logging,
    run_agents_backend_stream,
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

    show_damage = st.checkbox("Show operator flood polygons", value=True)
    show_flood = st.checkbox("Show AI-detected flooded areas", value=True)
    show_route = st.checkbox("Show primary safe route", value=True)
    show_alt_routes = st.checkbox("Show alternative routes", value=True)
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
    realtime_image_dir = st.text_input(
        "Realtime image folder (optional)",
        value="",
        placeholder="Defaults to georescue/ml_serving/data/raw",
        help="If empty, the agent reads latest image from default georescue folder path.",
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

    uploaded_file = st.file_uploader(
        "Upload a satellite image for flood polygon detection",
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
    local_route_clicked = run_col.button("Compute Safe Route")
    mock_clicked = mock_col.button("Load Sample Flood Polygon")
    run_agents_clicked = st.button("Run Flood Agents (HF Qwen + local LoRA + HF Llama 3B)")
    mock_route_clicked = st.button("Load Sample Route (Orchestrator)")

    if mock_clicked:
        st.session_state.drawn_polygons = [shape(feature["geometry"]) for feature in SAMPLE_DAMAGE_GEOJSON["features"]]
        st.info("Loaded sample flood polygon.")

    if mock_route_clicked:
        st.session_state.route_geojson = SAMPLE_ORCHESTRATOR_ROUTE_GEOJSON
        st.info("Loaded sample route from orchestrator.")

    if local_route_clicked or run_agents_clicked:
        if user_prompt.strip():
            st.session_state.status_log = STATUS_TEMPLATE
        else:
            st.session_state.status_log = ["Mission prompt is empty; routing still allowed."]

    st.markdown("---")
    st.subheader("Route Statistics")
    if st.session_state.route_geojson:
        stats = {}
        features = st.session_state.route_geojson.get("features", [])
        if features:
            stats = features[0].get("properties", {}) or {}
        if stats:
            st.write(f"- Distance: {stats.get('distance_km', 'n/a')} km")
            st.write(f"- Travel time: {stats.get('travel_time_min', 'n/a')} min")
            st.write(f"- Blocked roads: {stats.get('blocked_edges', 'n/a')}")
        else:
            st.write("Stats will appear after routing.")
    else:
        st.write("Stats will appear after routing.")

    with st.expander("Raw GeoJSON Output"):
        st.write(
            export_geojson(
                st.session_state.route_geojson,
                st.session_state.damage_geojson,
                st.session_state.flood_geojson,
                st.session_state.blocked_geojson,
                st.session_state.alternative_routes_geojson,
                st.session_state.realtime_exports,
            )
        )

    st.caption("AI agents can be integrated for imagery analysis and hazard detection.")

    if st.session_state.supervisor_plan:
        st.markdown("---")
        st.subheader("Supervisor (HF Llama 3B)")
        st.write(st.session_state.supervisor_plan)
        if st.session_state.supervisor_summary:
            st.caption(st.session_state.supervisor_summary)

    if st.session_state.vision_result:
        st.subheader("Vision Detection")
        st.write(f"Severity: {st.session_state.vision_result.get('severity', 'unknown')}")
        st.write(st.session_state.vision_result.get("findings", ""))

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

    if show_flood and st.session_state.flood_geojson:
        folium.GeoJson(
            st.session_state.flood_geojson,
            name="Detected Flooded Areas",
            style_function=lambda _: {
                "fillColor": "#ff5252",
                "color": "#b71c1c",
                "weight": 2,
                "fillOpacity": 0.35,
            },
        ).add_to(base_map)

    if local_route_clicked:
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
            for feature in route_geojson.get("features", []):
                feature.setdefault("properties", {}).update(stats)
            st.session_state.route_geojson = route_geojson

    if show_route and st.session_state.route_geojson:
        folium.GeoJson(
            st.session_state.route_geojson,
            name="Primary Safe Route",
            style_function=lambda _: {"color": "#3ddc84", "weight": 4},
        ).add_to(base_map)

    if show_alt_routes and st.session_state.alternative_routes_geojson:
        for idx, alt_route in enumerate(st.session_state.alternative_routes_geojson, start=1):
            folium.GeoJson(
                alt_route,
                name=f"Alternative Route {idx}",
                style_function=lambda _: {
                    "color": "#ffca28",
                    "weight": 4,
                    "opacity": 0.9,
                },
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


if run_agents_clicked:
    uploaded_bytes = uploaded_file.read() if uploaded_file else None
    st.session_state.status_log = []
    final_payload = None

    with st.status("Running flood response agents...", expanded=True):
        for event in run_agents_backend_stream(
            mission=user_prompt,
            disaster_type="flood",
            start=(st.session_state.start_lat, st.session_state.start_lon),
            dest=(st.session_state.dest_lat, st.session_state.dest_lon),
            uploaded_image_bytes=uploaded_bytes,
            realtime_image_dir=realtime_image_dir or None,
            map_center=map_center,
            graph_radius_km=float(graph_radius_km),
        ):
            log_line = f"{event.get('agent', 'Agent')}: {event.get('message', '')}"
            st.write(log_line)
            st.session_state.status_log.append(log_line)
            if event.get("status") == "complete":
                final_payload = event.get("data") or {}

    if final_payload:
        st.session_state.flood_geojson = final_payload.get("flood_geojson")
        st.session_state.blocked_geojson = final_payload.get("blocked_geojson")
        st.session_state.route_geojson = final_payload.get("primary_route_geojson")
        st.session_state.alternative_routes_geojson = final_payload.get("alternative_routes_geojson", [])
        st.session_state.vision_result = final_payload.get("vision_result")
        st.session_state.supervisor_plan = final_payload.get("supervisor_plan", "")
        st.session_state.supervisor_summary = final_payload.get("supervisor_summary", "")
        st.session_state.realtime_exports = final_payload.get("realtime_exports")

        stats = final_payload.get("route_stats", {})
        if st.session_state.route_geojson and stats:
            for feat in st.session_state.route_geojson.get("features", []):
                feat.setdefault("properties", {}).update(stats)

        st.success("Flooded areas and alternative routes updated on map.")
        st.rerun()
