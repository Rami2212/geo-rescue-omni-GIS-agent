import streamlit as st
import folium
from streamlit_folium import st_folium
from shapely.geometry import shape

from georescue.UI.georescue import (
    DEFAULT_CENTER_LAT,
    DEFAULT_CENTER_LON,
    DEFAULT_ZOOM,
    SAMPLE_DAMAGE_GEOJSON,
    STATUS_TEMPLATE,
    SAMPLE_ORCHESTRATOR_ROUTE_GEOJSON,
    setup_logging,
    get_orchestrator_url,
    initialize_session_state,
    load_base_map,
    render_damage_layers,
    render_road_network,
    render_flood_layer,
    render_blocked_roads,
    fetch_sri_lanka_graph,
    parse_drawn_polygons,
    calculate_safe_route,
    export_geojson,
    run_gis_cycle,
    get_gis_status,
    get_flood_polygon,
    get_blocked_roads,
    get_safe_route,
    analyze_image,
    geojson_to_polygons,
    extract_route_stats,
)

logger = setup_logging()

# ------------------------------
# Streamlit App
# ------------------------------

st.set_page_config(page_title="GeoRescue Sri Lanka", page_icon="🛰️", layout="wide")
initialize_session_state()

st.title("GeoRescue Sri Lanka Command Center")
st.caption("Disaster routing and GIS safety planning for Colombo")


def sync_orchestrator_data(base_url: str) -> None:
    if not base_url:
        st.warning("Orchestrator URL is not configured.")
        return

    cycle, cycle_error = run_gis_cycle(base_url)
    if cycle_error:
        st.warning(cycle_error)
        st.session_state.orchestrator_error = cycle_error
        return

    flood, flood_error = get_flood_polygon(base_url)
    blocked, blocked_error = get_blocked_roads(base_url)
    route, route_error = get_safe_route(base_url)
    status, status_error = get_gis_status(base_url)

    st.session_state.flood_geojson = flood
    st.session_state.blocked_roads_geojson = blocked
    st.session_state.route_geojson = route
    st.session_state.orchestrator_status = status

    error_list = [err for err in [flood_error, blocked_error, route_error, status_error] if err]
    if error_list:
        st.session_state.orchestrator_error = "; ".join(error_list)
        st.warning(st.session_state.orchestrator_error)
    else:
        st.session_state.orchestrator_error = None
        st.success("Orchestrator data refreshed.")


with st.sidebar:
    st.header("Mission Controls")

    orchestrator_url = get_orchestrator_url()
    orchestrator_url = st.text_input(
        "Orchestrator API URL (optional)",
        value=orchestrator_url,
        placeholder="http://localhost:8000",
        help="Set ORCHESTRATOR_URL in secrets or environment for production.",
    )

    if not orchestrator_url:
        st.warning("Orchestrator URL is not configured. Running in mock mode.")

    use_orchestrator = st.checkbox(
        "Use orchestrator data", value=bool(orchestrator_url)
    )

    show_damage = st.checkbox("Show damage zones", value=True)
    show_flood = st.checkbox("Show flood zones", value=True)
    show_blocked = st.checkbox("Show blocked roads", value=False)
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

    disaster_type = st.selectbox(
        "Disaster type",
        ["flood", "earthquake", "fire", "landslide"],
        index=0,
    )
    uploaded_image = st.file_uploader(
        "Optional: upload a satellite image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "webp"],
        accept_multiple_files=False,
    )

    analyze_clicked = st.button("Analyze Image")
    if analyze_clicked:
        if not orchestrator_url:
            st.warning("Set an orchestrator URL to run image analysis.")
        else:
            analysis, analysis_error = analyze_image(
                orchestrator_url, uploaded_image, disaster_type
            )
            if analysis_error:
                st.warning(analysis_error)
            else:
                st.session_state.last_analysis = analysis
                geojson = analysis.get("geojson") if analysis else None
                if geojson:
                    st.session_state.damage_geojson = geojson
                    st.session_state.drawn_polygons = geojson_to_polygons(geojson)
                    st.success("Damage zones updated from AI analysis.")
                else:
                    st.warning("Analysis completed but no GeoJSON was returned.")

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
    sync_clicked = st.button("Sync Orchestrator Data")

    if mock_clicked:
        st.session_state.drawn_polygons = [
            shape(feature["geometry"]) for feature in SAMPLE_DAMAGE_GEOJSON["features"]
        ]
        st.info("Loaded sample damage polygon.")

    if mock_route_clicked:
        st.session_state.route_geojson = SAMPLE_ORCHESTRATOR_ROUTE_GEOJSON
        st.info("Loaded sample route from orchestrator.")

    if sync_clicked:
        sync_orchestrator_data(orchestrator_url)

    if run_clicked:
        if user_prompt.strip():
            st.session_state.status_log = STATUS_TEMPLATE
        else:
            st.session_state.status_log = [
                "Mission prompt is empty; routing still allowed."
            ]

        if use_orchestrator and orchestrator_url:
            sync_orchestrator_data(orchestrator_url)

    st.markdown("---")
    st.subheader("Route Statistics")
    stats = extract_route_stats(st.session_state.route_geojson)
    if not stats and st.session_state.orchestrator_status:
        stats = st.session_state.orchestrator_status.get("data", {})

    distance_km = stats.get("distance_km")
    if distance_km is None and "route_length_m" in stats:
        try:
            distance_km = round(float(stats["route_length_m"]) / 1000, 2)
        except (TypeError, ValueError):
            distance_km = None

    blocked_count = stats.get("blocked_edges")
    if blocked_count is None and st.session_state.blocked_roads_geojson:
        blocked_count = len(st.session_state.blocked_roads_geojson.get("features", []))

    if stats:
        st.write(f"- Distance: {distance_km if distance_km is not None else 'n/a'} km")
        st.write(f"- Travel time: {stats.get('travel_time_min', 'n/a')} min")
        st.write(f"- Blocked roads: {blocked_count if blocked_count is not None else 'n/a'}")
    else:
        st.write("Stats will appear after routing.")

    with st.expander("Raw GeoJSON Output"):
        st.write(
            export_geojson(
                st.session_state.route_geojson, st.session_state.damage_geojson
            )
        )

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

    render_flood_layer(base_map, st.session_state.flood_geojson, show_flood)
    render_blocked_roads(base_map, st.session_state.blocked_roads_geojson, show_blocked)

    if run_clicked and not (use_orchestrator and orchestrator_url):
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
