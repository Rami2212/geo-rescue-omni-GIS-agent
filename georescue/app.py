"""
GeoRescue — Omni GIS Agent
Streamlit application entry point.

Integrates:
  • Member 1 (Imansha)  — CrewAI agent orchestration (this file + agents/)
  • Member 2 (Minindu)  — GIS tools / OSMnx routing (georescue/routing.py)
  • Member 3 (Supun)    — AMD MI300X ML serving / FastAPI (external server)
  • Member 4 (Ramitha)  — UI / Folium map (georescue/map_layers.py)

Run with:
    streamlit run app.py
"""

import streamlit as st
import folium
from streamlit_folium import st_folium

from georescue import (
    AGENT_ICONS,
    SAMPLE_DAMAGE_GEOJSON,
    SAMPLE_ROUTE_GEOJSON,
    append_log,
    calculate_safe_route,
    clear_log,
    export_results,
    fetch_road_graph,
    geojson_feature_count,
    geojson_to_str,
    get_dest,
    get_settings,
    get_start,
    initialize_session_state,
    load_base_map,
    parse_drawn_polygons,
    render_blocked_roads,
    render_damage_layers,
    render_flood_zone,
    render_road_network,
    render_route_markers,
    render_safe_route,
    set_dest,
    set_start,
    setup_logging,
)
from agents import run_pipeline_with_status
from shapely.geometry import shape

logger = setup_logging("georescue.app")
cfg = get_settings()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Page configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.set_page_config(
    page_title=cfg.app_title,
    page_icon=cfg.app_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

initialize_session_state()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Header
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.title(f"{cfg.app_icon} {cfg.app_title}")
st.caption(
    "Multi-agent disaster response platform · Colombo, Sri Lanka  "
    "| Satellite Vision (Qwen2-VL) · Live Flood GIS · Safe Routing · AMD MI300X + Ollama Llama 3.2"
)
st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sidebar — configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Map")
    zoom = st.slider("Zoom level", 9, 16, cfg.map_default_zoom)
    center_lat = st.number_input("Centre latitude", value=cfg.map_center_lat, format="%.4f")
    center_lon = st.number_input("Centre longitude", value=cfg.map_center_lon, format="%.4f")
    graph_radius_km = st.slider("Routing graph radius (km)", 4, 30, int(cfg.map_graph_radius_km))

    st.subheader("Layer Visibility")
    show_drawn_zones = st.checkbox("Drawn hazard zones", value=True)
    show_flood_zone = st.checkbox("Live flood zone (API)", value=True)
    show_blocked_roads = st.checkbox("Blocked roads (API)", value=True)
    show_route = st.checkbox("Safe route", value=True)
    show_road_network = st.checkbox("Road network (slow)", value=False)
    show_markers = st.checkbox("Start / destination pins", value=True)

    st.divider()
    st.subheader("Services")
    st.caption(f"GIS API: `{cfg.gis_api_url}`")
    st.caption(f"Ollama: `{cfg.ollama_base_url}` · model: `{cfg.ollama_model}`")

    st.divider()
    st.subheader("Agent Activity Log")
    if st.session_state.status_log:
        log_container = st.container(height=280)
        for entry in st.session_state.status_log:
            log_container.markdown(entry)
    else:
        st.caption("No agent activity yet.")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main layout — left (controls + report) | right (map)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

left_col, right_col = st.columns([1, 2], gap="large")

# ── LEFT: Mission controls ─────────────────────────────────────────────────

with left_col:
    st.subheader("📡 Mission Request")

    user_prompt = st.text_area(
        "Describe the incident and routing goal",
        value="Assess flood impact near Colombo Fort and compute a safe evacuation route.",
        height=110,
        help="Natural language prompt processed by the AI agent crew.",
    )

    disaster_type = st.selectbox(
        "Disaster type",
        ["flood", "earthquake", "landslide", "fire"],
        index=0,
    )

    uploaded_file = st.file_uploader(
        "Satellite image (optional — activates Vision Analyst)",
        type=["png", "jpg", "jpeg", "tif", "webp"],
        help="Upload a pre/post-disaster satellite or aerial image for Qwen2-VL analysis.",
    )

    st.divider()

    # Routing points
    st.subheader("📍 Routing Points")

    col_s, col_d = st.columns(2)
    with col_s:
        st.session_state.start_lat = st.number_input(
            "Start lat", value=st.session_state.start_lat, format="%.5f", key="inp_slat"
        )
        st.session_state.start_lon = st.number_input(
            "Start lon", value=st.session_state.start_lon, format="%.5f", key="inp_slon"
        )
    with col_d:
        st.session_state.dest_lat = st.number_input(
            "Dest lat", value=st.session_state.dest_lat, format="%.5f", key="inp_dlat"
        )
        st.session_state.dest_lon = st.number_input(
            "Dest lon", value=st.session_state.dest_lon, format="%.5f", key="inp_dlon"
        )

    click_col1, click_col2 = st.columns(2)
    if click_col1.button("📌 Set start from click"):
        if st.session_state.last_click:
            set_start(st.session_state.last_click["lat"], st.session_state.last_click["lng"])
            st.rerun()
    if click_col2.button("🎯 Set dest from click"):
        if st.session_state.last_click:
            set_dest(st.session_state.last_click["lat"], st.session_state.last_click["lng"])
            st.rerun()

    st.divider()

    # Action buttons
    st.subheader("🚀 Actions")

    run_btn = st.button(
        "▶ Run Agent Pipeline",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.pipeline_running,
        help="Launches the full 4-agent CrewAI pipeline: Vision → Data Scout → Spatial Navigator → Report",
    )

    col_a, col_b = st.columns(2)
    load_sample_btn = col_a.button("Load Sample Hazard", use_container_width=True)
    local_route_btn = col_b.button("Local Route Only", use_container_width=True)

    if load_sample_btn:
        st.session_state.drawn_polygons = [
            shape(f["geometry"]) for f in SAMPLE_DAMAGE_GEOJSON["features"]
        ]
        st.info("Loaded sample damage polygon.")

    st.divider()

    # Route statistics
    st.subheader("📊 Route Statistics")
    stats = st.session_state.route_stats
    if stats:
        m1, m2 = st.columns(2)
        m1.metric("Distance", f"{stats.get('distance_km', '—')} km")
        m2.metric("Travel time", f"{stats.get('travel_time_min', '—')} min")
        m3, m4 = st.columns(2)
        m3.metric("Blocked edges", stats.get("blocked_edges", "—"))
        m4.metric("Severity", st.session_state.gis_severity.upper() or "—")
    else:
        st.caption("Stats appear after routing.")

    # Export
    with st.expander("📥 Export GeoJSON", expanded=False):
        export_data = export_results(
            st.session_state.route_geojson,
            st.session_state.damage_geojson,
            st.session_state.flood_geojson,
            st.session_state.blocked_geojson,
        )
        st.download_button(
            "Download results.json",
            data=geojson_to_str(export_data),
            file_name="georescue_results.json",
            mime="application/json",
            use_container_width=True,
        )

# ── RIGHT: Map ─────────────────────────────────────────────────────────────

with right_col:
    st.subheader("🗺️ Operational Map")

    map_center = (center_lat, center_lon)
    base_map = load_base_map(map_center, zoom)

    # Fetch road graph (cached)
    graph = fetch_road_graph(map_center, float(graph_radius_km))

    # Road network overlay (optional)
    if show_road_network:
        render_road_network(base_map, graph)

    # Layers from session state
    damage_geojson = render_damage_layers(
        base_map, st.session_state.drawn_polygons, show_drawn_zones
    )
    if damage_geojson:
        st.session_state.damage_geojson = damage_geojson

    render_flood_zone(base_map, st.session_state.flood_geojson, show_flood_zone)
    render_blocked_roads(base_map, st.session_state.blocked_geojson, show_blocked_roads)

    # Active route (prefer API route, fallback to local)
    active_route = st.session_state.api_route_geojson or st.session_state.route_geojson
    render_safe_route(base_map, active_route, visible=show_route)

    if show_markers:
        render_route_markers(base_map, get_start(), get_dest())

    folium.LayerControl(collapsed=False).add_to(base_map)

    map_output = st_folium(base_map, use_container_width=True, height=560)

    # Capture map interactions
    if map_output:
        if map_output.get("last_clicked"):
            st.session_state.last_click = map_output["last_clicked"]
        drawings = map_output.get("all_drawings", []) or []
        drawn = parse_drawn_polygons(drawings)
        if drawn:
            st.session_state.drawn_polygons = drawn

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent pipeline execution (runs after map render to avoid layout jump)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if run_btn:
    clear_log()
    st.session_state.pipeline_running = True
    image_bytes = uploaded_file.read() if uploaded_file else None

    with st.status("🤖 GeoRescue Agent Pipeline Running…", expanded=True) as pipeline_status:
        final_data: dict = {}

        for update in run_pipeline_with_status(user_prompt, image_bytes, disaster_type):
            icon = AGENT_ICONS.get(update.agent, "🔄")
            colour = {
                "running": "🔵",
                "done": "✅",
                "warning": "⚠️",
                "error": "❌",
                "complete": "🎯",
            }.get(update.status, "•")

            line = f"{colour} **{icon} {update.agent}:** {update.message}"
            st.write(line)
            append_log(line)

            if update.status == "complete" and update.data:
                final_data = update.data

        pipeline_status.update(
            label="✅ Mission complete — map and report updated.",
            state="complete",
            expanded=False,
        )

    # Apply results to session state
    if final_data:
        if final_data.get("flood_geojson"):
            st.session_state.flood_geojson = final_data["flood_geojson"]
        if final_data.get("blocked_geojson"):
            st.session_state.blocked_geojson = final_data["blocked_geojson"]
        if final_data.get("route_geojson"):
            st.session_state.api_route_geojson = final_data["route_geojson"]
        if final_data.get("report"):
            st.session_state.agent_report = final_data["report"]
        result = final_data.get("result")
        if result:
            st.session_state.gis_severity = getattr(result, "severity", "")
            st.session_state.route_stats = {
                "distance_km": getattr(result, "route_length_km", "—"),
                "travel_time_min": "—",
                "blocked_edges": "—",
            }

    st.session_state.pipeline_running = False
    st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Local-only route (no API required)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if local_route_btn:
    with st.spinner("Computing local safe route via OSMnx + NetworkX…"):
        route_geojson, _, stats, error = calculate_safe_route(
            graph,
            get_start(),
            get_dest(),
            st.session_state.drawn_polygons,
        )
    if error:
        st.warning(f"Routing error: {error}")
    else:
        route_geojson["features"][0]["properties"].update(stats)
        st.session_state.route_geojson = route_geojson
        st.session_state.route_stats = stats
        st.success(
            f"Local route computed: {stats['distance_km']} km · {stats['travel_time_min']} min"
        )
        st.rerun()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Agent report panel (full width, below map)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if st.session_state.agent_report:
    st.divider()
    report_col, meta_col = st.columns([3, 1], gap="large")

    with report_col:
        st.subheader("📋 Emergency Response Report")
        st.markdown(st.session_state.agent_report)

    with meta_col:
        st.subheader("🔬 Analysis Metadata")

        vision = st.session_state.vision_result
        if vision and "error" not in vision:
            st.markdown("**Vision Analysis**")
            st.markdown(f"- Severity: `{vision.get('severity', 'n/a')}`")
            st.markdown(f"- Inference: `{vision.get('inference_time_ms', '?')} ms`")
            st.divider()

        st.markdown("**Spatial Data**")
        st.markdown(
            f"- Flood zones: `{geojson_feature_count(st.session_state.flood_geojson)}`"
        )
        st.markdown(
            f"- Blocked roads: `{geojson_feature_count(st.session_state.blocked_geojson)}`"
        )
        route_src = "API" if st.session_state.api_route_geojson else "Local OSMnx"
        st.markdown(f"- Route source: `{route_src}`")

        st.divider()
        st.markdown("**Infrastructure**")
        st.markdown(f"- Vision: Qwen2-VL-7B")
        st.markdown(f"- LLM: Ollama / {cfg.ollama_model}")
        st.markdown(f"- GPU: AMD Instinct MI300X")
        st.markdown(f"- Roads: OpenStreetMap (OSMnx)")
