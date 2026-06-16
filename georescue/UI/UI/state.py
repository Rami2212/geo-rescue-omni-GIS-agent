from .config import DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON


def initialize_session_state() -> None:
    import streamlit as st

    if "damage_geojson" not in st.session_state:
        st.session_state.damage_geojson = None
    if "flood_geojson" not in st.session_state:
        st.session_state.flood_geojson = None
    if "blocked_geojson" not in st.session_state:
        st.session_state.blocked_geojson = None
    if "route_geojson" not in st.session_state:
        st.session_state.route_geojson = None
    if "alternative_routes_geojson" not in st.session_state:
        st.session_state.alternative_routes_geojson = []
    if "status_log" not in st.session_state:
        st.session_state.status_log = []
    if "vision_result" not in st.session_state:
        st.session_state.vision_result = None
    if "supervisor_plan" not in st.session_state:
        st.session_state.supervisor_plan = ""
    if "supervisor_summary" not in st.session_state:
        st.session_state.supervisor_summary = ""
    if "realtime_exports" not in st.session_state:
        st.session_state.realtime_exports = None
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

