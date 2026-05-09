from .config import DEFAULT_CENTER_LAT, DEFAULT_CENTER_LON


def initialize_session_state() -> None:
    import streamlit as st

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
    if "blocked_roads_geojson" not in st.session_state:
        st.session_state.blocked_roads_geojson = None
    if "flood_geojson" not in st.session_state:
        st.session_state.flood_geojson = None
    if "orchestrator_status" not in st.session_state:
        st.session_state.orchestrator_status = None
    if "orchestrator_error" not in st.session_state:
        st.session_state.orchestrator_error = None
    if "last_analysis" not in st.session_state:
        st.session_state.last_analysis = None
