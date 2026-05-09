"""Streamlit session-state initialisation and typed accessors."""

from typing import Optional

import streamlit as st

from georescue.config import get_settings


def initialize_session_state() -> None:
    """Initialise all session state keys with safe defaults on first run."""
    cfg = get_settings()

    # Map interaction
    _default("last_click", None)

    # Routing points
    _default("start_lat", cfg.default_start_lat)
    _default("start_lon", cfg.default_start_lon)
    _default("dest_lat", cfg.default_dest_lat)
    _default("dest_lon", cfg.default_dest_lon)

    # Drawn / computed layers
    _default("drawn_polygons", [])       # List[Shapely Polygon] from user draws
    _default("damage_geojson", None)     # GeoJSON FeatureCollection of drawn zones
    _default("route_geojson", None)      # GeoJSON from local OSMnx routing
    _default("flood_geojson", None)      # GeoJSON from GIS API (live flood zone)
    _default("blocked_geojson", None)    # GeoJSON from GIS API (blocked roads)
    _default("api_route_geojson", None)  # GeoJSON from GIS API safe route

    # Agent pipeline
    _default("status_log", [])           # List[str] — chronological agent messages
    _default("agent_report", "")         # Final Markdown report from Reporting Agent
    _default("pipeline_running", False)  # Guard against double-submission
    _default("vision_result", None)      # Dict from /analyze-image response
    _default("gis_severity", "")         # Latest flood severity string
    _default("route_stats", {})          # Dict with distance_km, travel_time_min, etc.


def _default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


# Typed convenience accessors (avoids repeated st.session_state["key"] noise)

def get_start() -> tuple[float, float]:
    return st.session_state.start_lat, st.session_state.start_lon


def get_dest() -> tuple[float, float]:
    return st.session_state.dest_lat, st.session_state.dest_lon


def set_start(lat: float, lon: float) -> None:
    st.session_state.start_lat = lat
    st.session_state.start_lon = lon


def set_dest(lat: float, lon: float) -> None:
    st.session_state.dest_lat = lat
    st.session_state.dest_lon = lon


def append_log(message: str) -> None:
    st.session_state.status_log.append(message)


def clear_log() -> None:
    st.session_state.status_log = []
