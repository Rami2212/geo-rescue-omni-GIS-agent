"""
Application configuration via pydantic-settings.

All values are read from environment variables (or .env file).
Field names map directly to uppercase env var names, e.g.:
    map_center_lat  →  MAP_CENTER_LAT
"""

import functools

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GeoRescueSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Map defaults ────────────────────────────────────────────────────
    map_center_lat: float = Field(6.9271, description="Default map center latitude (Colombo Fort)")
    map_center_lon: float = Field(79.8612, description="Default map center longitude")
    map_default_zoom: int = Field(12, description="Default Folium zoom level")
    map_graph_radius_km: float = Field(12.0, description="OSMnx road graph fetch radius in km")
    default_speed_kmh: float = Field(30.0, description="Assumed vehicle speed for travel-time estimates")

    # ── GIS / ML API (Member 3 — Supun) ─────────────────────────────────
    gis_api_url: str = Field("http://localhost:9000", description="FastAPI GIS & vision server URL")
    gis_api_timeout_short: int = Field(15, description="Short HTTP timeout (health checks, status)")
    gis_api_timeout_long: int = Field(120, description="Long HTTP timeout (GIS cycle, image analysis)")

    # ── Ollama LLM (Member 1 — Imansha) ─────────────────────────────────
    ollama_base_url: str = Field("http://localhost:11434", description="Ollama server base URL")
    ollama_model: str = Field("llama3.2", description="Ollama model name, e.g. llama3.2 / llama3.1:8b")
    llm_temperature: float = Field(0.1, description="LLM sampling temperature (low = deterministic)")
    llm_max_tokens: int = Field(2048, description="Maximum tokens per LLM response")

    # ── App metadata ─────────────────────────────────────────────────────
    app_title: str = Field("GeoRescue — Omni GIS Agent", description="Streamlit page title")
    app_icon: str = Field("🛰️", description="Streamlit page icon")
    log_level: str = Field("INFO", description="Logging level: DEBUG / INFO / WARNING / ERROR")

    # ── Routing defaults (Colombo) ───────────────────────────────────────
    default_start_lat: float = Field(6.9171, description="Default route start latitude")
    default_start_lon: float = Field(79.8512, description="Default route start longitude")
    default_dest_lat: float = Field(6.9371, description="Default route destination latitude")
    default_dest_lon: float = Field(79.8712, description="Default route destination longitude")


@functools.lru_cache(maxsize=1)
def get_settings() -> GeoRescueSettings:
    """Return the singleton settings instance (loaded once, cached forever)."""
    return GeoRescueSettings()


# ---------------------------------------------------------------------------
# Static UI content (not env-configurable — intentional)
# ---------------------------------------------------------------------------

AGENT_ICONS: dict[str, str] = {
    "Supervisor": "🎯",
    "Vision Analyst": "👁️",
    "Data Scout": "📡",
    "Spatial Navigator": "🗺️",
    "Reporting Coordinator": "📋",
}

SAMPLE_DAMAGE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Sample Flood Zone — Colombo Fort"},
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

SAMPLE_ROUTE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "name": "Sample Safe Route",
                "distance_km": 4.2,
                "travel_time_min": 8.4,
                "blocked_edges": 3,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [79.8468, 6.9257],
                    [79.8545, 6.9242],
                    [79.8641, 6.9279],
                    [79.8728, 6.9315],
                    [79.8792, 6.9341],
                ],
            },
        }
    ],
}
