"""GeoRescue UI utility package."""

from .config import (
    AGENT_ICONS,
    SAMPLE_DAMAGE_GEOJSON,
    SAMPLE_ROUTE_GEOJSON,
    get_settings,
)
from .geojson_utils import export_results, geojson_feature_count, geojson_to_str
from .logging_setup import setup_logging
from .map_layers import (
    load_base_map,
    render_blocked_roads,
    render_damage_layers,
    render_flood_zone,
    render_road_network,
    render_route_markers,
    render_safe_route,
)
from .routing import (
    calculate_safe_route,
    fetch_road_graph,
    parse_drawn_polygons,
)
from .state import (
    append_log,
    clear_log,
    get_dest,
    get_start,
    initialize_session_state,
    set_dest,
    set_start,
)

__all__ = [
    # config
    "get_settings",
    "AGENT_ICONS",
    "SAMPLE_DAMAGE_GEOJSON",
    "SAMPLE_ROUTE_GEOJSON",
    # logging
    "setup_logging",
    # state
    "initialize_session_state",
    "get_start",
    "get_dest",
    "set_start",
    "set_dest",
    "append_log",
    "clear_log",
    # map layers
    "load_base_map",
    "render_damage_layers",
    "render_flood_zone",
    "render_blocked_roads",
    "render_safe_route",
    "render_road_network",
    "render_route_markers",
    # routing
    "fetch_road_graph",
    "parse_drawn_polygons",
    "calculate_safe_route",
    # geojson
    "export_results",
    "geojson_feature_count",
    "geojson_to_str",
]
