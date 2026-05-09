"""
GeoRescue CrewAI tools — thin, typed wrappers over the GIS / ML API server.

All tools handle connection errors and timeouts gracefully, returning a JSON
error payload instead of raising so the agent can self-correct.
"""

import json
import logging
import os
from functools import lru_cache

import requests
from requests.adapters import HTTPAdapter
from crewai.tools import tool
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP session with automatic retry
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.4,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _api_url() -> str:
    return os.getenv("GIS_API_URL", "http://localhost:9000")


def _get(path: str, timeout: int = 15) -> dict:
    url = f"{_api_url()}{path}"
    try:
        resp = _session().get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        logger.warning("API unreachable: %s", url)
        return {"error": "API server not reachable. Check GIS_API_URL and ensure the server is running.", "url": url}
    except requests.exceptions.Timeout:
        logger.warning("API timeout: %s", url)
        return {"error": f"Request timed out after {timeout}s", "url": url}
    except requests.exceptions.HTTPError as exc:
        return {"error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"}
    except Exception as exc:
        logger.exception("Unexpected API error")
        return {"error": str(exc)}


def _post(path: str, timeout: int = 120, **kwargs) -> dict:
    url = f"{_api_url()}{path}"
    try:
        resp = _session().post(url, timeout=timeout, **kwargs)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "API server not reachable. Check GIS_API_URL.", "url": url}
    except requests.exceptions.Timeout:
        return {"error": f"Request timed out after {timeout}s. The GIS cycle may still be running."}
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# CrewAI tools — one per endpoint
# ---------------------------------------------------------------------------

@tool("Check GeoRescue API Health")
def check_api_health() -> str:
    """
    Check the health and availability of all GeoRescue AI services.
    Returns the status of the Llama LLM server, Qwen-VL vision model, and GPU.
    Always call this first to confirm services are online before analysis.
    """
    result = _get("/health", timeout=10)
    return json.dumps(result, indent=2)


@tool("Trigger Live Flood Analysis Cycle")
def run_gis_cycle() -> str:
    """
    Trigger a fresh real-time flood analysis cycle.
    This fetches live weather data from Open-Meteo, generates a flood zone
    polygon based on rainfall severity, identifies which road segments are
    blocked by flooding, and computes an optimal safe driving route using
    NetworkX shortest-path on the OSMnx road graph.
    Returns flood severity level (low/moderate/high/extreme), number of affected
    roads, total affected road length in meters, and safe route length in meters.
    You MUST call this before fetching the flood polygon, blocked roads, or safe route.
    """
    result = _post("/gis/run-cycle", timeout=120)
    return json.dumps(result, indent=2)


@tool("Get Current Flood Situation Status")
def get_gis_status() -> str:
    """
    Retrieve the most recent flood cycle status without triggering a new run.
    Returns flood severity level, affected road count, route length, and timestamp.
    Use this to check whether fresh data already exists before running a new cycle.
    """
    result = _get("/gis/status")
    return json.dumps(result, indent=2)


@tool("Get Flood Zone Polygon Details")
def get_flood_polygon() -> str:
    """
    Retrieve the current flood zone as a summarized GeoJSON polygon.
    Returns severity level, timestamp, and zone count.
    Requires a prior call to 'Trigger Live Flood Analysis Cycle'.
    """
    result = _get("/gis/flood-polygon")
    if "error" in result:
        return json.dumps(result)
    features = result.get("features", [])
    if not features:
        return json.dumps({"error": "No flood polygon found. Run analysis cycle first."})
    props = features[0].get("properties", {})
    return json.dumps(
        {
            "flood_severity": props.get("severity", "unknown"),
            "analysis_timestamp": props.get("timestamp", "unknown"),
            "flood_zone_count": len(features),
            "geojson_available": True,
        },
        indent=2,
    )


@tool("Get Blocked Roads from Flooding")
def get_blocked_roads() -> str:
    """
    Retrieve road segments blocked by the current flood zone.
    Returns road names, highway classifications (primary/secondary/residential),
    and affected lengths in meters for each blocked segment.
    Use this to tell first responders which specific roads are impassable.
    Requires a prior call to 'Trigger Live Flood Analysis Cycle'.
    """
    result = _get("/gis/blocked-roads")
    if "error" in result:
        return json.dumps(result)
    features = result.get("features", [])
    total_length = 0.0
    roads = []
    for feat in features[:15]:
        props = feat.get("properties", {})
        length = float(props.get("affected_length_m", 0))
        total_length += length
        roads.append(
            {
                "road_name": props.get("name", "Unnamed Road"),
                "highway_type": props.get("highway", "unknown"),
                "affected_length_m": round(length, 1),
                "impact_level": props.get("impact_level", "blocked"),
            }
        )
    return json.dumps(
        {
            "total_blocked_roads": len(features),
            "total_affected_length_m": round(total_length, 1),
            "blocked_roads": roads,
        },
        indent=2,
    )


@tool("Get Safe Evacuation Route")
def get_safe_route() -> str:
    """
    Retrieve the currently computed safe route that avoids all flooded roads.
    Returns route length in km, segment count, and key street names.
    The route is computed using NetworkX shortest-path on the OSMnx drive graph.
    Requires a prior call to 'Trigger Live Flood Analysis Cycle'.
    """
    result = _get("/gis/safe-route")
    if "error" in result:
        return json.dumps(result)
    features = result.get("features", [])
    total_length = sum(f.get("properties", {}).get("length_m", 0) for f in features)
    streets = list(
        {
            f.get("properties", {}).get("street_name", "")
            for f in features
            if f.get("properties", {}).get("street_name")
        }
    )
    return json.dumps(
        {
            "safe_route_available": True,
            "total_route_length_km": round(total_length / 1000, 2) if total_length else 0,
            "route_segments": len(features),
            "key_streets": streets[:8],
        },
        indent=2,
    )


def analyze_satellite_image(image_bytes: bytes, disaster_type: str = "flood") -> dict:
    """
    Direct (non-tool) call to the /analyze-image endpoint.
    Used by the pipeline for image uploads — not exposed as a CrewAI tool
    because image bytes can't be passed as a string argument.
    """
    url = f"{_api_url()}/analyze-image"
    try:
        resp = _session().post(
            url,
            files={"file": ("satellite.jpg", image_bytes, "image/jpeg")},
            data={"disaster_type": disaster_type},
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Vision API not reachable. Qwen-VL analysis skipped."}
    except requests.exceptions.Timeout:
        return {"error": "Vision inference timed out (>180s). Image may be too large."}
    except Exception as exc:
        return {"error": str(exc)}


def is_api_healthy() -> bool:
    """Quick boolean health check used by the pipeline before starting agents."""
    result = _get("/health", timeout=8)
    return "error" not in result
