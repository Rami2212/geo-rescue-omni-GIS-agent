import logging
from typing import Any, Dict, Optional, Tuple

import requests

logger = logging.getLogger("georescue-ui")

DEFAULT_TIMEOUT_SEC = 25


def _base_url(url: str) -> str:
    return url.rstrip("/")


def _get_json(url: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        response = requests.get(url, timeout=DEFAULT_TIMEOUT_SEC)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        logger.warning("API GET failed: %s", exc)
        return None, f"Request failed: {exc}"


def _post_json(url: str) -> Tuple[Optional[dict], Optional[str]]:
    try:
        response = requests.post(url, timeout=DEFAULT_TIMEOUT_SEC)
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        logger.warning("API POST failed: %s", exc)
        return None, f"Request failed: {exc}"


def run_gis_cycle(base_url: str) -> Tuple[Optional[dict], Optional[str]]:
    if not base_url:
        return None, "Orchestrator URL missing."
    return _post_json(f"{_base_url(base_url)}/gis/run-cycle")


def get_gis_status(base_url: str) -> Tuple[Optional[dict], Optional[str]]:
    if not base_url:
        return None, "Orchestrator URL missing."
    return _get_json(f"{_base_url(base_url)}/gis/status")


def get_flood_polygon(base_url: str) -> Tuple[Optional[dict], Optional[str]]:
    if not base_url:
        return None, "Orchestrator URL missing."
    return _get_json(f"{_base_url(base_url)}/gis/flood-polygon")


def get_blocked_roads(base_url: str) -> Tuple[Optional[dict], Optional[str]]:
    if not base_url:
        return None, "Orchestrator URL missing."
    return _get_json(f"{_base_url(base_url)}/gis/blocked-roads")


def get_safe_route(base_url: str) -> Tuple[Optional[dict], Optional[str]]:
    if not base_url:
        return None, "Orchestrator URL missing."
    return _get_json(f"{_base_url(base_url)}/gis/safe-route")


def analyze_image(
    base_url: str,
    image_file: Any,
    disaster_type: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not base_url:
        return None, "Orchestrator URL missing."
    if image_file is None:
        return None, "No image uploaded."

    content_type = getattr(image_file, "type", None) or "application/octet-stream"
    filename = getattr(image_file, "name", "upload")

    try:
        files = {"file": (filename, image_file.getvalue(), content_type)}
        data = {"disaster_type": disaster_type}
        response = requests.post(
            f"{_base_url(base_url)}/analyze-image",
            files=files,
            data=data,
            timeout=DEFAULT_TIMEOUT_SEC,
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as exc:
        logger.warning("Image analysis failed: %s", exc)
        return None, f"Image analysis failed: {exc}"

