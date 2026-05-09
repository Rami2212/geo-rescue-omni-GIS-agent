"""
GeoRescue main orchestration pipeline.

Runs a four-step agent pipeline and yields typed status updates so the
Streamlit UI can display live agent progress. Gracefully degrades when
the GIS API or Ollama server is unavailable:

  API available + LLM available → full CrewAI pipeline + live GIS data
  API available + LLM offline   → direct GIS API calls + template report
  API offline                   → local OSMnx routing + template report
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Generator, Optional

import requests

from agents.tools import analyze_satellite_image, is_api_healthy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class AgentUpdate:
    """A single status event emitted during the pipeline run."""
    agent: str
    status: str        # "running" | "done" | "warning" | "error" | "complete"
    message: str
    data: Optional[dict] = None


@dataclass
class PipelineResult:
    """Aggregated outputs from a completed pipeline run."""
    report: str = ""
    flood_geojson: Optional[dict] = None
    blocked_geojson: Optional[dict] = None
    route_geojson: Optional[dict] = None
    vision_result: Optional[dict] = None
    gis_status: Optional[dict] = None
    route_length_km: float = 0.0
    severity: str = "unknown"
    elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gis_url() -> str:
    return os.getenv("GIS_API_URL", "http://localhost:9000")


def _ollama_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


def _is_ollama_available() -> bool:
    try:
        r = requests.get(f"{_ollama_url()}/api/tags", timeout=6)
        return r.status_code == 200
    except Exception:
        return False


def _http_get(path: str, timeout: int = 20) -> Optional[dict]:
    try:
        r = requests.get(f"{_gis_url()}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("GET %s failed: %s", path, exc)
        return None


def _http_post(path: str, timeout: int = 120, **kwargs) -> Optional[dict]:
    try:
        r = requests.post(f"{_gis_url()}{path}", timeout=timeout, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        logger.warning("POST %s failed: %s", path, exc)
        return None


def _route_length_km(route_geojson: Optional[dict]) -> float:
    if not route_geojson:
        return 0.0
    features = route_geojson.get("features", [])
    total_m = sum(f.get("properties", {}).get("length_m", 0) for f in features)
    return round(total_m / 1000, 2)


def _build_template_report(
    user_prompt: str,
    gis_status: Optional[dict],
    vision_result: Optional[dict],
    route_length_km: float,
) -> str:
    """Generate a structured report from raw API data when the LLM is unavailable."""
    severity = "unknown"
    affected_roads = "N/A"
    timestamp = time.strftime("%Y-%m-%d %H:%M UTC")

    if gis_status:
        data = gis_status.get("data", gis_status)
        severity = data.get("severity", severity)
        affected_roads = data.get("affected_roads", affected_roads)

    vision_section = ""
    if vision_result and "error" not in vision_result:
        findings = vision_result.get("findings", "")[:400]
        vision_section = f"\n- **Satellite Image Analysis:** {findings}"

    route_str = f"{route_length_km} km" if route_length_km else "Calculating via local routing"

    return (
        "## GEORESCUE EMERGENCY RESPONSE REPORT\n"
        f"**Mission:** {user_prompt}  \n"
        f"**Generated:** {timestamp}\n\n"
        "---\n\n"
        "### 1. SITUATION SUMMARY\n"
        f"- **Flood Severity:** {severity.upper()}\n"
        f"- **Analysis Area:** Colombo, Sri Lanka (real-time Open-Meteo data){vision_section}\n\n"
        "### 2. AFFECTED INFRASTRUCTURE\n"
        f"- **Blocked Road Segments:** {affected_roads}\n"
        "- Flood zone intersected with OSMnx road network\n"
        "- Roads classified by highway type impact level\n\n"
        "### 3. SAFE CORRIDOR\n"
        f"- **Route Length:** {route_str}\n"
        "- Optimal path computed avoiding all flooded segments\n"
        "- Route visualised on map as green overlay\n\n"
        "### 4. RECOMMENDED ACTIONS\n"
        "- Route all emergency vehicles via the highlighted safe corridor\n"
        "- Avoid roads shown in red on the operational map\n"
        f"- Current flood severity is **{severity}** — monitor for escalation\n"
        "- Re-run analysis cycle every 30 minutes for updated conditions\n"
        "- Coordinate with Colombo Municipal Council for evacuation priorities\n\n"
        "### 5. DATA CONFIDENCE & TIMESTAMP\n"
        f"- **Analysis Time:** {timestamp}\n"
        "- **Weather Source:** Open-Meteo real-time API\n"
        "- **Road Network:** OpenStreetMap via OSMnx\n"
        "- **Vision Model:** Qwen2-VL-7B on AMD MI300X\n"
        "- **Report Mode:** Template (LLM offline — start Ollama for AI-generated reports)\n"
    )


# ---------------------------------------------------------------------------
# Public pipeline
# ---------------------------------------------------------------------------

def run_pipeline_with_status(
    user_prompt: str,
    image_bytes: Optional[bytes] = None,
    disaster_type: str = "flood",
) -> Generator[AgentUpdate, None, None]:
    """
    Run the full GeoRescue agent pipeline and yield real-time status updates.

    This generator is consumed by the Streamlit UI to display live agent
    progress. The final `AgentUpdate` (status="complete") carries the
    full `PipelineResult` in its `data` field.

    Args:
        user_prompt:   Natural language mission description.
        image_bytes:   Optional raw satellite image bytes for Qwen-VL analysis.
        disaster_type: Disaster category passed to the vision model.

    Yields:
        AgentUpdate instances in chronological order.
    """
    t_start = time.time()
    result = PipelineResult()

    # ── Pre-flight checks ──────────────────────────────────────────────────
    api_ok = is_api_healthy()
    llm_ok = _is_ollama_available()

    yield AgentUpdate(
        agent="Supervisor",
        status="running",
        message=(
            f"Parsing mission: '{user_prompt[:80]}{'...' if len(user_prompt) > 80 else ''}' | "
            f"GIS API: {'✓ online' if api_ok else '✗ offline'} | "
            f"Ollama LLM: {'✓ online' if llm_ok else '✗ offline (template mode)'}"
        ),
    )

    # ── Step 1: Vision Analyst (Qwen-VL) ──────────────────────────────────
    vision_findings = ""
    if image_bytes:
        if api_ok:
            yield AgentUpdate(
                agent="Vision Analyst",
                status="running",
                message="Uploading satellite image to Qwen2-VL-7B for damage detection...",
            )
            vision_data = analyze_satellite_image(image_bytes, disaster_type)
            result.vision_result = vision_data

            if "error" in vision_data:
                yield AgentUpdate(
                    agent="Vision Analyst",
                    status="warning",
                    message=f"Vision analysis failed: {vision_data['error']}",
                    data=vision_data,
                )
            else:
                severity_img = vision_data.get("severity", "unknown")
                vision_findings = vision_data.get("findings", "")[:500]
                yield AgentUpdate(
                    agent="Vision Analyst",
                    status="done",
                    message=f"Image analysis complete — severity: {severity_img} | inference: {vision_data.get('inference_time_ms', '?')} ms",
                    data=vision_data,
                )
        else:
            yield AgentUpdate(
                agent="Vision Analyst",
                status="warning",
                message="GIS API offline — satellite image analysis skipped.",
            )
    else:
        yield AgentUpdate(
            agent="Vision Analyst",
            status="done",
            message="No satellite image provided — proceeding with live weather data only.",
        )

    # ── Step 2: Data Scout (live GIS cycle) ────────────────────────────────
    if api_ok:
        yield AgentUpdate(
            agent="Data Scout",
            status="running",
            message="Fetching live weather from Open-Meteo and triggering flood analysis cycle...",
        )
        cycle = _http_post("/gis/run-cycle", timeout=120)
        if cycle and "error" not in cycle:
            result.gis_status = cycle
            result.severity = cycle.get("severity", "unknown")
            affected = cycle.get("affected_roads", "?")
            route_m = cycle.get("route_length_m")
            result.route_length_km = round(route_m / 1000, 2) if route_m else 0.0
            yield AgentUpdate(
                agent="Data Scout",
                status="done",
                message=(
                    f"Flood cycle complete — severity: {result.severity.upper()} | "
                    f"affected roads: {affected} | "
                    f"route: {result.route_length_km} km"
                ),
                data=cycle,
            )
        else:
            err = cycle.get("error", "unknown error") if cycle else "no response"
            yield AgentUpdate(
                agent="Data Scout",
                status="warning",
                message=f"GIS cycle failed: {err}. Attempting to use cached data.",
            )
            cached = _http_get("/gis/status")
            if cached and cached.get("status") == "ok":
                result.gis_status = cached.get("data")
                yield AgentUpdate(
                    agent="Data Scout",
                    status="done",
                    message="Using cached GIS data from previous cycle.",
                    data=cached,
                )
    else:
        yield AgentUpdate(
            agent="Data Scout",
            status="warning",
            message="GIS API offline — using local OSMnx routing only. Start the API server for live analysis.",
        )

    # ── Step 3: Spatial Navigator ──────────────────────────────────────────
    if api_ok:
        yield AgentUpdate(
            agent="Spatial Navigator",
            status="running",
            message="Retrieving flood polygon, blocked road segments, and safe route...",
        )
        flood = _http_get("/gis/flood-polygon")
        blocked = _http_get("/gis/blocked-roads")
        route = _http_get("/gis/safe-route")

        result.flood_geojson = flood
        result.blocked_geojson = blocked
        result.route_geojson = route

        if route:
            result.route_length_km = _route_length_km(route)

        blocked_count = len((blocked or {}).get("features", []))
        yield AgentUpdate(
            agent="Spatial Navigator",
            status="done",
            message=(
                f"Spatial layers fetched — flood zones: {len((flood or {}).get('features', []))} | "
                f"blocked roads: {blocked_count} | "
                f"route: {result.route_length_km} km"
            ),
            data={
                "flood_geojson": flood,
                "blocked_geojson": blocked,
                "route_geojson": route,
            },
        )
    else:
        yield AgentUpdate(
            agent="Spatial Navigator",
            status="warning",
            message="GIS API offline — map will use local OSMnx routing. Flood and blocked-road layers unavailable.",
        )

    # ── Step 4: Reporting Coordinator ─────────────────────────────────────
    yield AgentUpdate(
        agent="Reporting Coordinator",
        status="running",
        message=(
            "Generating emergency response report via Ollama Llama 3.2..."
            if llm_ok
            else "Generating template report (Ollama offline)..."
        ),
    )

    if llm_ok and api_ok:
        try:
            from agents.crew import run_crew
            report = run_crew(user_prompt, vision_findings)
            result.report = report
            yield AgentUpdate(
                agent="Reporting Coordinator",
                status="done",
                message="AI-generated report ready.",
                data={"report": report},
            )
        except Exception as exc:
            logger.warning("CrewAI failed, falling back to template: %s", exc)
            report = _build_template_report(
                user_prompt, result.gis_status, result.vision_result, result.route_length_km
            )
            result.report = report
            yield AgentUpdate(
                agent="Reporting Coordinator",
                status="warning",
                message=f"LLM crew failed ({exc}). Template report generated.",
                data={"report": report},
            )
    else:
        report = _build_template_report(
            user_prompt, result.gis_status, result.vision_result, result.route_length_km
        )
        result.report = report
        yield AgentUpdate(
            agent="Reporting Coordinator",
            status="done",
            message="Template report generated. Start Ollama for AI-generated reports.",
            data={"report": report},
        )

    # ── Final event ────────────────────────────────────────────────────────
    result.elapsed_sec = round(time.time() - t_start, 1)
    yield AgentUpdate(
        agent="Supervisor",
        status="complete",
        message=f"Mission complete in {result.elapsed_sec}s.",
        data={
            "result": result,
            "report": result.report,
            "flood_geojson": result.flood_geojson,
            "blocked_geojson": result.blocked_geojson,
            "route_geojson": result.route_geojson,
            "vision_result": result.vision_result,
        },
    )
