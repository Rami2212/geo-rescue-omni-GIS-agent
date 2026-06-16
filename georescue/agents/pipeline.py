from __future__ import annotations

import time
from typing import Generator

from .orchestrator import LlamaOrchestrator
from .routing import RouteAgent
from .schemas import AgentUpdate, BackendRunRequest
from .vision import FloodVisionAgent


def run_backend_pipeline_stream(
    request: BackendRunRequest,
) -> Generator[dict, None, None]:
    start_time = time.time()
    orchestrator = LlamaOrchestrator()
    vision_agent = FloodVisionAgent()
    route_agent = RouteAgent()

    yield AgentUpdate(
        "Supervisor (HF Llama 3B)",
        "running",
        "Planning flood-image analysis and safe-route generation.",
    ).as_dict()
    supervisor = orchestrator.plan(request)
    yield AgentUpdate(
        "Supervisor (HF Llama 3B)",
        "done",
        "Mission plan ready.",
        supervisor,
    ).as_dict()

    yield AgentUpdate(
        "Vision Agent (HF Qwen2.5-VL + local LoRA)",
        "running",
        "Running Qwen2.5-VL with the local LoRA adapter for flood polygon extraction.",
    ).as_dict()
    vision_result = vision_agent.analyze(request)
    flood_geojson = vision_result.get("flood_geojson")
    feature_count = len((flood_geojson or {}).get("features", []))
    yield AgentUpdate(
        "Vision Agent (HF Qwen2.5-VL + local LoRA)",
        "done",
        f"Flood polygon extraction complete: {feature_count} polygon(s), severity {vision_result.get('severity', 'unknown')}.",
        vision_result,
    ).as_dict()

    yield AgentUpdate(
        "Route Agent",
        "running",
        "Computing safe and alternative routes that avoid detected flood polygons.",
    ).as_dict()
    route_result = route_agent.build_routes(request, flood_geojson or {})
    stats = route_result.get("route_stats", {})
    yield AgentUpdate(
        "Route Agent",
        "done",
        (
            f"Route ready: {stats.get('distance_km', 'n/a')} km, "
            f"{stats.get('blocked_edges', 0)} flooded road edge(s) avoided."
        ),
        route_result,
    ).as_dict()

    summary = orchestrator.summarize(request, vision_result, stats)
    elapsed_sec = round(time.time() - start_time, 1)
    payload = {
        "flood_geojson": flood_geojson,
        "blocked_geojson": route_result.get("blocked_geojson"),
        "primary_route_geojson": route_result.get("primary_route_geojson"),
        "alternative_routes_geojson": route_result.get("alternative_routes_geojson", []),
        "route_stats": stats,
        "vision_result": {
            key: value
            for key, value in vision_result.items()
            if key not in {"flood_geojson", "raw_model_response"}
        },
        "supervisor_plan": supervisor.get("plan", ""),
        "supervisor_summary": summary.get("summary", ""),
        "realtime_exports": {
            "elapsed_sec": elapsed_sec,
            "vision_method": vision_result.get("method"),
            "vision_model": vision_result.get("model"),
            "vision_base_model": vision_result.get("base_model"),
            "vision_adapter_dir": vision_result.get("adapter_dir"),
            "georeferencing": vision_result.get("georeferencing"),
        },
    }

    yield AgentUpdate(
        "Supervisor (HF Llama 3B)",
        "complete",
        f"Flood response package generated in {elapsed_sec}s.",
        payload,
    ).as_dict()


def run_pipeline_with_status(*args, **kwargs):
    """Backward-compatible alias for older UI imports."""
    yield from run_backend_pipeline_stream(*args, **kwargs)
