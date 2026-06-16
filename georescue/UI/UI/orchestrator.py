import os
import streamlit as st

from georescue.agents import BackendRunRequest, run_backend_pipeline_stream


def get_orchestrator_url() -> str:
    try:
        return st.secrets.get("ORCHESTRATOR_URL", "")
    except Exception:
        return os.getenv("ORCHESTRATOR_URL", "")


def run_agents_backend_stream(
    mission: str,
    disaster_type: str,
    start: tuple[float, float],
    dest: tuple[float, float],
    uploaded_image_bytes: bytes | None = None,
    realtime_image_dir: str | None = None,
    map_center: tuple[float, float] | None = None,
    graph_radius_km: float | None = None,
):
    """
    UI-facing wrapper over backend agents.
    Keeps model/orchestration logic out of UI layer.
    """
    req = BackendRunRequest(
        mission=mission,
        disaster_type=disaster_type,
        start=start,
        dest=dest,
        uploaded_image_bytes=uploaded_image_bytes,
        realtime_image_dir=realtime_image_dir,
        map_center=map_center,
        graph_radius_km=graph_radius_km,
    )
    yield from run_backend_pipeline_stream(req)
