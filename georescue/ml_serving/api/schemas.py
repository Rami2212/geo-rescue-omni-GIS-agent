"""Pydantic schemas for the GeoRescue AI API."""

from pydantic import BaseModel
from typing import Optional


class AnalysisResponse(BaseModel):
    """Response from the /analyze-image endpoint."""
    status: str
    severity: Optional[str] = None
    findings: Optional[str] = None
    geojson: Optional[dict] = None
    inference_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""
    status: str
    llama_status: str
    qwen_status: str
    gpu_available: bool


class GISStatusResponse(BaseModel):
    """Response from GET /gis/status."""
    status: str           # "ok" | "no_data"
    data: Optional[dict] = None


class GISCycleResponse(BaseModel):
    """Response from POST /gis/run-cycle."""
    status: str
    severity: Optional[str] = None
    affected_roads: Optional[int] = None
    total_affected_length_m: Optional[float] = None
    route_length_m: Optional[float] = None
    elapsed_ms: Optional[float] = None
