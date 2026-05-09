"""GIS pipeline endpoints — flood status, polygon, blocked roads, safe route."""

import json
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool

from api.schemas import GISStatusResponse, GISCycleResponse

router = APIRouter(prefix="/gis", tags=["GIS"])

_DATA = Path(__file__).parent.parent / "data" / "processed"


def _read_geojson(filename: str):
    path = _DATA / filename
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found. Run POST /gis/run-cycle first.",
        )
    return json.loads(path.read_text(encoding="utf-8"))


@router.get("/status", response_model=GISStatusResponse)
async def gis_status():
    """Latest flood-cycle status (severity, affected roads, route length)."""
    path = _DATA / "latest_status.json"
    if not path.exists():
        return GISStatusResponse(status="no_data")
    data = json.loads(path.read_text(encoding="utf-8"))
    return GISStatusResponse(status="ok", data=data)


@router.get("/flood-polygon")
async def flood_polygon():
    """Live flood polygon as GeoJSON (updated each cycle)."""
    return JSONResponse(_read_geojson("live_flood_polygon.geojson"))


@router.get("/blocked-roads")
async def blocked_roads():
    """Road segments blocked by the current flood zone as GeoJSON."""
    return JSONResponse(_read_geojson("blocked_roads_flood.geojson"))


@router.get("/safe-route")
async def safe_route():
    """Most recent safe route avoiding blocked roads as GeoJSON."""
    return JSONResponse(_read_geojson("latest_route.geojson"))


@router.post("/run-cycle", response_model=GISCycleResponse)
async def run_cycle():
    """Trigger a fresh flood-analysis cycle (fetches live weather, re-routes)."""
    from gis_pipeline.pipeline import run_cycle as _run

    start = time.time()
    try:
        status = await run_in_threadpool(_run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = (time.time() - start) * 1000
    return GISCycleResponse(
        status="ok",
        severity=status.get("severity"),
        affected_roads=status.get("affected_roads"),
        total_affected_length_m=status.get("total_affected_length_m"),
        route_length_m=status.get("route_length_m"),
        elapsed_ms=round(elapsed, 2),
    )
