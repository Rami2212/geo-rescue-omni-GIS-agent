"""API routes for the GeoRescue AI service."""

import time
import torch
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from api.schemas import AnalysisResponse, HealthResponse
from qwen_vl.inference import analyze_image
from qwen_vl.geojson_generator import parse_zones_to_geojson

router = APIRouter()


@router.post("/analyze-image", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(..., description="Satellite or aerial disaster image"),
    disaster_type: str = Form("flood", description="Type: flood, earthquake, fire, etc.")
):
    """Analyze a disaster image and return affected zones as GeoJSON.
    
    Accepts an image upload, runs it through Qwen2-VL for disaster analysis,
    and returns identified danger zones as a GeoJSON FeatureCollection.
    """
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/tiff", "image/webp"]:
        raise HTTPException(status_code=400, detail="Unsupported image format. Use JPEG, PNG, TIFF, or WebP.")

    start = time.time()

    image_bytes = await file.read()
    ai_output = analyze_image(image_bytes, disaster_type)
    geojson_data, severity, findings = parse_zones_to_geojson(
        ai_output, metadata={"disaster_type": disaster_type}
    )

    elapsed = (time.time() - start) * 1000

    return AnalysisResponse(
        status="success",
        severity=severity,
        findings=findings or ai_output,
        geojson=geojson_data,
        inference_time_ms=round(elapsed, 2),
    )


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint for monitoring service status."""
    gpu_ok = torch.cuda.is_available()
    return HealthResponse(
        status="healthy",
        llama_status="running",
        qwen_status="running",
        gpu_available=gpu_ok,
    )
