"""
GeoRescue AI Orchestration Layer — Member 1 (Team Lead / AI Orchestrator).

Implements a four-agent CrewAI pipeline backed by Ollama (Llama 3.2):
  1. Data Scout          — triggers live flood cycle, checks service health
  2. Vision Analyst      — analyzes satellite imagery via Qwen-VL API
  3. Spatial Navigator   — retrieves flood polygon, blocked roads, safe route
  4. Reporting Coordinator — synthesizes findings into an actionable report

Public API:
    run_pipeline_with_status(prompt, image_bytes, disaster_type)
        → Generator[AgentUpdate, None, None]
"""

from .pipeline import AgentUpdate, PipelineResult, run_pipeline_with_status

__all__ = [
    "AgentUpdate",
    "PipelineResult",
    "run_pipeline_with_status",
]
