from __future__ import annotations

import json
import os
from typing import Optional

from .hf_client import HFChatClient
from .schemas import BackendRunRequest


class LlamaOrchestrator:
    """Optional lightweight supervisor using a Hugging Face-hosted Llama 3B model."""

    def __init__(self) -> None:
        self.model = os.getenv(
            "GEORESCUE_ORCHESTRATOR_MODEL",
            "meta-llama/Llama-3.2-3B-Instruct",
        )
        self.client = HFChatClient(
            self.model,
            timeout=int(os.getenv("GEORESCUE_ORCHESTRATOR_TIMEOUT", "60")),
        )

    def plan(self, request: BackendRunRequest) -> dict:
        prompt = (
            "You are GeoRescue's disaster-response supervisor. Create a concise "
            "execution plan for these agents: Vision Agent, Route Agent, Reporting Agent.\n\n"
            f"Mission: {request.mission}\n"
            f"Disaster type: {request.disaster_type}\n"
            f"Start: {request.start}\n"
            f"Destination: {request.dest}\n"
            "Return 3 short bullet points."
        )
        text = self._generate(prompt)
        if text:
            return {
                "plan": text,
                "summary": f"Supervisor model: {self.client.model} via Hugging Face",
            }
        return {
            "plan": (
                "- Detect flood extent as an irregular polygon from the satellite image.\n"
                "- Intersect the polygon with the road graph and remove flooded edges.\n"
                "- Generate a primary safe route plus alternatives and package GeoJSON for the UI."
            ),
            "summary": "Template supervisor plan used because Hugging Face Llama 3B was unavailable.",
        }

    def summarize(
        self, request: BackendRunRequest, vision_result: dict, route_stats: dict
    ) -> dict:
        prompt = (
            "Summarize this flood-routing result for an emergency commander in "
            "two sentences.\n\n"
            f"Mission: {request.mission}\n"
            f"Vision: {json.dumps(vision_result, default=str)[:1500]}\n"
            f"Route stats: {json.dumps(route_stats, default=str)}"
        )
        text = self._generate(prompt)
        if text:
            return {"summary": text}

        severity = vision_result.get("severity", "unknown")
        distance = route_stats.get("distance_km", "n/a")
        blocked = route_stats.get("blocked_edges", "n/a")
        return {
            "summary": (
                f"Detected flood severity is {severity}; routing avoided "
                f"{blocked} flooded road edges. Primary route distance is {distance} km."
            )
        }

    def _generate(self, prompt: str) -> Optional[str]:
        try:
            return self.client.complete(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=int(os.getenv("GEORESCUE_ORCHESTRATOR_MAX_TOKENS", "300")),
                temperature=float(os.getenv("GEORESCUE_ORCHESTRATOR_TEMPERATURE", "0.1")),
            )
        except Exception:
            return None
