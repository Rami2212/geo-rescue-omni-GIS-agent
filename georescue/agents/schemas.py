from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


LatLon = Tuple[float, float]


@dataclass
class BackendRunRequest:
    mission: str
    disaster_type: str = "flood"
    start: LatLon = (0.0, 0.0)
    dest: LatLon = (0.0, 0.0)
    uploaded_image_bytes: Optional[bytes] = None
    realtime_image_dir: Optional[str] = None
    map_center: Optional[LatLon] = None
    graph_radius_km: Optional[float] = None


@dataclass
class AgentUpdate:
    agent: str
    status: str
    message: str
    data: Optional[dict] = None

    def as_dict(self) -> dict:
        payload = {
            "agent": self.agent,
            "status": self.status,
            "message": self.message,
        }
        if self.data is not None:
            payload["data"] = self.data
        return payload
