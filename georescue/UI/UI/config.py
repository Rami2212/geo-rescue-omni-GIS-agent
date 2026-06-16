import os


def _env_float(name: str, default: str) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def _env_int(name: str, default: str) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return int(default)


DEFAULT_CENTER_LAT = _env_float("MAP_CENTER_LAT", "6.9271")
DEFAULT_CENTER_LON = _env_float("MAP_CENTER_LON", "79.8612")
DEFAULT_ZOOM = _env_int("MAP_DEFAULT_ZOOM", "12")
DEFAULT_GRAPH_RADIUS_KM = _env_float("MAP_GRAPH_RADIUS_KM", "12")
DEFAULT_SPEED_KMH = _env_float("DEFAULT_SPEED_KMH", "30")

SAMPLE_DAMAGE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Sample Damage Zone"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [79.8512, 6.9171],
                        [79.8712, 6.9171],
                        [79.8712, 6.9371],
                        [79.8512, 6.9371],
                        [79.8512, 6.9171],
                    ]
                ],
            },
        }
    ],
}

STATUS_TEMPLATE = [
    "Supervisor (HF Llama 3B): planning mission",
    "Image Scout: loading realtime image",
    "Vision Agent (HF Qwen2.5-VL + local LoRA): extracting flooded polygons",
    "Route Agent: computing safe + alternative routes",
    "Supervisor: final operational summary",
]

SAMPLE_ORCHESTRATOR_ROUTE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {
                "name": "Orchestrator Safe Route",
                "distance_km": 4.2,
                "travel_time_min": 12.5,
                "blocked_edges": 3,
            },
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [79.8468, 6.9257],
                    [79.8545, 6.9242],
                    [79.8641, 6.9279],
                    [79.8728, 6.9315],
                    [79.8792, 6.9341],
                ],
            },
        }
    ],
}
