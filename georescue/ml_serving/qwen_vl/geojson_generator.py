"""Convert AI vision output into valid GeoJSON FeatureCollections."""

import json
import geojson


def parse_zones_to_geojson(
    ai_output: str, metadata: dict = None
) -> "tuple[dict, str, str]":
    """Parse AI model output into GeoJSON plus extracted severity and findings.

    Handles both well-formed JSON and graceful fallback when the model returns
    malformed or markdown-wrapped output.

    Args:
        ai_output: Raw text from Qwen-VL (expected to be a JSON object).
        metadata: Optional extra properties to attach to each GeoJSON feature.

    Returns:
        (geojson_feature_collection, severity_str, findings_text)
        On parse failure: (empty FeatureCollection, "unknown", "")
    """
    severity = "unknown"
    findings = ""

    try:
        # Strip markdown code fences the model sometimes wraps output in
        cleaned = ai_output.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()

        parsed = json.loads(cleaned)
        zones = parsed.get("affected_zones", [])
        severity = parsed.get("severity", "unknown")
        findings = parsed.get("findings", "")
    except (json.JSONDecodeError, AttributeError, IndexError):
        return geojson.FeatureCollection([]), severity, findings

    features = []
    for i, zone_coords in enumerate(zones):
        if not zone_coords:
            continue

        # Ensure polygon is closed (GeoJSON requirement)
        if zone_coords[0] != zone_coords[-1]:
            zone_coords.append(zone_coords[0])

        polygon = geojson.Polygon([zone_coords])
        feature = geojson.Feature(
            geometry=polygon,
            properties={
                "zone_id": i + 1,
                "severity": severity,
                "source": "qwen-vl-analysis",
                **(metadata or {}),
            },
        )
        features.append(feature)

    return geojson.FeatureCollection(features), severity, findings
