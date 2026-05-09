"""Download Sentinel-2 RGB imagery from Sentinel Hub for a given bounding box."""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image

from .config import (
    COLOMBO_BBOX,
    RAW_DIR,
    SENTINEL_CLIENT_ID,
    SENTINEL_CLIENT_SECRET,
)

EVALSCRIPT_RGB = """
//VERSION=3
function setup() {
    return { input: [{ bands: ["B04", "B03", "B02"] }], output: { bands: 3 } };
}
function evaluatePixel(sample) {
    return [sample.B04 / 4095, sample.B03 / 4095, sample.B02 / 4095];
}
"""


def download_sentinel2_rgb(
    bbox: "list[float] | None" = None,
    date_from: "str | None" = None,
    date_to: "str | None" = None,
    output_path: "Path | None" = None,
    resolution: int = 10,
) -> str:
    from sentinelhub import (
        BBox,
        CRS,
        DataCollection,
        MimeType,
        SentinelHubRequest,
        SHConfig,
        bbox_to_dimensions,
    )

    if not SENTINEL_CLIENT_ID or not SENTINEL_CLIENT_SECRET:
        raise ValueError("SENTINEL_CLIENT_ID and SENTINEL_CLIENT_SECRET must be set in .env")

    config = SHConfig()
    config.sh_client_id = SENTINEL_CLIENT_ID
    config.sh_client_secret = SENTINEL_CLIENT_SECRET

    target_bbox = bbox or COLOMBO_BBOX
    sh_bbox = BBox(bbox=target_bbox, crs=CRS.WGS84)
    size = bbox_to_dimensions(sh_bbox, resolution=resolution)
    # Sentinel Hub hard limit: max 2500 px per side
    size = (min(size[0], 2500), min(size[1], 2500))

    end = datetime.utcnow()
    start = end - timedelta(days=30)
    date_from = date_from or start.strftime("%Y-%m-%d")
    date_to = date_to or end.strftime("%Y-%m-%d")

    request = SentinelHubRequest(
        evalscript=EVALSCRIPT_RGB,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(date_from, date_to),
                mosaicking_order="leastCC",
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=sh_bbox,
        size=size,
        config=config,
    )

    data = request.get_data()[0]
    img_array = (np.clip(data, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(img_array)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = output_path or (RAW_DIR / f"sentinel2_rgb_{timestamp}.png")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_path))
    return str(out_path)
