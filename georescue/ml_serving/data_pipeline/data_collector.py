"""Orchestrate collection of (satellite image, GIS label) training sample pairs."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd

from .config import PROCESSED_DIR, RAW_DIR, TRAINING_DATA_DIR
from .flood_overlay import analyze_flood_impact
from .live_flood_feed import fetch_live_flood
from .sentinel_downloader import download_sentinel2_rgb


# ---------------------------------------------------------------------------
# Original single-sample collector (full pipeline including Sentinel download)
# ---------------------------------------------------------------------------

def collect_training_sample(
    lat: float = 6.9271,
    lon: float = 79.8612,
    sample_id: "str | None" = None,
) -> dict:
    """Collect one (satellite image, GIS label) training pair from live data."""
    sample_id = sample_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sample_dir = TRAINING_DATA_DIR / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    polygon_path, weather = fetch_live_flood(lat=lat, lon=lon, save_dir=sample_dir)

    road_net_path = PROCESSED_DIR / "colombo_road_network.geojson"
    if not road_net_path.exists():
        raise FileNotFoundError(
            f"Road network not found at {road_net_path}. "
            "Run data_pipeline.road_network.download_road_network() first."
        )

    try:
        img_path = download_sentinel2_rgb(output_path=sample_dir / "satellite.png")
    except Exception as exc:
        img_path = None
        print(f"[warn] Sentinel-2 download failed ({exc}) — sample {sample_id} will have no image.")

    impact = analyze_flood_impact(
        road_network_path=road_net_path,
        flood_polygon_path=polygon_path,
        save_dir=sample_dir,
    )

    label = {
        "sample_id": sample_id,
        "timestamp_utc": weather["timestamp_utc"],
        "latitude": lat,
        "longitude": lon,
        "severity": weather["severity"],
        "max_precip_mm": weather["max_precip_mm"],
        "radius_km": weather["radius_km"],
        "affected_roads": impact["affected_roads"],
        "total_roads": impact["total_roads"],
        "affected_length_m": impact["affected_length_m"],
        "flood_polygon_path": polygon_path,
        "blocked_roads_path": impact["blocked_roads_path"],
        "satellite_image_path": img_path,
    }
    (sample_dir / "label.json").write_text(json.dumps(label, indent=2))
    return label


def collect_dataset(n_samples: int = 10, interval_seconds: int = 300) -> list:
    """Collect multiple training samples at regular intervals."""
    samples = []
    for i in range(n_samples):
        print(f"[{i + 1}/{n_samples}] Collecting sample...")
        try:
            label = collect_training_sample()
            samples.append(label)
            print(f"  severity={label['severity']}  affected_roads={label['affected_roads']}")
        except Exception as exc:
            print(f"  [error] {exc}")

        if i < n_samples - 1:
            time.sleep(interval_seconds)

    index_path = TRAINING_DATA_DIR / "dataset_index.json"
    index_path.write_text(json.dumps(samples, indent=2))
    print(f"\nDataset index saved to {index_path}")
    return samples


# ---------------------------------------------------------------------------
# Fast in-memory overlay helper (road network pre-loaded, avoids disk reads)
# ---------------------------------------------------------------------------

def _overlay_in_memory(
    roads_gdf: gpd.GeoDataFrame,
    flood_polygon_path: str,
    sample_dir: Path,
) -> dict:
    """Run flood overlay using a pre-loaded road GeoDataFrame — much faster per cycle."""
    flood = gpd.read_file(str(flood_polygon_path))
    if roads_gdf.crs != flood.crs:
        flood = flood.to_crs(roads_gdf.crs)

    flood_union = flood.geometry.union_all()
    intersecting = roads_gdf[roads_gdf.geometry.intersects(flood_union)].copy()

    blocked_path = str(sample_dir / "blocked_roads_flood.geojson")
    if not intersecting.empty:
        out = intersecting.copy()
        if out.crs is None:
            out = out.set_crs("EPSG:4326")
        elif out.crs.to_epsg() != 4326:
            out = out.to_crs("EPSG:4326")
        out.to_file(blocked_path, driver="GeoJSON")

    roads_m = roads_gdf.to_crs("EPSG:3857")
    inter_m = intersecting.to_crs("EPSG:3857") if not intersecting.empty else intersecting

    return {
        "total_roads": len(roads_gdf),
        "affected_roads": len(intersecting),
        "total_length_m": float(roads_m.geometry.length.sum()),
        "affected_length_m": float(inter_m.geometry.length.sum()) if not intersecting.empty else 0.0,
        "blocked_roads_path": blocked_path,
    }


# ---------------------------------------------------------------------------
# Fast live collector — 10-second intervals, 4-hour duration
# ---------------------------------------------------------------------------

def fast_collect_live(
    duration_hours: float = 4.0,
    interval_seconds: int = 10,
    satellite_interval_minutes: int = 30,
    lat: float = 6.9271,
    lon: float = 79.8612,
    append_to_existing: bool = True,
) -> list:
    """
    Collect live GIS training data at fast intervals (default 10 s) for a fixed duration.

    Per-cycle work  (~2 s):
      1. Fetch Open-Meteo precipitation  →  flood polygon + severity
      2. Overlay flood on in-memory road network  →  blocked road metrics
      3. Save label.json

    Every `satellite_interval_minutes` (default 30 min):
      - Download a fresh Sentinel-2 RGB image and link it to all subsequent samples.

    With 10 s intervals and 4 hours:  ~1 440 samples, 8 satellite refreshes.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # --- guard: road network must exist -------------------------------------
    road_net_path = PROCESSED_DIR / "colombo_road_network.geojson"
    if not road_net_path.exists():
        raise FileNotFoundError(
            f"\nRoad network not found at:\n  {road_net_path}\n\n"
            "Fix:  python -c \"from ml_serving.data_pipeline.road_network import "
            "download_road_network; download_road_network()\"\n"
            "OR copy Minindu's file:\n"
            "  geo-rescue-omni-GIS-agent-dev-minindu/georescue-amd-hackathon/"
            "data/processed/colombo_road_network.geojson"
        )

    total_seconds = int(duration_hours * 3600)
    n_samples = total_seconds // interval_seconds
    sat_every_n = max(1, int(satellite_interval_minutes * 60 / interval_seconds))

    print(f"\n{'='*60}")
    print(f"  GeoRescue Fast Live Collector")
    print(f"  Duration  : {duration_hours:.1f} h  ({total_seconds} s)")
    print(f"  Interval  : {interval_seconds} s")
    print(f"  Samples   : {n_samples} planned")
    print(f"  Satellite : refresh every {satellite_interval_minutes} min ({sat_every_n} cycles)")
    print(f"{'='*60}\n")

    # --- pre-load road network into memory ----------------------------------
    print("Loading road network into memory...", flush=True)
    roads_gdf = gpd.read_file(str(road_net_path))
    print(f"  {len(roads_gdf)} road segments loaded.\n")

    # --- initial satellite image download -----------------------------------
    current_sat_path = None
    print("Downloading initial Sentinel-2 image (may take ~30 s)...", flush=True)
    try:
        sat_out = TRAINING_DATA_DIR / "raw" / "satellite_initial.png"
        current_sat_path = download_sentinel2_rgb(output_path=sat_out)
        print(f"  Saved → {current_sat_path}\n")
    except Exception as exc:
        print(f"  [warn] Satellite download skipped: {exc}\n")

    # --- load existing dataset index if appending ---------------------------
    index_path = TRAINING_DATA_DIR / "dataset_index.json"
    samples: list = []
    if append_to_existing and index_path.exists():
        try:
            samples = json.loads(index_path.read_text())
            print(f"Appending to existing index ({len(samples)} samples already collected).\n")
        except Exception:
            samples = []

    # --- main collection loop -----------------------------------------------
    global_start = time.time()
    errors = 0

    for i in range(n_samples):
        cycle_start = time.time()

        # ETA display
        elapsed = cycle_start - global_start
        remaining = max(0.0, total_seconds - elapsed)
        h, rem = divmod(int(remaining), 3600)
        m, s = divmod(rem, 60)
        pct = 100 * i / n_samples
        print(
            f"[{i+1:>5}/{n_samples}]  {pct:5.1f}%  ETA {h:02d}:{m:02d}:{s:02d}  "
            f"errors={errors}",
            end="\r",
            flush=True,
        )

        # --- periodic satellite refresh -------------------------
        if i > 0 and i % sat_every_n == 0:
            print(f"\n[info] Refreshing satellite image (cycle {i})...", flush=True)
            try:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                new_sat = download_sentinel2_rgb(
                    output_path=TRAINING_DATA_DIR / "raw" / f"satellite_{ts}.png"
                )
                current_sat_path = new_sat
                print(f"[info] New satellite image → {new_sat}\n")
            except Exception as exc:
                print(f"[warn] Satellite refresh failed: {exc}\n")

        # --- collect one sample ---------------------------------
        sample_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        sample_dir = TRAINING_DATA_DIR / "samples" / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        try:
            polygon_path, weather = fetch_live_flood(lat=lat, lon=lon, save_dir=sample_dir)
            impact = _overlay_in_memory(roads_gdf, polygon_path, sample_dir)

            label = {
                "sample_id": sample_id,
                "timestamp_utc": weather["timestamp_utc"],
                "latitude": lat,
                "longitude": lon,
                "severity": weather["severity"],
                "max_precip_mm": weather["max_precip_mm"],
                "radius_km": weather["radius_km"],
                "affected_roads": impact["affected_roads"],
                "total_roads": impact["total_roads"],
                "affected_length_m": impact["affected_length_m"],
                "flood_polygon_path": polygon_path,
                "blocked_roads_path": impact["blocked_roads_path"],
                "satellite_image_path": current_sat_path,
            }
            (sample_dir / "label.json").write_text(json.dumps(label, indent=2))
            samples.append(label)

        except Exception as exc:
            errors += 1
            print(f"\n[warn] Sample {sample_id} failed: {exc}")

        # --- flush index every 50 samples -----------------------
        if len(samples) % 50 == 0:
            index_path.write_text(json.dumps(samples, indent=2))

        # --- sleep to hit target interval -----------------------
        cycle_elapsed = time.time() - cycle_start
        sleep_for = max(0.0, interval_seconds - cycle_elapsed)
        if sleep_for > 0:
            time.sleep(sleep_for)

    # --- final flush --------------------------------------------------------
    index_path.write_text(json.dumps(samples, indent=2))
    total_elapsed = time.time() - global_start
    print(f"\n\n{'='*60}")
    print(f"  Collection complete")
    print(f"  Samples collected : {len(samples)}")
    print(f"  Errors            : {errors}")
    print(f"  Total time        : {total_elapsed/3600:.2f} h")
    print(f"  Dataset index     : {index_path}")
    print(f"{'='*60}\n")
    return samples
