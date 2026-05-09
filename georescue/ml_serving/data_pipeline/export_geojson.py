from pathlib import Path
import geopandas as gpd


def save_geojson(gdf: gpd.GeoDataFrame, path: "str | Path") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    gdf.to_file(str(path), driver="GeoJSON")


def load_geojson(path: "str | Path") -> gpd.GeoDataFrame:
    return gpd.read_file(str(path))


def save_geojson_batch(items: "dict[str, gpd.GeoDataFrame]") -> None:
    for path, gdf in items.items():
        save_geojson(gdf, path)
