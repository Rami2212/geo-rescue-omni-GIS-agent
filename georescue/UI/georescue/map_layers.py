from typing import List, Optional, Tuple

import folium
import networkx as nx
import osmnx as ox
from shapely.geometry import Polygon, mapping
from folium.plugins import Draw, Fullscreen, MiniMap, MousePosition


def load_base_map(center: Tuple[float, float], zoom: int) -> folium.Map:
    base_map = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    folium.TileLayer("CartoDB dark_matter", name="Dark Mode").add_to(base_map)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Satellite",
        attr="Esri",
    ).add_to(base_map)

    Fullscreen().add_to(base_map)
    MiniMap(toggle_display=True).add_to(base_map)
    MousePosition(position="bottomright").add_to(base_map)

    Draw(
        export=False,
        draw_options={
            "polyline": False,
            "rectangle": True,
            "polygon": True,
            "circle": False,
            "circlemarker": False,
            "marker": False,
        },
        edit_options={"edit": True, "remove": True},
    ).add_to(base_map)

    return base_map


def render_damage_layers(
    base_map: folium.Map,
    polygons: List[Polygon],
    show_damage: bool,
) -> Optional[dict]:
    if not polygons or not show_damage:
        return None

    features = []
    for idx, poly in enumerate(polygons, start=1):
        features.append(
            {
                "type": "Feature",
                "properties": {"name": f"Damage Zone {idx}"},
                "geometry": mapping(poly),
            }
        )

    geojson = {"type": "FeatureCollection", "features": features}
    folium.GeoJson(
        geojson,
        name="Damage Zones",
        style_function=lambda _: {
            "fillColor": "#ff5f5f",
            "color": "#ff5f5f",
            "weight": 2,
            "fillOpacity": 0.35,
        },
    ).add_to(base_map)

    return geojson


def render_road_network(base_map: folium.Map, graph: nx.MultiDiGraph) -> None:
    edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True, fill_edge_geometry=True)
    geojson = edges_gdf[["geometry"]].to_json()
    folium.GeoJson(
        geojson,
        name="Road Network",
        style_function=lambda _: {"color": "#6c7a89", "weight": 1, "opacity": 0.6},
    ).add_to(base_map)


def render_geojson_layer(
    base_map: folium.Map,
    geojson: Optional[dict],
    name: str,
    style: dict,
) -> None:
    if not geojson:
        return

    folium.GeoJson(
        geojson,
        name=name,
        style_function=lambda _: style,
    ).add_to(base_map)


def render_flood_layer(base_map: folium.Map, geojson: Optional[dict], show: bool) -> None:
    if not show:
        return
    render_geojson_layer(
        base_map,
        geojson,
        "Flood Zone",
        {
            "fillColor": "#ff6b6b",
            "color": "#ff6b6b",
            "weight": 2,
            "fillOpacity": 0.35,
        },
    )


def render_blocked_roads(base_map: folium.Map, geojson: Optional[dict], show: bool) -> None:
    if not show:
        return
    render_geojson_layer(
        base_map,
        geojson,
        "Blocked Roads",
        {"color": "#ff9f1a", "weight": 3, "opacity": 0.85},
    )
