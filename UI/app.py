import os
import streamlit as st
import folium
from streamlit_folium import st_folium


SAMPLE_DAMAGE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Damage Zone"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-122.438, 37.774],
                        [-122.425, 37.774],
                        [-122.425, 37.784],
                        [-122.438, 37.784],
                        [-122.438, 37.774],
                    ]
                ],
            },
        }
    ],
}

SAMPLE_ROUTE_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Safe Route"},
            "geometry": {
                "type": "LineString",
                "coordinates": [
                    [-122.449, 37.768],
                    [-122.444, 37.772],
                    [-122.438, 37.776],
                    [-122.432, 37.781],
                    [-122.425, 37.786],
                ],
            },
        }
    ],
}

STATUS_TEMPLATE = [
    "Supervisor: parsing intent",
    "Data Agent: fetching imagery",
    "Vision Agent: extracting damage polygon",
    "Spatial Agent: computing safe route",
    "Reporting Agent: preparing GeoJSON",
]


st.set_page_config(page_title="GeoRescue UI", page_icon="🛰️", layout="wide")

if "damage_geojson" not in st.session_state:
    st.session_state.damage_geojson = None
if "route_geojson" not in st.session_state:
    st.session_state.route_geojson = None
if "status_log" not in st.session_state:
    st.session_state.status_log = []

st.title("GeoRescue Command Center")
st.caption("Streamlit UI for the GeoRescue multi-agent GIS workflow")

with st.sidebar:
    st.header("Mission Controls")
    default_orchestrator = st.secrets.get(
        "ORCHESTRATOR_URL", os.getenv("ORCHESTRATOR_URL", "")
    )
    orchestrator_url = st.text_input(
        "Orchestrator API URL (optional)",
        value=default_orchestrator,
        placeholder="http://localhost:8000",
    )
    show_damage = st.checkbox("Show damage zone", value=True)
    show_route = st.checkbox("Show safe route", value=True)
    zoom = st.slider("Map zoom", min_value=8, max_value=16, value=12)
    center_lat = st.number_input("Center latitude", value=37.774, format="%.3f")
    center_lon = st.number_input("Center longitude", value=-122.437, format="%.3f")
    st.markdown("---")
    st.subheader("Status Feed")
    if st.session_state.status_log:
        for item in st.session_state.status_log:
            st.write(f"- {item}")
    else:
        st.write("No agent activity yet.")

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.subheader("Mission Request")
    user_prompt = st.text_area(
        "Describe the incident and the routing goal",
        value="Assess flood damage near downtown and produce a safe route",
        height=140,
    )
    st.file_uploader(
        "Optional: upload a satellite image",
        type=["png", "jpg", "jpeg", "tif"],
        accept_multiple_files=False,
    )
    run_col, mock_col = st.columns(2)
    run_clicked = run_col.button("Run Agents")
    mock_clicked = mock_col.button("Load Mock Data")

    if run_clicked:
        if user_prompt.strip():
            st.session_state.status_log = STATUS_TEMPLATE
            st.session_state.damage_geojson = SAMPLE_DAMAGE_GEOJSON
            st.session_state.route_geojson = SAMPLE_ROUTE_GEOJSON
            st.success("Agent pipeline started. Displaying mock outputs.")
        else:
            st.warning("Add a mission request before running the agents.")

    if mock_clicked:
        st.session_state.status_log = ["Loaded sample outputs"]
        st.session_state.damage_geojson = SAMPLE_DAMAGE_GEOJSON
        st.session_state.route_geojson = SAMPLE_ROUTE_GEOJSON
        st.info("Mock data loaded.")

    with st.expander("Raw GeoJSON Output"):
        st.write(
            {
                "damage_zone": st.session_state.damage_geojson,
                "safe_route": st.session_state.route_geojson,
            }
        )

    st.caption(
        "Backend orchestration is stubbed for now. Connect the API when ready."
    )

with right_col:
    st.subheader("Operational Map")
    map_center = [center_lat, center_lon]
    base_map = folium.Map(
        location=map_center,
        zoom_start=zoom,
        tiles="cartodbpositron",
        control_scale=True,
    )

    if show_damage and st.session_state.damage_geojson:
        folium.GeoJson(
            st.session_state.damage_geojson,
            name="Damage Zone",
            style_function=lambda _: {
                "fillColor": "#ff5f5f",
                "color": "#ff5f5f",
                "weight": 2,
                "fillOpacity": 0.35,
            },
        ).add_to(base_map)

    if show_route and st.session_state.route_geojson:
        folium.GeoJson(
            st.session_state.route_geojson,
            name="Safe Route",
            style_function=lambda _: {
                "color": "#3ddc84",
                "weight": 4,
            },
        ).add_to(base_map)

    folium.LayerControl().add_to(base_map)
    st_folium(base_map, use_container_width=True, height=560)

    if orchestrator_url:
        st.caption(f"Planned API target: {orchestrator_url}")

