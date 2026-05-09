# GeoRescue Frontend (Streamlit)

## Overview
This Streamlit UI is the Member 4 deliverable from the hackathon plan. It provides a command console, a live status feed for agent progress, and a Folium-powered operational map for damage zones and safe routes.

## Key Features
- Mission request input (text and optional imagery upload)
- Live orchestrator integration for flood polygons, blocked roads, and safe routes
- AI image analysis via `/analyze-image` (when orchestrator is configured)
- Folium map layers for damage polygons, flood zones, blocked roads, and safe routes
- Sidebar controls for map visibility and configuration

## Run Locally
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m streamlit run app.py
```

## Project Files
- `app.py`: Streamlit entry point
- `.streamlit/config.toml`: Dark theme styling
- `docs/step-by-step-guide.md`: Full setup and usage guide

## Backend Integration
- Set `ORCHESTRATOR_URL` as an environment variable or Streamlit secret.
- The UI can pull live data from:
  - `POST /gis/run-cycle`
  - `GET /gis/flood-polygon`
  - `GET /gis/blocked-roads`
  - `GET /gis/safe-route`
  - `POST /analyze-image`
- If the orchestrator is not configured, the UI falls back to local routing and sample data.

## Deployment Note
Hugging Face Spaces works out-of-the-box with `app.py` as the entry file. Add `ORCHESTRATOR_URL` as a Space secret if the backend is remote.
