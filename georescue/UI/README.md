# GeoRescue Frontend (Streamlit)

## Overview
This Streamlit UI is the Member 4 deliverable from the hackathon plan. It provides a command console, a live status feed for agent progress, and a Folium-powered operational map for damage zones and safe routes.

## Key Features
- Mission request input (text and optional imagery upload placeholder)
- Status feed that mirrors the multi-agent pipeline phases
- Folium map layers for damage polygons and safe routes
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
The UI currently uses mock GeoJSON outputs. When the orchestrator is ready, wire the API calls in `app.py` and replace the mock data with live responses.

## Deployment Note
Hugging Face Spaces works out-of-the-box with `app.py` as the entry file. Add `ORCHESTRATOR_URL` as a Space secret if the backend is remote.


