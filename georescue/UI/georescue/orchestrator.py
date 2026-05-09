import os
import streamlit as st


def get_orchestrator_url() -> str:
    try:
        url = st.secrets.get("GIS_API_URL", "")
        if not url:
            url = st.secrets.get("ORCHESTRATOR_URL", "")
        return url
    except Exception:
        return os.getenv("GIS_API_URL", os.getenv("ORCHESTRATOR_URL", ""))
