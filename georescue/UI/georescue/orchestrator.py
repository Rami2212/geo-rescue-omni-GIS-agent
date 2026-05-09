import os
import streamlit as st


def get_orchestrator_url() -> str:
    try:
        return st.secrets.get("ORCHESTRATOR_URL", "")
    except Exception:
        return os.getenv("ORCHESTRATOR_URL", "")

