"""
GeoRescue CrewAI crew definition.

Defines four specialist agents backed by Ollama (Llama 3.2) and assembles them
into a sequential crew that processes a disaster response mission end-to-end.

Agent roster:
  • Data Scout          — situational awareness via live weather + GIS status
  • Spatial Navigator   — flood polygon, blocked roads, safe-route analysis
  • Reporting Coordinator — synthesises all inputs into an actionable report

Note: Vision analysis (Qwen-VL image upload) is handled directly by the pipeline
rather than through CrewAI because image bytes cannot be passed as string tool args.
"""

import logging
import os

from crewai import LLM, Agent, Crew, Process, Task

from agents.tools import (
    check_api_health,
    get_blocked_roads,
    get_flood_polygon,
    get_gis_status,
    get_safe_route,
    run_gis_cycle,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM factory — Ollama Llama 3.2 by default, fully env-configurable
# ---------------------------------------------------------------------------

def _build_llm() -> LLM:
    model = os.getenv("OLLAMA_MODEL", "llama3.2")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2048"))

    logger.info("Building LLM: ollama/%s @ %s", model, base_url)
    return LLM(
        model=f"ollama/{model}",
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# Agent builders
# ---------------------------------------------------------------------------

def _data_scout(llm: LLM) -> Agent:
    return Agent(
        role="Disaster Intelligence Data Scout",
        goal=(
            "Verify service availability and obtain current real-time flood "
            "conditions for the Colombo, Sri Lanka area. Trigger a fresh analysis "
            "cycle and summarise the situation for the rest of the team."
        ),
        backstory=(
            "You are a field intelligence specialist with deep expertise in "
            "real-time disaster monitoring systems. You know how to query weather "
            "and GIS APIs, interpret severity levels, and give the team an accurate "
            "picture of the situation within minutes. You are methodical: you always "
            "check service health first, then trigger fresh data collection."
        ),
        tools=[check_api_health, run_gis_cycle, get_gis_status],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=4,
        cache=True,
    )


def _spatial_navigator(llm: LLM) -> Agent:
    return Agent(
        role="GIS Spatial Navigation Specialist",
        goal=(
            "Analyse the flood extent, identify every blocked road segment, "
            "and confirm the optimal safe evacuation route for emergency vehicles."
        ),
        backstory=(
            "You are a GIS analyst specialising in emergency routing and disaster "
            "spatial analysis. You work with real-time geospatial data to identify "
            "flood zones, classify blocked infrastructure, and compute alternative "
            "safe corridors. You are precise: you always report road names, "
            "highway classifications, and distances in measurable units."
        ),
        tools=[get_flood_polygon, get_blocked_roads, get_safe_route],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=4,
        cache=True,
    )


def _reporting_coordinator(llm: LLM) -> Agent:
    return Agent(
        role="Emergency Response Reporting Coordinator",
        goal=(
            "Synthesise all field data and GIS analysis into a clear, structured, "
            "actionable emergency response report for first-responder command."
        ),
        backstory=(
            "You are a senior emergency management coordinator with 15 years of "
            "experience translating complex GIS and weather data into operational "
            "reports. You know what field commanders need: severity, blocked "
            "corridors, recommended routes, and immediate actions — nothing else. "
            "Your reports are always structured, concise, and unambiguous."
        ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
    )


# ---------------------------------------------------------------------------
# Task builders
# ---------------------------------------------------------------------------

def _task_data_collection(agent: Agent, user_prompt: str) -> Task:
    return Task(
        description=(
            f"MISSION REQUEST: {user_prompt}\n\n"
            "Execute the following steps in order:\n"
            "1. Call 'Check GeoRescue API Health' to confirm services are online.\n"
            "2. Call 'Trigger Live Flood Analysis Cycle' to fetch current weather "
            "   and generate updated flood data.\n"
            "3. Call 'Get Current Flood Situation Status' to retrieve the results.\n"
            "4. Compile a structured status summary including: service health, "
            "   flood severity level, number of affected roads, and route length."
        ),
        expected_output=(
            "A structured JSON-like summary containing:\n"
            "- service_health: online/degraded/offline\n"
            "- flood_severity: low/moderate/high/extreme\n"
            "- affected_roads: integer count\n"
            "- route_length_km: float\n"
            "- analysis_timestamp: ISO datetime string"
        ),
        agent=agent,
    )


def _task_spatial_analysis(agent: Agent) -> Task:
    return Task(
        description=(
            "Using the flood data from the Data Scout, perform spatial analysis:\n"
            "1. Call 'Get Flood Zone Polygon Details' — note severity and timestamp.\n"
            "2. Call 'Get Blocked Roads from Flooding' — list all blocked roads "
            "   with their type and affected length.\n"
            "3. Call 'Get Safe Evacuation Route' — confirm route availability and "
            "   length.\n"
            "4. Produce a spatial analysis summary: flood extent, key blocked roads "
            "   by highway type, safe route length and key streets."
        ),
        expected_output=(
            "A structured spatial summary containing:\n"
            "- flood_severity and zone details\n"
            "- blocked_roads list (name, type, length_m) — top 10 most critical\n"
            "- safe_route_length_km and key_streets list\n"
            "- overall passability assessment"
        ),
        agent=agent,
    )


def _task_report(agent: Agent, user_prompt: str, vision_findings: str) -> Task:
    vision_section = (
        f"\n\nSATELLITE IMAGE ANALYSIS:\n{vision_findings}"
        if vision_findings
        else "\n\nSATELLITE IMAGE: Not provided."
    )

    return Task(
        description=(
            f"MISSION: {user_prompt}{vision_section}\n\n"
            "Using all data gathered by the Data Scout and Spatial Navigator, "
            "produce the final emergency response report. Structure it exactly as:\n\n"
            "## GEORESCUE EMERGENCY RESPONSE REPORT\n"
            "### 1. SITUATION SUMMARY\n"
            "### 2. AFFECTED INFRASTRUCTURE\n"
            "### 3. SAFE CORRIDOR\n"
            "### 4. RECOMMENDED ACTIONS\n"
            "### 5. DATA CONFIDENCE & TIMESTAMP\n\n"
            "Keep each section to 3-5 bullet points. Be specific — include road "
            "names, distances in km, severity labels. No filler text."
        ),
        expected_output=(
            "A formatted Markdown emergency response report with the five sections "
            "above, including specific road names, distances, severity level, "
            "and 3-5 concrete recommended actions for first responders."
        ),
        agent=agent,
    )


# ---------------------------------------------------------------------------
# Public crew builder
# ---------------------------------------------------------------------------

def build_crew(user_prompt: str, vision_findings: str = "") -> Crew:
    """
    Construct a sequential CrewAI crew for one disaster response mission.

    Args:
        user_prompt:     Natural language mission description from the operator.
        vision_findings: Optional findings from Qwen-VL satellite image analysis.

    Returns:
        A configured Crew ready to be kicked off.
    """
    llm = _build_llm()

    scout = _data_scout(llm)
    navigator = _spatial_navigator(llm)
    coordinator = _reporting_coordinator(llm)

    t_data = _task_data_collection(scout, user_prompt)
    t_spatial = _task_spatial_analysis(navigator)
    t_report = _task_report(coordinator, user_prompt, vision_findings)

    # Explicit context chain: report sees both prior task outputs
    t_spatial.context = [t_data]
    t_report.context = [t_data, t_spatial]

    return Crew(
        agents=[scout, navigator, coordinator],
        tasks=[t_data, t_spatial, t_report],
        process=Process.sequential,
        verbose=True,
        memory=False,
    )


def run_crew(user_prompt: str, vision_findings: str = "") -> str:
    """
    Build and run the crew synchronously. Returns the final report string.
    Raises RuntimeError if the crew fails.
    """
    try:
        crew = build_crew(user_prompt, vision_findings)
        result = crew.kickoff()
        return str(result)
    except Exception as exc:
        logger.exception("CrewAI execution failed")
        raise RuntimeError(f"Agent crew failed: {exc}") from exc
