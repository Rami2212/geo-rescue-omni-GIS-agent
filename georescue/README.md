<div align="center">

# 🛰️ GeoRescue — Omni GIS Agent

### AI-Powered Disaster Response Platform

*When floods strike, every second counts. GeoRescue puts satellite vision, live weather intelligence,*  
*and autonomous AI agents in the hands of first responders — in real time.*

---

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![AMD MI300X](https://img.shields.io/badge/AMD-Instinct%20MI300X-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-FF6B35?style=for-the-badge)](https://crewai.com)
[![Ollama](https://img.shields.io/badge/Ollama-Llama%203.2-000000?style=for-the-badge)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![AMD Hackathon](https://img.shields.io/badge/AMD-Developer%20Hackathon-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://lablab.ai)

---

[Overview](#-overview) · [Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Agent Pipeline](#-agent-pipeline) · [API Reference](#-api-reference) · [Team](#-team)

</div>

---

## 🌊 Overview

GeoRescue is a multi-agent disaster response platform built for the **AMD Developer Hackathon**. It combines satellite vision AI, live geospatial analysis, and autonomous agent reasoning to give emergency coordinators an instant, accurate picture of flood impact — and the safest route through it.

**The problem:** During a flood, command centers need three things fast — *where is the water*, *which roads are blocked*, and *how do we get there safely*. Traditional GIS workflows take hours. GeoRescue does it in seconds.

**The solution:** A four-agent CrewAI pipeline — backed by Qwen2-VL running on the AMD Instinct MI300X and Ollama Llama 3.2 — that ingests satellite imagery, pulls live weather data, performs spatial overlay analysis, and delivers a structured emergency response report, all in one click.

---

## ✨ Features

| | Feature | Description |
|---|---------|-------------|
| 🛰️ | **Satellite Damage Detection** | Upload pre/post-disaster imagery → Qwen2-VL-7B (fine-tuned with LoRA on AMD MI300X) extracts flood zones as GeoJSON polygons |
| 🌧️ | **Live Flood Analysis** | Real-time Open-Meteo precipitation data → circular flood polygon sized by rainfall severity (low / moderate / high / extreme) |
| 🚧 | **Road Impact Overlay** | Spatial intersection of flood polygon with OSMnx road network → identifies every blocked segment with name, type, and affected length |
| 🗺️ | **Safe Route Planning** | NetworkX shortest-path on the unblocked road graph → optimal evacuation corridor rendered as a live green overlay on the Folium map |
| 🤖 | **4-Agent AI Crew** | CrewAI agents with Ollama Llama 3.2: Vision Analyst → Data Scout → Spatial Navigator → Reporting Coordinator, all visible in real time |
| 📋 | **Incident Report** | AI-generated structured Markdown report: situation summary, blocked infrastructure, safe corridor, and recommended actions for first responders |
| 🔌 | **Graceful Degradation** | Works at all four combinations of API online/offline and LLM online/offline — never a blank screen |
| 🗺️ | **Interactive Map** | Folium map with satellite/dark/OSM tile layers, draw-your-own hazard zones, click-to-set routing points, and real-time layer toggle |

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                     STREAMLIT COMMAND CENTER                            ║
║                          app.py  :8501                                  ║
║                                                                          ║
║  ┌─────────────────┐  ┌────────────────────┐  ┌──────────────────────┐ ║
║  │  Mission Prompt │  │  Folium Map        │  │  Agent Activity Feed │ ║
║  │  + Image Upload │  │  Flood / Roads /   │  │  + Incident Report   │ ║
║  │  + Route Points │  │  Safe Route layers │  │  + Export GeoJSON    │ ║
║  └────────┬────────┘  └────────────────────┘  └──────────────────────┘ ║
╚═══════════╪══════════════════════════════════════════════════════════════╝
            │  run_pipeline_with_status()   [generator, yields AgentUpdate]
            ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                     CREWAI AGENT PIPELINE                               ║
║                    agents/pipeline.py                                   ║
║              LLM backbone: Ollama  llama3.2  (:11434)                  ║
║                                                                          ║
║  ① Vision Analyst ──────────────────────────────────────────────────── ║
║    tools: POST /analyze-image                                           ║
║    task:  Satellite image → Qwen2-VL → severity + damage GeoJSON       ║
║                                                                          ║
║  ② Data Scout ──────────────────────────────────────────────────────── ║
║    tools: POST /gis/run-cycle  ·  GET /health  ·  GET /gis/status      ║
║    task:  Fetch live weather → trigger flood analysis cycle             ║
║                                                                          ║
║  ③ Spatial Navigator ───────────────────────────────────────────────── ║
║    tools: GET /gis/flood-polygon  ·  /blocked-roads  ·  /safe-route    ║
║    task:  Retrieve spatial layers, summarise blocked roads + route      ║
║                                                                          ║
║  ④ Reporting Coordinator ───────────────────────────────────────────── ║
║    tools: none (pure reasoning)                                         ║
║    task:  Synthesise all inputs → structured Markdown incident report   ║
╚═══════════╪══════════════════════════════════════════════════════════════╝
            │  HTTP REST  (localhost:9000)
            ▼
╔══════════════════════════════════════════════════════════════════════════╗
║                FASTAPI GIS & VISION SERVER                              ║
║                     ml_serving/  :9000                                  ║
║                                                                          ║
║  VISION LAYER                                                            ║
║  ┌──────────────────────────────────────────────────────────────────┐  ║
║  │  Qwen2-VL-7B  +  LoRA adapter (final3/)                         │  ║
║  │  Hardware: AMD Instinct MI300X  ·  Runtime: ROCm + vLLM          │  ║
║  │  Input: satellite/aerial image  →  Output: severity + GeoJSON    │  ║
║  └──────────────────────────────────────────────────────────────────┘  ║
║                                                                          ║
║  GIS LAYER                                                               ║
║  ┌──────────────────────────────────────────────────────────────────┐  ║
║  │  Open-Meteo API  →  rainfall mm/6h  →  flood zone polygon        │  ║
║  │  GeoPandas overlay  →  blocked road segments (name + length_m)   │  ║
║  │  OSMnx + NetworkX  →  shortest safe route (GeoJSON LineString)   │  ║
║  └──────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Project Structure

```
georescue/
│
├── agents/                         # 🤖 AI Orchestration Layer — Member 1
│   ├── __init__.py                 #    Public API: run_pipeline_with_status()
│   ├── tools.py                    #    CrewAI @tool wrappers, retry-safe HTTP client
│   ├── crew.py                     #    Agent + Task definitions, Ollama LLM factory
│   └── pipeline.py                 #    Generator pipeline, AgentUpdate / PipelineResult
│
├── georescue/                      # 🗺️ UI Utility Package — Members 1 + 4
│   ├── __init__.py                 #    Clean re-exports of all public symbols
│   ├── config.py                   #    Pydantic-settings, singleton get_settings()
│   ├── state.py                    #    Streamlit session-state init + typed accessors
│   ├── map_layers.py               #    Folium builders: flood, blocked, route, markers
│   ├── routing.py                  #    OSMnx cached graph, NetworkX safe-route
│   ├── geojson_utils.py            #    GeoJSON export / count helpers
│   └── logging_setup.py            #    Centralised structured logging
│
├── ml_serving/                     # ⚡ GIS + Vision API Server — Member 3
│   ├── api/
│   │   ├── app.py                  #    FastAPI entry, CORS, lifespan (model warmup)
│   │   ├── routes.py               #    POST /analyze-image  ·  GET /health
│   │   ├── gis_routes.py           #    GET|POST /gis/* endpoints
│   │   └── schemas.py              #    Pydantic response models
│   ├── gis_pipeline/
│   │   ├── pipeline.py             #    run_cycle(): weather → polygon → overlay → route
│   │   ├── live_flood_feed.py      #    Open-Meteo API → GeoJSON flood polygon
│   │   ├── flood_overlay.py        #    GeoPandas spatial intersection
│   │   └── routing.py              #    NetworkX safe-route planner
│   ├── qwen_vl/
│   │   ├── model_loader.py         #    Qwen2-VL-7B + LoRA adapter (singleton)
│   │   ├── inference.py            #    Vision inference pipeline
│   │   ├── image_processor.py      #    Resize / normalise input images
│   │   └── geojson_generator.py    #    Raw LLM output → GeoJSON FeatureCollection
│   ├── training/                   #    LoRA fine-tuning scripts (Colab / AMD Cloud)
│   ├── benchmarks/                 #    GPU throughput & latency tests
│   ├── data_pipeline/              #    Sentinel Hub satellite data collector
│   ├── llama_server/               #    vLLM Llama server config for AMD MI300X
│   ├── docker/Dockerfile           #    Container image for AMD ROCm environment
│   ├── data/processed/             #    ⚠️  Runtime GeoJSON + GraphML (gitignored)
│   ├── final3/                     #    ⚠️  LoRA adapter weights (gitignored)
│   └── requirements.txt            #    GPU/ML dependencies (torch, transformers…)
│
├── app.py                          # 🚀 Streamlit application entry point
├── requirements.txt                # 📦 Core dependencies — no GPU required
├── Makefile                        # 🛠️  make install | api | app | all | lint
├── .env.example                    # 📋 Configuration template — commit this
├── .env                            # 🔒 Local secrets — gitignored, never commit
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

<table>
<tr><th>Tool</th><th>Version</th><th>Install</th><th>Required for</th></tr>
<tr><td><b>Python</b></td><td>3.11+</td><td><a href="https://python.org">python.org</a></td><td>Everything</td></tr>
<tr><td><b>Ollama</b></td><td>latest</td><td><a href="https://ollama.com/download">ollama.com</a></td><td>AI agent reasoning</td></tr>
<tr><td><b>AMD ROCm</b> or <b>CUDA</b></td><td>6.x / 12.x</td><td>See GPU setup below</td><td>Qwen-VL inference only</td></tr>
</table>

> **No GPU?** The Streamlit UI and agent pipeline run fine on CPU — only the  
> `/analyze-image` vision endpoint requires a GPU. Everything else degrades gracefully.

---

### Step 1 — Get the code

```bash
git clone <repo-url>
cd georescue
```

---

### Step 2 — Configure environment

```bash
cp .env.example .env
```

The defaults work for local development. Edit `.env` only if your ports differ or you want to change the Ollama model:

```bash
# Key settings (full list in .env.example)
OLLAMA_MODEL=llama3.2          # or llama3.1:8b for better tool-use
GIS_API_URL=http://localhost:9000
OLLAMA_BASE_URL=http://localhost:11434
```

---

### Step 3 — Install dependencies

```bash
# Core: UI + agent framework  (CPU only, ~2 min)
pip install -r requirements.txt

# ML serving: GPU inference  (requires ROCm or CUDA, ~10 min)
cd ml_serving && pip install -r requirements.txt && cd ..
```

Or with Make:

```bash
make install       # core
make install-ml    # ML serving
```

---

### Step 4 — Pull the Ollama model

```bash
ollama pull llama3.2
```

> Alternatives: `ollama pull llama3.1:8b` (stronger tool-use, 5 GB) or  
> `ollama pull mistral:7b` (faster, slightly less accurate).

---

### Step 5 — Start services

Open **two terminals** in the `georescue/` directory:

**Terminal 1 — GIS & Vision API server**
```bash
cd ml_serving
uvicorn api.app:app --host 0.0.0.0 --port 9000 --reload
```
→ Swagger docs available at **http://localhost:9000/docs**

**Terminal 2 — Streamlit UI**
```bash
streamlit run app.py
```
→ Application available at **http://localhost:8501**

Or launch both with a single command:
```bash
make all
```

---

## 🤖 Agent Pipeline

When you click **▶ Run Agent Pipeline**, GeoRescue executes a 4-agent CrewAI crew
in sequence. Each agent's activity appears live in the `st.status()` panel.

```
 User clicks ▶ Run Agent Pipeline
       │
       ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  🎯 SUPERVISOR  checks API health · routes mission to agents        │
 └─────────────────────────┬───────────────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼  (sequential)
 ┌───────────┐       ┌───────────┐       ┌───────────┐
 │ 👁️ VISION  │──────▶│ 📡 DATA   │──────▶│ 🗺️ SPATIAL│
 │ ANALYST   │       │ SCOUT     │       │ NAVIGATOR │
 │           │       │           │       │           │
 │ Satellite │       │ Triggers  │       │ Flood zone│
 │ image →   │       │ live flood│       │ blocked   │
 │ Qwen2-VL  │       │ cycle via │       │ roads &   │
 │ damage    │       │ Open-Meteo│       │ safe route│
 │ zones     │       │ weather   │       │ from API  │
 └───────────┘       └───────────┘       └─────┬─────┘
                                               │
                           ┌───────────────────┘
                           ▼
                     ┌───────────┐
                     │ 📋 REPORT │
                     │ COORD.    │
                     │           │
                     │ Ollama    │
                     │ Llama 3.2 │
                     │ →Markdown │
                     │  report   │
                     └─────┬─────┘
                           │
                           ▼
              Map updated + Report rendered
```

### Agent Details

| Agent | Role | Tools | Output |
|-------|------|-------|--------|
| **Vision Analyst** | Analyse satellite/aerial imagery | `POST /analyze-image` | Severity level + damage GeoJSON zones |
| **Data Scout** | Situational awareness | `POST /gis/run-cycle` · `GET /health` · `GET /gis/status` | Live flood severity, road impact count, route length |
| **Spatial Navigator** | GIS layer retrieval | `GET /gis/flood-polygon` · `/blocked-roads` · `/safe-route` | Flood zone details, blocked road list, route confirmation |
| **Reporting Coordinator** | Incident report synthesis | *(Ollama LLM — no API tools)* | Structured Markdown incident report |

---

## 🔌 Degradation Matrix

GeoRescue **always works**, even when services are unavailable:

| GIS API `:9000` | Ollama `:11434` | Behaviour |
|:---:|:---:|---|
| ✅ Online | ✅ Online | **Full pipeline** — live flood polygon + blocked roads + AI-generated incident report |
| ✅ Online | ❌ Offline | Live GIS layers on map + structured template report |
| ❌ Offline | ✅ Online | Local OSMnx routing from user-drawn hazard zones + AI report |
| ❌ Offline | ❌ Offline | Local OSMnx routing + template report — *always usable* |

---

## 🌐 API Reference

The GIS & Vision server runs at **http://localhost:9000** — interactive docs at `/docs`.

### Vision

| Method | Endpoint | Body | Response |
|--------|----------|------|----------|
| `GET` | `/health` | — | `{ status, llama_status, qwen_status, gpu_available }` |
| `POST` | `/analyze-image` | `file` (image) + `disaster_type` (form) | `{ severity, findings, geojson, inference_time_ms }` |

### GIS Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/gis/run-cycle` | Trigger full flood analysis (weather → polygon → roads → route) |
| `GET` | `/gis/status` | Latest cycle results: severity, affected roads, route length |
| `GET` | `/gis/flood-polygon` | Current flood zone as GeoJSON FeatureCollection |
| `GET` | `/gis/blocked-roads` | Road segments intersecting flood zone (name, type, length_m) |
| `GET` | `/gis/safe-route` | Computed safe route as GeoJSON LineString |

#### Example — trigger a flood cycle

```bash
# Run analysis
curl -X POST http://localhost:9000/gis/run-cycle
# → { "status": "ok", "severity": "high", "affected_roads": 14, "route_length_m": 3842 }

# Get the safe route GeoJSON
curl http://localhost:9000/gis/safe-route | python -m json.tool
```

#### Example — analyse a satellite image

```bash
curl -X POST http://localhost:9000/analyze-image \
     -F "file=@colombo_flood.jpg" \
     -F "disaster_type=flood"
# → { "severity": "high", "findings": "...", "geojson": { "type": "FeatureCollection", ... } }
```

---

## ⚙️ Environment Variables

Copy [`.env.example`](.env.example) to `.env` — all variables have sensible defaults for local development.

| Variable | Default | Description |
|----------|---------|-------------|
| `GIS_API_URL` | `http://localhost:9000` | FastAPI GIS & Vision server |
| `ADAPTER_PATH` | `final3` | LoRA adapter path, relative to `ml_serving/` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama inference server |
| `OLLAMA_MODEL` | `llama3.2` | Model name — `llama3.2` or `llama3.1:8b` |
| `LLM_TEMPERATURE` | `0.1` | Sampling temperature (low = deterministic outputs) |
| `LLM_MAX_TOKENS` | `2048` | Maximum tokens per agent response |
| `MAP_CENTER_LAT` | `6.9271` | Default map centre latitude (Colombo Fort) |
| `MAP_CENTER_LON` | `79.8612` | Default map centre longitude |
| `MAP_GRAPH_RADIUS_KM` | `12` | OSMnx road graph fetch radius |
| `DEFAULT_SPEED_KMH` | `30` | Vehicle speed for travel-time estimates |
| `LOG_LEVEL` | `INFO` | Logging verbosity: `DEBUG` / `INFO` / `WARNING` |

---

## 🛠️ Development

```bash
# Install linter
pip install ruff

# Lint all source files
make lint

# Remove compiled Python files
make clean
```

### Adding a new agent tool

1. Add a function to [`agents/tools.py`](agents/tools.py) with the `@tool` decorator
2. Import it in [`agents/crew.py`](agents/crew.py) and assign it to the relevant agent's `tools` list
3. Add the corresponding API endpoint to [`ml_serving/api/gis_routes.py`](ml_serving/api/gis_routes.py) if needed

### Running only the UI (no GPU)

The Streamlit app and agent framework have **no GPU dependency**. Only `ml_serving/` requires ROCm/CUDA. You can develop the full UI and agent pipeline on any laptop — the pipeline degrades gracefully when the API is unreachable.

---

## 👥 Team

> Built in 4 days for the AMD Developer Hackathon.

| | Member | Role | Responsibilities |
|---|--------|------|-----------------|
| 🎯 | **Imansha** *(Team Lead)* | AI Orchestrator | CrewAI pipeline · Ollama integration · agent prompts · project integration |
| 🗺️ | **Minindu** | GIS & Data Engineer | GeoPandas overlay · OSMnx routing · Open-Meteo flood feed · road network data |
| ⚡ | **Supun** | ML / Hardware | Qwen2-VL fine-tuning · AMD MI300X deployment · ROCm optimisation · FastAPI server |
| 🖥️ | **Ramitha** | UI & Deployment | Streamlit dashboard · Folium maps · Hugging Face Spaces deployment |

---

## 🏆 Hackathon Highlights

- **AMD Instinct MI300X** — Qwen2-VL-7B inference with LoRA fine-tuning; ROCm-accelerated pipeline handles 1024px satellite imagery in < 5 seconds
- **Multi-agent architecture** — Four specialist CrewAI agents backed by Llama 3.2 (Ollama), each with scoped tools and explicit context chaining
- **Real-world data** — Open-Meteo live precipitation, OpenStreetMap road network via OSMnx, Sentinel Hub satellite imagery
- **Production-ready patterns** — Pydantic-settings config, retry-safe HTTP client, typed dataclasses, graceful degradation at every layer

---

## 📄 License

This project was created for the AMD Developer Hackathon. See individual sub-components for their respective licences.

---

<div align="center">

*Built by Team Zynaptrix · AMD Developer Hackathon 2026*

</div>
