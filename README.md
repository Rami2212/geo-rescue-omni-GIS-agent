<div align="center">

# 🛰️ GeoRescue — Omni GIS Agent

### *When floods strike, every second counts.*

GeoRescue is an AI-powered disaster response platform that puts satellite vision,  
live flood intelligence, and autonomous agent reasoning in the hands of first responders — **in real time**.

---

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![AMD MI300X](https://img.shields.io/badge/AMD-Instinct%20MI300X-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://www.amd.com/en/products/accelerators/instinct/mi300.html)
[![ROCm](https://img.shields.io/badge/ROCm-6.x-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://rocm.docs.amd.com)
[![CrewAI](https://img.shields.io/badge/CrewAI-Multi--Agent-FF6B35?style=for-the-badge)](https://crewai.com)
[![Ollama](https://img.shields.io/badge/Ollama-Llama%203.2-000000?style=for-the-badge)](https://ollama.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![AMD Hackathon](https://img.shields.io/badge/AMD-Developer%20Hackathon%202025-ED1C24?style=for-the-badge&logo=amd&logoColor=white)](https://lablab.ai)

</div>

---

## 🌊 The Problem

During a flood emergency, command centres need three answers **immediately**:

1. **Where is the water?** — Flood extent and severity
2. **Which roads are blocked?** — Infrastructure impact
3. **How do we get there safely?** — Optimal evacuation route

Traditional GIS workflows take hours. Lives are lost in minutes.

---

## 💡 The Solution

GeoRescue is a four-agent CrewAI system that answers all three questions in one click:

- **Satellite Vision** — Qwen2-VL-7B (fine-tuned with LoRA on the **AMD Instinct MI300X**) detects flood zones and damage from aerial imagery in seconds
- **Live Flood Intelligence** — Real-time Open-Meteo precipitation → dynamic flood polygon sized by rainfall severity
- **Road Impact Overlay** — GeoPandas spatial intersection identifies every blocked road segment by name, type, and affected length
- **Safe Route Planning** — NetworkX shortest-path on the unblocked OSMnx road graph delivers the optimal evacuation corridor
- **AI Incident Report** — Ollama Llama 3.2 synthesises all findings into a structured Markdown report for first-responder command

---

## 🏗️ System Architecture

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        STREAMLIT COMMAND CENTER  :8501                      ║
║  Mission prompt · Satellite upload · Interactive Folium map · Agent panel   ║
╚═════════════════════════════════╤════════════════════════════════════════════╝
                                  │  run_pipeline_with_status()
                                  ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                       CREWAI AGENT PIPELINE                                 ║
║                    Backbone LLM: Ollama  llama3.2                           ║
║                                                                              ║
║   ① 👁️  Vision Analyst     POST /analyze-image   → damage GeoJSON           ║
║   ② 📡  Data Scout         POST /gis/run-cycle   → flood severity + stats   ║
║   ③ 🗺️  Spatial Navigator  GET  /gis/flood-polygon                           ║
║                            GET  /gis/blocked-roads                           ║
║                            GET  /gis/safe-route                              ║
║   ④ 📋  Reporting Coord.   Ollama reasoning      → Markdown incident report  ║
╚═════════════════════════════════╤════════════════════════════════════════════╝
                                  │  HTTP REST  localhost:9000
                                  ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                    FASTAPI GIS & VISION SERVER  :9000                       ║
║                                                                              ║
║  Vision  │  Qwen2-VL-7B + LoRA  │  AMD Instinct MI300X · ROCm · vLLM       ║
║  GIS     │  Open-Meteo → flood polygon → GeoPandas road overlay             ║
║          │  OSMnx + NetworkX → safe route (GeoJSON LineString)              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📁 Repository Structure

```
geo-rescue-omni-GIS-agent/
│
└── georescue/                      ← integrated project (run everything from here)
    │
    ├── agents/                     # 🤖 AI Orchestration — Imansha (Member 1)
    │   ├── tools.py                #    CrewAI @tool wrappers, retry-safe HTTP client
    │   ├── crew.py                 #    4 agents + tasks, Ollama LLM factory
    │   └── pipeline.py             #    Generator pipeline, graceful degradation
    │
    ├── georescue/                  # 🗺️  UI Package — Imansha + Ramitha (Members 1, 4)
    │   ├── config.py               #    Pydantic-settings (reads .env)
    │   ├── map_layers.py           #    Folium: flood zone, blocked roads, route, markers
    │   ├── routing.py              #    OSMnx cached graph + NetworkX safe-route
    │   └── state.py                #    Streamlit session-state management
    │
    ├── ml_serving/                 # ⚡  GIS + Vision API — Supun (Member 3)
    │   ├── api/                    #    FastAPI routes + Pydantic schemas
    │   ├── gis_pipeline/           #    Flood cycle: weather → polygon → overlay → route
    │   ├── qwen_vl/                #    Qwen2-VL-7B inference + GeoJSON parser
    │   ├── training/               #    LoRA fine-tuning scripts (AMD MI300X)
    │   ├── benchmarks/             #    GPU throughput + latency tests
    │   ├── final3/                 #    LoRA adapter config (weights gitignored)
    │   └── requirements.txt        #    GPU/ML dependencies
    │
    ├── app.py                      # 🚀  Streamlit entry point
    ├── requirements.txt            # 📦  Core deps — no GPU required
    ├── Makefile                    # 🛠️   make install | api | app | all
    ├── .env.example                # 📋  Configuration template
    └── README.md                   # 📖  Full project documentation
```

> **Full documentation** — setup guide, agent details, API reference, and environment  
> variables — is in **[georescue/README.md](georescue/README.md)**.

---

## ⚡ Quick Start

```bash
# 1. Install Ollama and pull the model
#    https://ollama.com/download
ollama pull llama3.2

# 2. Enter the project
cd georescue

# 3. Configure
cp .env.example .env

# 4. Install core dependencies  (no GPU needed)
pip install -r requirements.txt

# 5. Start the GIS + Vision API  (Terminal 1)
cd ml_serving && uvicorn api.app:app --host 0.0.0.0 --port 9000

# 6. Start the Streamlit UI  (Terminal 2)
streamlit run app.py
```

Open **http://localhost:8501**

> The app works offline too — it falls back to local OSMnx routing and a template  
> report when the API server or Ollama is unavailable.

---

## 🤖 How the Agents Work

| # | Agent | What it does |
|---|-------|-------------|
| ① | **Vision Analyst** | Sends the uploaded satellite image to Qwen2-VL-7B; returns severity level and damage zone GeoJSON |
| ② | **Data Scout** | Triggers the live flood cycle — fetches Open-Meteo precipitation, generates a flood polygon, identifies blocked roads |
| ③ | **Spatial Navigator** | Retrieves the flood polygon, blocked road list, and computed safe route from the API |
| ④ | **Reporting Coordinator** | Asks Ollama Llama 3.2 to synthesise all prior findings into a structured Markdown incident report |

Each agent's progress appears live in the UI's `st.status()` panel — you watch the crew think.

---

## 🔌 Offline Resilience

GeoRescue never shows a blank screen:

| GIS API | Ollama | Result |
|:---:|:---:|---|
| ✅ | ✅ | Full pipeline — live flood map + AI incident report |
| ✅ | ❌ | Live GIS layers + structured template report |
| ❌ | ✅ | Local OSMnx routing + AI-generated report |
| ❌ | ❌ | Local OSMnx routing + template report |

---

## 👥 Team Zynaptrix

| | Member | Role |
|---|--------|------|
| 🎯 | **Imansha** *(Lead)* | AI Orchestrator — CrewAI pipeline, Ollama integration, project integration |
| 🗺️ | **Minindu** | GIS & Data Engineer — OSMnx routing, GeoPandas overlay, Open-Meteo flood feed |
| ⚡ | **Supun** | ML / Hardware — Qwen2-VL fine-tuning, AMD MI300X deployment, ROCm, FastAPI |
| 🖥️ | **Ramitha** | UI & Deployment — Streamlit dashboard, Folium maps, Hugging Face Spaces |

---

## 🏆 AMD Hackathon Highlights

- **AMD Instinct MI300X** accelerates Qwen2-VL-7B inference with a custom LoRA adapter — satellite images analysed in < 5 seconds
- **ROCm-native** training pipeline for LoRA fine-tuning on disaster imagery directly on AMD hardware
- **Multi-agent CrewAI** architecture with Ollama Llama 3.2 — transparent, step-by-step reasoning visible in the UI
- **End-to-end real-world data** — Open-Meteo live weather, OpenStreetMap road network, Sentinel Hub satellite imagery

---

<div align="center">

*Built by Team Zynaptrix · AMD Developer Hackathon 2026*

</div>
