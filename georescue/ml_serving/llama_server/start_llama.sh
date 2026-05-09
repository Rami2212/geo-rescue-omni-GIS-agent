#!/bin/bash
# GeoRescue — Start Llama-3 via vLLM with OpenAI-compatible API
# Run this on the AMD MI300X GPU instance

export HF_TOKEN="${HF_TOKEN:-EMPTY}"

python3 -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3-8B-Instruct \
  --dtype float16 \
  --port 8000 \
  --max-model-len 4096
