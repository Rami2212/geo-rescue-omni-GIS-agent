"""Configuration for the Llama-3 vLLM server."""

LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA_PORT = 8000
LLAMA_BASE_URL = f"http://localhost:{LLAMA_PORT}/v1"
MAX_TOKENS = 2048
TEMPERATURE = 0.7
