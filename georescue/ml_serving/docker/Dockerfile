# GeoRescue ML Serving — Dockerfile for AMD MI300X (ROCm)
FROM rocm/vllm:latest

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports: 8000 (Llama vLLM), 9000 (FastAPI Vision API)
EXPOSE 8000 9000

# Start the FastAPI vision service (Llama is run separately via start_llama.sh)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "9000"]
