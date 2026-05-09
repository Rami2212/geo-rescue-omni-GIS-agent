"""Test script for the Llama-3 vLLM API endpoint."""

from openai import OpenAI

# --- Configuration ---
LLAMA_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
LLAMA_BASE_URL = "http://localhost:8000/v1"

def test_basic_completion():
    """Test a basic chat completion against the vLLM server."""
    client = OpenAI(base_url=LLAMA_BASE_URL, api_key="EMPTY")

    response = client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a disaster response AI assistant for GeoRescue."
            },
            {
                "role": "user",
                "content": "Analyze flood risk for a low-lying coastal area with recent heavy rainfall."
            }
        ],
        max_tokens=512
    )
    print("=== Llama-3 Response ===")
    print(response.choices[0].message.content)
    print(f"\nTokens used: {response.usage.total_tokens}")
    return response


if __name__ == "__main__":
    test_basic_completion()
