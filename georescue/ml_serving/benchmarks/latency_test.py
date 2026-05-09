"""Latency benchmarks for Llama-3 and Qwen-VL endpoints."""

import time
import requests


def test_llama_latency(base_url="http://localhost:8000/v1", n=5):
    """Measure average response time for Llama-3 chat completions."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    times = []

    for i in range(n):
        start = time.time()
        client.chat.completions.create(
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": "Describe flood evacuation steps briefly."}],
            max_tokens=128,
        )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    avg = sum(times) / len(times)
    print(f"\n📊 Llama-3 avg latency: {avg:.2f}s over {n} runs\n")


def test_vision_latency(api_url="http://localhost:9000", image_path="sample.jpg", n=3):
    """Measure average response time for vision analysis."""
    times = []

    for i in range(n):
        start = time.time()
        with open(image_path, "rb") as f:
            resp = requests.post(
                f"{api_url}/analyze-image",
                files={"file": ("image.jpg", f, "image/jpeg")},
                data={"disaster_type": "flood"},
            )
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s (status: {resp.status_code})")

    avg = sum(times) / len(times)
    print(f"\n📊 Vision avg latency: {avg:.2f}s over {n} runs\n")


if __name__ == "__main__":
    print("=== Llama-3 Latency Test ===")
    test_llama_latency()
    print("=== Vision Latency Test ===")
    test_vision_latency()
