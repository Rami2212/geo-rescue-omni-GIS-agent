"""Throughput / stress test — simulate concurrent requests."""

import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


def single_llama_request(base_url="http://localhost:8000/v1"):
    """Send a single request to the Llama-3 endpoint."""
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="EMPTY")
    start = time.time()
    resp = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role": "user", "content": "What are flood safety tips?"}],
        max_tokens=64,
    )
    return time.time() - start


def stress_test_llama(concurrent_users=10, base_url="http://localhost:8000/v1"):
    """Simulate concurrent users hitting the Llama-3 API."""
    print(f"🔥 Stress testing Llama-3 with {concurrent_users} concurrent requests...")

    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(single_llama_request, base_url) for _ in range(concurrent_users)]
        times = [f.result() for f in as_completed(futures)]

    print(f"  Total time: {max(times):.2f}s")
    print(f"  Avg latency: {sum(times)/len(times):.2f}s")
    print(f"  Throughput: {len(times)/max(times):.1f} req/s")


if __name__ == "__main__":
    stress_test_llama(concurrent_users=10)
