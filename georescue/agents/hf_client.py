from __future__ import annotations

import os
from typing import Optional

import requests


DEFAULT_HF_BASE_URL = "https://router.huggingface.co/v1"


def hf_token() -> Optional[str]:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
    )


class HFChatClient:
    """Tiny OpenAI-compatible Hugging Face router client."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        timeout: int = 90,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("GEORESCUE_HF_BASE_URL") or DEFAULT_HF_BASE_URL).rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.1,
        response_format: dict | None = None,
    ) -> str:
        token = hf_token()
        if not token:
            raise RuntimeError(
                "HF_TOKEN is not configured. Set HF_TOKEN with Inference Providers permission."
            )

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
