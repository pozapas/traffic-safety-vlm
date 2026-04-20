"""
vlm_dispatcher.py
─────────────────────────────────────────────────────────────────
Three VLM provider adapters with a unified interface:
  • GeminiAdapter     — google-genai SDK (gemini-flash-latest, etc.)
  • OpenRouterAdapter — OpenAI-compatible REST + SSE streaming
  • OllamaAdapter     — Local Ollama /api/chat endpoint

All adapters accept a list of FrameSample objects and a prompt
pair (system + user) and return a raw string (the VLM response).
─────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Generator, Optional

import requests

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Base adapter
# ──────────────────────────────────────────────

class BaseAdapter(ABC):

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def analyze_frames(
        self,
        frames,          # list[FrameSample]
        system_prompt: str,
        user_prompt: str,
        stream_callback=None,   # callable(chunk: str) for live streaming to UI
    ) -> str:
        """
        Send frames + prompts to the VLM.
        Returns the full response string.
        If stream_callback is provided, call it with each text chunk.
        """
        ...

    def _frames_to_b64_list(self, frames) -> list[str]:
        return [f.base64_png for f in frames]


# ──────────────────────────────────────────────
# Gemini Adapter
# ──────────────────────────────────────────────

class GeminiAdapter(BaseAdapter):
    """
    Uses the google-genai SDK.
    Supports multi-image input and streaming.
    """

    DEFAULT_MODEL = "gemini-flash-latest"

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        super().__init__(model)
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                raise ImportError("google-genai not installed. Run: pip install google-genai")
        return self._client

    def analyze_frames(
        self,
        frames,
        system_prompt: str,
        user_prompt: str,
        stream_callback=None,
    ) -> str:
        from google.genai import types

        client = self._get_client()

        # Build content parts: images + user text
        parts = []
        for frame in frames:
            parts.append(
                types.Part.from_bytes(
                    mime_type="image/png",
                    data=base64.b64decode(frame.base64_png),
                )
            )
        parts.append(types.Part.from_text(text=user_prompt))

        contents = [
            types.Content(role="user", parts=parts)
        ]

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            thinking_config=types.ThinkingConfig(
                thinking_budget=5000,
            ),
            temperature=0.1,    # low temp for structured JSON output
            max_output_tokens=8192,
        )

        full_response = ""
        try:
            for chunk in client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            ):
                if hasattr(chunk, "text") and chunk.text:
                    full_response += chunk.text
                    if stream_callback:
                        stream_callback(chunk.text)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

        return full_response


# ──────────────────────────────────────────────
# OpenRouter Adapter
# ──────────────────────────────────────────────

class OpenRouterAdapter(BaseAdapter):
    """
    OpenAI-compatible REST API with SSE streaming.
    Supports any vision-capable model on OpenRouter.
    """

    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "google/gemini-flash-1.5"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        site_url: str = "http://localhost:8501",
        app_name: str = "VLM Traffic Safety Analyzer",
    ):
        super().__init__(model)
        self.api_key = api_key
        self.site_url = site_url
        self.app_name = app_name

    def analyze_frames(
        self,
        frames,
        system_prompt: str,
        user_prompt: str,
        stream_callback=None,
    ) -> str:
        # Build multimodal message content
        content: list[dict] = [{"type": "text", "text": user_prompt}]
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{frame.base64_png}",
                    "detail": "high",
                },
            })

        payload = {
            "model": self.model,
            "stream": True,
            "temperature": 0.1,
            "max_tokens": 8192,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url,
            "X-Title": self.app_name,
        }

        full_response = ""
        try:
            with requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"]
                            text = delta.get("content", "")
                            if text:
                                full_response += text
                                if stream_callback:
                                    stream_callback(text)
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
        except Exception as e:
            logger.error(f"OpenRouter API error: {e}")
            raise

        return full_response


# ──────────────────────────────────────────────
# Ollama Adapter
# ──────────────────────────────────────────────

class OllamaAdapter(BaseAdapter):
    """
    Local Ollama instance via /api/chat.
    Supports any multimodal model available locally.
    """

    DEFAULT_MODEL = "llava:latest"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = "http://localhost:11434",
    ):
        super().__init__(model)
        self.host = host.rstrip("/")

    def analyze_frames(
        self,
        frames,
        system_prompt: str,
        user_prompt: str,
        stream_callback=None,
    ) -> str:
        images_b64 = [f.base64_png for f in frames]

        payload = {
            "model": self.model,
            "stream": True,
            "options": {"temperature": 0.1, "num_predict": 8192},
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": images_b64,
                },
            ],
        }

        full_response = ""
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json=payload,
                stream=True,
                timeout=300,   # local models can be slow
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        text = chunk.get("message", {}).get("content", "")
                        if text:
                            full_response += text
                            if stream_callback:
                                stream_callback(text)
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            raise

        return full_response


# ──────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────

def create_adapter(
    provider: str,
    model: str,
    api_key: str = "",
    ollama_host: str = "http://localhost:11434",
) -> BaseAdapter:
    """
    Factory function to create the appropriate adapter.

    Args:
        provider: one of "gemini", "openrouter", "ollama"
        model:    model name
        api_key:  API key (Gemini or OpenRouter)
        ollama_host: Ollama server URL
    """
    provider = provider.lower()
    if provider == "gemini":
        if not api_key:
            raise ValueError("Gemini API key is required.")
        return GeminiAdapter(api_key=api_key, model=model)
    elif provider == "openrouter":
        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        return OpenRouterAdapter(api_key=api_key, model=model)
    elif provider == "ollama":
        return OllamaAdapter(model=model, host=ollama_host)
    else:
        raise ValueError(f"Unknown provider: {provider}. Choose gemini, openrouter, or ollama.")
