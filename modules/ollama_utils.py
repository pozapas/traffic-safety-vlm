"""
ollama_utils.py
─────────────────────────────────────────────────────────────────
Utilities for discovering and communicating with a local Ollama
instance.
"""

from __future__ import annotations

import requests
from typing import Optional


def check_ollama_running(host: str = "http://localhost:11434") -> bool:
    """Return True if Ollama is reachable at the given host."""
    try:
        r = requests.get(f"{host}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_ollama_models(host: str = "http://localhost:11434") -> list[str]:
    """
    Return a sorted list of locally available Ollama model names.
    Returns an empty list if Ollama is not running or an error occurs.
    """
    try:
        r = requests.get(f"{host}/api/tags", timeout=5)
        if r.status_code != 200:
            return []
        data = r.json()
        models = data.get("models", [])
        return sorted(m["name"] for m in models if "name" in m)
    except Exception:
        return []


def get_ollama_model_info(model: str, host: str = "http://localhost:11434") -> Optional[dict]:
    """Return detailed model info from Ollama (parameters, family, etc.)."""
    try:
        r = requests.post(f"{host}/api/show", json={"name": model}, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def is_vision_model(model_name: str) -> bool:
    """
    Heuristic check: does the model name suggest vision capability?
    Ollama does not expose this directly; we rely on naming conventions.
    """
    vision_keywords = [
        "vision", "llava", "qwen2.5-vl", "qwen-vl", "gemma4",
        "minicpm-v", "moondream", "bakllava", "pixtral", "llama3.2",
        "internvl", "cogvlm", "phi3-vision", "phi-3-vision",
    ]
    lower = model_name.lower()
    return any(kw in lower for kw in vision_keywords)
