# experiments/models.py
"""
Model registry for SpendSight benchmark experiments.

Add new models by extending MODEL_REGISTRY. The pipeline is model-agnostic —
any model added here is automatically available to all experiment scripts.
"""
from __future__ import annotations
from typing import Any


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # ── LlamaCpp models (GGUF Local) ────────────────────────────────────

    # EXAONE
    "exaone-4.0-1.2b-q8": {
        "id": "EXAONE-4.0-1.2B-Q8_0.gguf",
        "params": "1.2B",
        "quant": "Q8_0",
        "family": "EXAONE-4.0",
        "arch": "Dense transformer",
        "backend": "llamacpp",
        "supports_thinking": False,
        "ram_gb": 1.5,
    },

    # Gemma
    "gemma-3-1b-it-q8": {
        "id": "gemma-3-1b-it-Q8_0.gguf",
        "params": "1B",
        "quant": "Q8_0",
        "family": "Gemma-3",
        "arch": "Dense transformer",
        "backend": "llamacpp",
        "supports_thinking": False,
        "ram_gb": 1.1,
    },

    # LFM2.5
    "lfm2.5-1.2b-instruct-q8": {
        "id": "LFM2.5-1.2B-Instruct-Q8_0.gguf",
        "params": "1.2B",
        "quant": "Q8_0",
        "family": "LFM2.5",
        "arch": "Dense transformer",
        "backend": "llamacpp",
        "supports_thinking": False,
        "ram_gb": 1.3,
    },

    # MiniCPM
    "minicpm5-1b-q8": {
        "id": "minicpm5-1b-Q8_0.gguf",
        "params": "1B",
        "quant": "Q8_0",
        "family": "MiniCPM",
        "arch": "Dense transformer",
        "backend": "llamacpp",
        "supports_thinking": False,
        "ram_gb": 1.1,
    },

    # Qwen3.5
    "qwen3.5-0.8b-q8": {
        "id": "Qwen3.5-0.8B-Q8_0.gguf",
        "params": "0.8B",
        "quant": "Q8_0",
        "family": "Qwen3.5",
        "arch": "Dense transformer",
        "backend": "llamacpp",
        "supports_thinking": False,
        "ram_gb": 0.9,
    },

    "qwen3.5-2b-q8": {
        "id": "Qwen3.5-2B-Q8_0.gguf",
        "params": "2B",
        "quant": "Q8_0",
        "family": "Qwen3.5",
        "arch": "Dense transformer",
        "backend": "llamacpp",
        "supports_thinking": False,
        "ram_gb": 2.2,
    },

    # G9v3
    "g9v3-3b-q4": {
        "id": "ai9stars_G9v3-3B-Q4_K_M.gguf",
        "params": "3B",
        "quant": "Q4_K_M",
        "family": "G9v3",
        "arch": "Dense transformer",
        "backend": "llamacpp",
        "supports_thinking": False,
        "ram_gb": 2.0,
    },


    # ── API models ──────────────────────────────────────────────────────
}


# ── Helper functions ───────────────────────────────────────────────────

def get_models_by_backend(backend: str) -> list[dict[str, Any]]:
    """Return all models for a given backend."""
    return [
        {**v, "key": k}
        for k, v in MODEL_REGISTRY.items()
        if v["backend"] == backend
    ]


def get_llamacpp_models() -> list[str]:
    """Return model IDs for all LlamaCpp models."""
    return [
        v["id"]
        for v in MODEL_REGISTRY.values()
        if v["backend"] == "llamacpp"
    ]


def get_model_info(model_id: str) -> dict[str, Any] | None:
    """Look up model info by either key or HF ID."""
    if model_id in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_id]

    for v in MODEL_REGISTRY.values():
        if v["id"] == model_id:
            return v

    return None


def get_model_short_name(model_id: str) -> str:
    """Return a short display name from a full model ID."""
    return model_id.split("/")[-1]