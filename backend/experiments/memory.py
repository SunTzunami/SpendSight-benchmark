"""
memory.py – Model memory management for Apple Silicon (M1 Pro, 8 GB).

Frees model memory between runs to prevent OOM on memory-constrained hardware.
Fully standalone — no imports from backend/utils/.
"""
from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def free_model_memory() -> None:
    """
    Best-effort memory reclamation between models.
    Calls reset_model() from inference.py which properly closes the
    underlying llama-cpp-python C++ model and resets the singleton,
    then runs gc.collect() to reclaim GPU/Metal memory.
    """
    try:
        from experiments.inference import reset_model
        reset_model()
    except Exception:
        try:
            from inference import reset_model
            reset_model()
        except Exception:
            pass

    gc.collect()

    logger.info("Model memory freed.")
