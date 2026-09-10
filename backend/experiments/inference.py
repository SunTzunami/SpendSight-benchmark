"""
inference.py – Unified inference dispatch for LlamaCpp backend.

All backends return the same (response, elapsed_s, error_or_None) triple.
Fully standalone — no imports from backend/utils/.
"""
from __future__ import annotations

import time
import os
import logging
import jinja2
from typing import Optional

# Monkey-patch Jinja2 Environment to always include loopcontrols 
# This fixes "Encountered unknown tag 'continue'" in Llama.cpp chat templates (e.g. EXAONE-4.0)
# NOTE: This patch applies globally to all Jinja2 usage in the process. It is
# safe for the benchmark script (the only consumer) but could cause subtle side
# effects if inference.py is imported into a larger app that also uses Jinja2.
_JINJA_PATCHED = False
if not _JINJA_PATCHED:
    _original_jinja_init = jinja2.Environment.__init__
    def _patched_jinja_init(self, **kwargs):
        extensions = kwargs.get("extensions", [])
        if "jinja2.ext.loopcontrols" not in extensions:
            extensions = list(extensions) + ["jinja2.ext.loopcontrols"]
        kwargs["extensions"] = extensions
        _original_jinja_init(self, **kwargs)
    jinja2.Environment.__init__ = _patched_jinja_init
    _JINJA_PATCHED = True

logger = logging.getLogger(__name__)

# ── Singleton for LlamaCpp model (lazy-loaded) ────────────────────────────────────
_llamacpp_model = None
_last_usage: dict = {}  # token-level usage from most recent inference


def get_last_usage() -> dict:
    """Return token usage dict from the most recent LLM call.
    Keys: prompt_tokens, completion_tokens, total_tokens."""
    return dict(_last_usage)


def reset_model() -> None:
    """Properly free the LlamaCpp model's C++ memory and reset the singleton.

    This must be called instead of directly manipulating ``_llamacpp_model``
    from outside the module, because the module-level variable is bound by
    reference at import time — external assignments won't propagate back.

    Steps:
    1. Call ``model.close()`` (llama-cpp-python ≥ 0.2.58) to release the C++
       Metal/GPU memory.
    2. ``del`` the model reference so Python's ref-count drops to zero.
    3. Reset the singleton so a fresh model can be loaded on the next call.
    4. ``gc.collect()`` to sweep any residual cyclic object graphs.
    """
    import gc

    global _llamacpp_model
    if _llamacpp_model is not None:
        if _llamacpp_model.model is not None:
            # Prefer .close() for explicit C++ resource release (llama-cpp-python ≥ 0.2.58)
            if hasattr(_llamacpp_model.model, "close"):
                try:
                    _llamacpp_model.model.close()
                except Exception:
                    pass
            del _llamacpp_model.model
            _llamacpp_model.model = None
        _llamacpp_model.current_model_path = None
        # Reset the singleton so LlamaCppModel.__new__ creates a fresh instance
        LlamaCppModel._instance = None
        _llamacpp_model = None
    gc.collect()


class LlamaCppModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaCppModel, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.current_model_path = None
            cls._instance.current_enable_thinking = False
            cls._instance.last_usage = {}
        return cls._instance

    def resolve_path(self, model_identifier: str) -> str:
        """Resolves a model identifier to an absolute path.
        Looks in local directory and LM Studio caches."""
        if os.path.exists(model_identifier):
            return model_identifier
            
        script_dir = os.path.dirname(os.path.abspath(__file__))
        backend_root = os.path.dirname(script_dir)
        models_dir = os.path.join(backend_root, "models")
        
        # Check in experiments/models directly
        potential_path = os.path.join(models_dir, model_identifier)
        if os.path.exists(potential_path):
            return potential_path
            
        # Check via MODEL_REGISTRY in models.py
        try:
            from experiments.models import get_model_info
            info = get_model_info(model_identifier)
            if info and "id" in info:
                reg_path = os.path.join(models_dir, info["id"])
                if os.path.exists(reg_path):
                    return reg_path
        except Exception:
            pass
            
        # Check LM studio caches, just in case
        lm_studio_base = os.path.expanduser("~/.lmstudio/models")
        if os.path.exists(lm_studio_base):
            for publisher in os.listdir(lm_studio_base):
                if publisher.startswith('.'): continue
                pub_path = os.path.join(lm_studio_base, publisher)
                if not os.path.isdir(pub_path): continue
                for model_folder in os.listdir(pub_path):
                    if model_folder.startswith('.'): continue
                    potential_file = os.path.join(pub_path, model_folder, model_identifier)
                    if os.path.exists(potential_file):
                        return potential_file
                        
        return os.path.join(models_dir, model_identifier)  # fallback

    def load_model(self, model_identifier: str, enable_thinking: bool = False) -> None:
        from llama_cpp import Llama
        import gc
            
        model_path = self.resolve_path(model_identifier)
        
        if (self.current_model_path == model_path and 
            self.model is not None and 
            getattr(self, "current_enable_thinking", False) == enable_thinking):
            return

        # If a model is already loaded, free it first to avoid out-of-memory
        if self.model is not None:
            if hasattr(self.model, "close"):
                try:
                    self.model.close()
                except Exception:
                    pass
            del self.model
            self.model = None
            gc.collect()

        logger.info(f"Loading Llama.cpp model from: {model_path} (Identified as: {model_identifier}) with enable_thinking={enable_thinking}")
        try:
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=-1, # Accelerate as much as possible
                n_ctx=8192, # Context window size (raised from 4096 to fit single-agent prompt ~5900 tokens)
                verbose=False,
            )
            
            # Apply custom chat formatter to disable thinking for any model if needed
            if not enable_thinking:
                template = self.model.metadata.get('tokenizer.chat_template')
                if template and "enable_thinking" in template:
                    from llama_cpp.llama_chat_format import Jinja2ChatFormatter
                    eos_token_id = self.model.token_eos()
                    bos_token_id = self.model.token_bos()
                    eos_token = self.model._model.token_get_text(eos_token_id) if eos_token_id != -1 else ""
                    bos_token = self.model._model.token_get_text(bos_token_id) if bos_token_id != -1 else ""
                    
                    class CustomFormatter(Jinja2ChatFormatter):
                        def __call__(self, **kwargs):
                            kwargs["enable_thinking"] = False
                            return super().__call__(**kwargs)
                            
                    formatter = CustomFormatter(
                        template=template,
                        eos_token=eos_token,
                        bos_token=bos_token,
                        stop_token_ids=[eos_token_id]
                    )
                    self.model.chat_handler = formatter.to_chat_handler()

            self.current_model_path = model_path
            self.current_enable_thinking = enable_thinking
            logger.info("Llama.cpp model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Llama.cpp model: {e}")
            raise e

    def chat(
        self,
        model_identifier: str,
        messages: list,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        enable_thinking: bool = False,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        self.load_model(model_identifier, enable_thinking=enable_thinking)
        kwargs = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if top_k is not None:
            kwargs["top_k"] = top_k
        if min_p is not None:
            kwargs["min_p"] = min_p
        if stop is not None:
            kwargs["stop"] = stop
        try:
            response = self.model.create_chat_completion(**kwargs)
            self.last_usage = response.get("usage", {})
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Llama.cpp chat error: {e}")
            raise e


def _get_llamacpp_model():
    """Lazy-load the LlamaCpp model wrapper."""
    global _llamacpp_model
    if _llamacpp_model is None:
        _llamacpp_model = LlamaCppModel()
    return _llamacpp_model


# ── Llama.cpp inference ────────────────────────────────────────────────────────────

def generate_llamacpp(
    model_id: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    enable_thinking: bool = False,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    min_p: Optional[float] = None,
    stop: Optional[list[str]] = None,
) -> tuple[str, float, Optional[str]]:
    """Call the LlamaCpp model via local llamacpp_utils singleton and return (response, elapsed_s, error)."""
    try:
        llamacpp_model = _get_llamacpp_model()

        t0 = time.perf_counter()
        chat_kwargs = {
            "model_identifier": model_id,
            "messages": messages,
            "temperature": temperature,
            "enable_thinking": enable_thinking,
        }
        if top_p is not None:
            chat_kwargs["top_p"] = top_p
        if top_k is not None:
            chat_kwargs["top_k"] = top_k
        if max_tokens is not None:
            chat_kwargs["max_tokens"] = max_tokens
        if min_p is not None:
            chat_kwargs["min_p"] = min_p
        if stop is not None:
            chat_kwargs["stop"] = stop

        response = llamacpp_model.chat(**chat_kwargs)
        elapsed = time.perf_counter() - t0

        global _last_usage
        _last_usage = llamacpp_model.last_usage

        return response.strip(), elapsed, None
    except Exception as exc:
        logger.error(f"LlamaCpp inference error: {exc}")
        return "", 0.0, str(exc)


# ── Unified dispatch ─────────────────────────────────────────────────────────

def generate(
    backend: str,
    model_id: str,
    messages: list[dict[str, str]],
    temperature: float = 0.0,
    enable_thinking: bool = False,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    max_tokens: Optional[int] = None,
    min_p: Optional[float] = None,
    stop: Optional[list[str]] = None,
) -> tuple[str, float, Optional[str]]:
    """
    Unified inference dispatch.

    backend: 'llamacpp'
    Returns: (response_text, elapsed_seconds, error_string_or_None)
    """
    if backend == "llamacpp":
        return generate_llamacpp(model_id, messages, temperature, enable_thinking, top_p, top_k, max_tokens, min_p, stop)
    else:
        return "", 0.0, f"Unknown backend: {backend}"
