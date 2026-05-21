"""Local LLM backend for OpenNPC.

Provides a lightweight, GPU-accelerated language model for:
- NPC dialogue generation
- Personality-flavored text
- Strategic planning (VillainPlanner enhancement)

Uses HuggingFace Transformers with small models (≤ 1B params)
that fit in 4GB VRAM alongside Minecraft.

This module is OPTIONAL. The core decision engine works without it.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Default model — Qwen2.5-0.5B-Instruct is fast, smart, and fits in 1GB VRAM
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Fallback if user wants something even smaller
SMALL_MODELS = {
    "qwen-0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "smollm-360m": "HuggingFaceTB/SmolLM2-360M-Instruct",
    "smollm-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
}


@dataclass(slots=True)
class LLMResponse:
    """Structured response from the LLM."""

    text: str
    model: str
    latency_ms: float
    cached: bool = False
    tokens_generated: int = 0


class LLMEngine:
    """Local LLM engine with lazy loading, caching, and thread safety.

    The model is loaded on first use and kept in memory. Inference is
    thread-safe via a lock. Responses are cached with an LRU eviction policy.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or alias from SMALL_MODELS.
    device : str
        "cuda", "cpu", or "auto" (picks best available).
    max_new_tokens : int
        Maximum tokens to generate per request.
    cache_size : int
        Maximum cached responses (LRU eviction).
    temperature : float
        Sampling temperature. Lower = more deterministic.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        max_new_tokens: int = 200,
        cache_size: int = 128,
        temperature: float = 0.7,
    ) -> None:
        self.model_name = SMALL_MODELS.get(model_name, model_name)
        self._device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self._model: Any = None
        self._tokenizer: Any = None
        self._lock = threading.Lock()
        self._cache: OrderedDict[str, LLMResponse] = OrderedDict()
        self._cache_size = cache_size
        self._loaded = False
        self._load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def device(self) -> str:
        if self._device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self._device

    def load(self) -> None:
        """Load the model and tokenizer into memory.

        Called automatically on first inference. Can be called manually
        to pre-warm the model (e.g., at server startup).
        """
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                logger.info(
                    "Loading LLM: %s on %s ...", self.model_name, self.device
                )
                start = time.monotonic()

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                )

                dtype = torch.float16 if self.device == "cuda" else torch.float32

                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map=self.device if self.device == "auto" else None,
                    trust_remote_code=True,
                )

                if self.device != "auto":
                    self._model = self._model.to(self.device)

                self._model.eval()
                elapsed = (time.monotonic() - start) * 1000
                logger.info(
                    "LLM loaded in %.0fms | Device: %s | Params: %.0fM",
                    elapsed,
                    self.device,
                    sum(p.numel() for p in self._model.parameters()) / 1e6,
                )
                self._loaded = True

            except Exception as e:
                self._load_error = str(e)
                logger.error("Failed to load LLM: %s", e)
                raise

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        use_cache: bool = True,
    ) -> LLMResponse:
        """Generate text from the LLM.

        Parameters
        ----------
        prompt : str
            The user/task prompt.
        system_prompt : str
            System-level instructions for the model.
        max_new_tokens : int, optional
            Override default max tokens.
        temperature : float, optional
            Override default temperature.
        use_cache : bool
            Whether to check/store in the response cache.

        Returns
        -------
        LLMResponse
            The generated text with metadata.
        """
        if not self._loaded:
            self.load()

        cache_key = f"{system_prompt}|{prompt}|{max_new_tokens}|{temperature}"
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            self._cache.move_to_end(cache_key)
            return LLMResponse(
                text=cached.text,
                model=cached.model,
                latency_ms=0.0,
                cached=True,
                tokens_generated=cached.tokens_generated,
            )

        import torch

        tokens = max_new_tokens or self.max_new_tokens
        temp = temperature or self.temperature

        # Build chat messages for instruction-tuned models
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        with self._lock:
            start = time.monotonic()

            try:
                # Use chat template if available (most instruct models support this)
                text_input = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                # Fallback for models without chat template
                text_input = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

            inputs = self._tokenizer(
                text_input, return_tensors="pt", truncation=True, max_length=1024,
            ).to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=tokens,
                    temperature=temp,
                    do_sample=temp > 0.01,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

            # Decode only the generated portion (exclude input)
            generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
            result = self._tokenizer.decode(
                generated_ids, skip_special_tokens=True,
            ).strip()

            elapsed = (time.monotonic() - start) * 1000

        response = LLMResponse(
            text=result,
            model=self.model_name,
            latency_ms=round(elapsed, 1),
            cached=False,
            tokens_generated=len(generated_ids),
        )

        if use_cache:
            self._cache[cache_key] = response
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)

        logger.info(
            "LLM generated %d tokens in %.0fms",
            response.tokens_generated,
            response.latency_ms,
        )

        return response

    def status(self) -> dict[str, Any]:
        """Return engine status for debug endpoints."""
        return {
            "loaded": self._loaded,
            "model": self.model_name,
            "device": self.device,
            "cache_entries": len(self._cache),
            "cache_max": self._cache_size,
            "load_error": self._load_error,
        }

    def clear_cache(self) -> None:
        """Clear the response cache."""
        self._cache.clear()
