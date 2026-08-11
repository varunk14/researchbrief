"""Thin, provider-agnostic model client.

Business logic depends only on the small surface `call_json(prompt, tier) -> str`.
The active provider (Gemini by default, Anthropic as a fallback) is resolved here
from configuration, so no stage names a provider. Token usage is accumulated for
the run log.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .config import Config


@dataclass
class TokenUsage:
    """Running total of tokens spent across every call in a run."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def add(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.calls += 1


class LLM:
    """Real model client. Chooses the provider from cfg and returns JSON text.

    call_json returns the model's raw response string; parsing and schema checks
    are the caller's job, so a bad response can be retried with the error fed back.
    """

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.usage = TokenUsage()
        self._client = None  # created lazily on first call

    def _model_for(self, tier: str) -> str:
        return self.cfg.smart_model if tier == "smart" else self.cfg.fast_model

    def call_json(self, prompt: str, tier: str = "fast", temperature: float = 0.0) -> str:
        """Send one prompt and return the raw JSON text the model produced."""
        if self.cfg.provider == "anthropic":
            return self._call_anthropic(prompt, tier, temperature)
        return self._call_gemini(prompt, tier, temperature)

    def _call_gemini(self, prompt: str, tier: str, temperature: float) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            key = os.environ.get("GEMINI_API_KEY")
            if not key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            self._client = genai.Client(api_key=key)

        response = self._client.models.generate_content(
            model=self._model_for(tier),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.usage.add(
                getattr(usage, "prompt_token_count", 0) or 0,
                getattr(usage, "candidates_token_count", 0) or 0,
            )
        return response.text or ""

    def _call_anthropic(self, prompt: str, tier: str, temperature: float) -> str:
        import anthropic

        if self._client is None:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic(api_key=key)

        message = self._client.messages.create(
            model=self._model_for(tier),
            max_tokens=4096,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        usage = getattr(message, "usage", None)
        if usage is not None:
            self.usage.add(
                getattr(usage, "input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0,
            )
        return "".join(block.text for block in message.content if block.type == "text")
