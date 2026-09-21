"""Explicit LLM seam for paper-style memory construction (Phase 3).

All LLM-backed construction (seed-task proposal, trace reflection,
capability aggregation) goes through an injected ``LLMCall``. Tests and the
offline demo use ``FakeLLMCall``; live use wires :func:`llm_call_from_env`,
which is the *only* function here that touches credentials — and it only
runs when explicitly called, so ordinary ``pytest`` never triggers paid
calls. Credentials are passed to the provider request alone and never enter
memory entries.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .token_usage import TokenUsage, as_token_usage


@dataclass(slots=True)
class LLMResponse:
    """One completion plus its normalized token usage for cost accounting."""

    text: str
    usage: TokenUsage = field(default_factory=TokenUsage.missing)
    raw: dict[str, Any] = field(default_factory=dict)


class LLMCall(Protocol):
    """Injectable text completion: ``(system, user) -> LLMResponse``."""

    def __call__(self, system: str, user: str) -> LLMResponse: ...


class GeminiLLMCall:
    """Native Gemini REST completion (standard library only)."""

    def __init__(self, model: str = "", temperature: float = 0.0,
                 max_tokens: int = 2048) -> None:
        from .gemini_rest import GeminiRestClient
        self._client = GeminiRestClient(model=model)
        self.model = self._client.model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, system: str, user: str) -> LLMResponse:
        # Blocking stdlib call by design: safe from sync and async contexts.
        data = self._client.generate(
            system, [{"role": "user", "parts": [{"text": user}]}],
            None, self.temperature, self.max_tokens)
        usage = as_token_usage(data.get("raw_usage") or data.get("usage"), "gemini")
        return LLMResponse(text=data.get("text", ""), usage=usage, raw=data)


class OpenAICompatLLMCall:
    """OpenAI-compatible chat completion (requires the ``nim`` extra)."""

    def __init__(self, model: str = "", temperature: float = 0.0,
                 max_tokens: int = 2048) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("pip install openai>=1.0 for the OpenAI-compatible LLM seam") from exc
        base_url = (os.environ.get("LLM_BASE_URL") or "").strip().strip('"').strip("'")
        api_key = (os.environ.get("LLM_API_KEY") or "").strip().strip('"').strip("'")
        if not api_key:
            raise RuntimeError("set $env:LLM_API_KEY before using the OpenAI-compatible LLM seam")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self.model = (model or os.environ.get("LLM_MODEL", "")).strip().strip('"').strip("'")
        if not self.model:
            raise RuntimeError("set $env:LLM_MODEL before using the OpenAI-compatible LLM seam")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, system: str, user: str) -> LLMResponse:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=self.temperature, max_tokens=self.max_tokens)
        raw: Any = None
        try:
            raw = response.usage.model_dump()
        except Exception:
            raw = getattr(response, "usage", None)
        usage = as_token_usage(raw, "openai_compatible")
        text = (response.choices[0].message.content or "") if response.choices else ""
        return LLMResponse(text=text, usage=usage,
                           raw={"model": self.model, "id": getattr(response, "id", "")})


def llm_call_from_env(model: str = "", temperature: float = 0.0,
                      max_tokens: int = 2048) -> LLMCall:
    """Explicit live-LLM factory. Raises when no credential is configured."""
    from .gemini_rest import provider_name
    if provider_name() == "gemini":
        return GeminiLLMCall(model=model, temperature=temperature, max_tokens=max_tokens)
    return OpenAICompatLLMCall(model=model, temperature=temperature, max_tokens=max_tokens)


class FakeLLMCall:
    """Deterministic canned completions for tests and the offline demo.

    ``responses`` maps a substring of the user prompt to the response text;
    the first matching key wins, otherwise ``default`` is returned. Every
    call is recorded for assertions.
    """

    def __init__(self, responses: dict[str, str] | None = None,
                 default: str = "{}") -> None:
        self.responses = dict(responses or {})
        self.default = default
        self.calls: list[dict[str, str]] = []

    def __call__(self, system: str, user: str) -> LLMResponse:
        self.calls.append({"system": system, "user": user})
        for needle, text in self.responses.items():
            if needle in user:
                return LLMResponse(text=text)
        return LLMResponse(text=self.default)


LLMCallFactory = Callable[..., LLMCall]
