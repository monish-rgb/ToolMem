"""Native Google AI Studio (Gemini) REST provider.

Implements the exact pattern:
  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
  Header: x-goog-api-key: <key>
  Body:   {"system_instruction": ..., "contents": [...], "tools": [{"functionDeclarations": [...]}]}

Uses only the standard library (urllib), so no new dependency. Function
calling maps 1:1 onto the OpenAI tool loop used by the harnesses:
model functionCall parts -> execute against MCP -> functionResponse parts.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any

GEMINI_HOST = "https://generativelanguage.googleapis.com"

# Gemini function-declaration schemas accept this subset; strip the rest.
_SCHEMA_KEYS = {"type", "description", "properties", "required", "enum", "items"}


def _sanitize_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}
    clean: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "properties" and isinstance(value, dict):
            clean[key] = {name: _sanitize_schema(sub) for name, sub in value.items()}
        elif key == "items":
            clean[key] = _sanitize_schema(value)
        else:
            clean[key] = value
    clean.setdefault("type", "object")
    return clean


def openai_tools_to_gemini(openai_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    declarations = []
    for tool in openai_tools:
        fn = tool.get("function", tool)
        declarations.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": _sanitize_schema(fn.get("parameters", {})),
        })
    return [{"functionDeclarations": declarations}] if declarations else []


class GeminiRestClient:
    def __init__(self, api_key: str = "", model: str = "", host: str = GEMINI_HOST) -> None:
        self.api_key = (api_key or os.environ.get("GEMINI_API_KEY")
                        or os.environ.get("LLM_API_KEY", "")).strip().strip('"').strip("'")
        self.model = (model or os.environ.get("LLM_MODEL")
                      or os.environ.get("GEMINI_MODEL", "")).strip().strip('"').strip("'")
        self.host = (host or os.environ.get("GEMINI_HOST", GEMINI_HOST)).rstrip("/")
        if not self.api_key:
            raise RuntimeError("set $env:GEMINI_API_KEY (or $env:LLM_API_KEY) for the Gemini provider")
        if not self.model:
            raise RuntimeError("set $env:LLM_MODEL (e.g. gemini-2.5-flash) for the Gemini provider")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.host}/v1beta/models/{self.model}:generateContent"
        body = json.dumps(payload).encode()
        last_error = ""
        max_retries = max(1, int(os.environ.get("GEMINI_MAX_RETRIES", "10")))
        base_delay = max(0.1, float(os.environ.get("GEMINI_RETRY_INITIAL_SECONDS", "2")))
        max_delay = max(base_delay, float(os.environ.get("GEMINI_RETRY_MAX_SECONDS", "60")))
        retryable = (429, 500, 502, 503, 504)
        for attempt in range(max_retries):
            request = urllib.request.Request(
                url, data=body,
                headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
                method="POST")
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    return json.loads(response.read().decode())
            except urllib.error.HTTPError as exc:
                last_error = exc.read().decode()[:500]
                if exc.code not in retryable or attempt == max_retries - 1:
                    raise RuntimeError(f"Gemini API HTTP {exc.code}: {last_error}") from exc
                retry_after = exc.headers.get("Retry-After")
                delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, base_delay)
                if retry_after:
                    try:
                        delay = max(delay, min(max_delay, float(retry_after)))
                    except ValueError:
                        pass
                time.sleep(delay)
            except urllib.error.URLError as exc:
                last_error = str(exc)[:500]
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Gemini API transport error: {last_error}") from exc
                delay = min(max_delay, base_delay * (2 ** attempt)) + random.uniform(0, base_delay)
                time.sleep(delay)
        raise RuntimeError(f"Gemini API unavailable after retries: {last_error}")

    def generate(self, system: str, contents: list[dict[str, Any]],
                 tools: list[dict[str, Any]] | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": contents,
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        gemini_tools = openai_tools_to_gemini(tools or [])
        if gemini_tools:
            payload["tools"] = gemini_tools
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
        data = self._post(payload)
        candidates = data.get("candidates", [])
        parts = (candidates[0].get("content", {}).get("parts", []) if candidates else [])
        text = "".join(part.get("text", "") for part in parts if "text" in part)
        calls = [{"name": part["functionCall"]["name"],
                  "args": json.dumps(part["functionCall"].get("args", {}))}
                 for part in parts if "functionCall" in part]
        usage = data.get("usageMetadata", {}) or {}
        # Normalize per the shared token contract (Phase 1): preserve the raw
        # provider object, never coerce missing fields to zero in `usage`.
        from .token_usage import normalize_gemini_usage
        normalized = normalize_gemini_usage(usage, str(data.get("responseId", "")))
        # Return raw parts so callers can echo them back verbatim. Gemini 3+
        # requires thoughtSignature round-tripping on functionCall parts.
        return {"text": text, "tool_calls": calls, "raw_parts": parts,
                "prompt_tokens": normalized.input_tokens or 0,
                "completion_tokens": normalized.output_tokens or 0,
                "total_tokens": normalized.total_tokens,
                "usage": normalized.to_dict(),
                "usage_source": normalized.usage_source,
                "has_authoritative_input": normalized.has_authoritative_input,
                "raw_usage": usage}


def provider_name() -> str:
    """'gemini' for native REST, else 'openai_compatible'."""
    explicit = (os.environ.get("LLM_PROVIDER", "") or "").strip().lower()
    if explicit in ("gemini", "openai_compatible", "openai"):
        return "gemini" if explicit == "gemini" else "openai_compatible"
    base = (os.environ.get("LLM_BASE_URL", "") or "").lower()
    if "generativelanguage" in base and "/openai" not in base:
        return "gemini"
    return "openai_compatible"
