"""Shared normalized token-usage model (input-token reduction plan, Phase 1).

One ``TokenUsage`` structure is shared by all provider adapters. Every model
request keeps the provider's raw usage object and normalizes, where
available::

    input_tokens / cached_input_tokens / uncached_input_tokens
    output_tokens / reasoning_tokens / total_tokens
    usage_source (provider | tokenizer_estimate | missing)
    provider_request_id
    has_authoritative_input

Never silently convert a missing usage field to zero: absent provider fields
stay ``None`` and mark the request incomplete so it is excluded from the
primary analysis while retained as a diagnostic artifact.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


def estimate_tokens(text: str) -> int:
    """Deterministic size estimate (characters//4, no model needed)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


@dataclass(slots=True)
class TokenUsage:
    """Normalized per-request usage. Missing fields stay ``None``, never 0."""

    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    uncached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    usage_source: str = "missing"  # provider | tokenizer_estimate | missing
    provider_request_id: str = ""
    has_authoritative_input: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def missing(cls, request_id: str = "", raw: dict | None = None) -> "TokenUsage":
        return cls(
            usage_source="missing",
            provider_request_id=request_id,
            has_authoritative_input=False,
            raw=dict(raw or {}),
        )

    @property
    def is_complete(self) -> bool:
        return self.has_authoritative_input and self.input_tokens is not None


def _int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def normalize_gemini_usage(
    usage: dict[str, Any] | None, request_id: str = ""
) -> TokenUsage:
    """Normalize a Gemini ``usageMetadata`` object.

    Known fields: ``promptTokenCount``, ``cachedContentTokenCount``,
    ``candidatesTokenCount``, ``thoughtsTokenCount``, ``totalTokenCount``.
    ``candidatesTokenCount`` already includes thought tokens, so reasoning is
    reported separately without double counting.
    """
    raw = dict(usage or {})
    if not raw:
        return TokenUsage.missing(request_id, raw)
    prompt = _int(raw.get("promptTokenCount"))
    if prompt is None:
        return TokenUsage.missing(request_id, raw)
    cached = _int(raw.get("cachedContentTokenCount"))
    uncached: int | None = None
    if cached is not None:
        uncached = max(0, prompt - cached)
    candidates = _int(raw.get("candidatesTokenCount"))
    thoughts = _int(raw.get("thoughtsTokenCount"))
    total = _int(raw.get("totalTokenCount"))
    if total is None and (prompt is not None or candidates is not None):
        total = (prompt or 0) + (candidates or 0)
    return TokenUsage(
        input_tokens=prompt,
        cached_input_tokens=cached,
        uncached_input_tokens=uncached,
        output_tokens=candidates,
        reasoning_tokens=thoughts,
        total_tokens=total,
        usage_source="provider",
        provider_request_id=request_id,
        has_authoritative_input=True,
        raw=raw,
    )


def normalize_openai_usage(
    usage: Any | None, request_id: str = ""
) -> TokenUsage:
    """Normalize an OpenAI-compatible ``usage`` object or dict.

    ``completion_tokens`` already includes reasoning tokens, so they are
    reported separately without being added again. Cached prompt tokens come
    from ``prompt_tokens_details.cached_tokens`` when present.
    """
    if usage is None:
        return TokenUsage.missing(request_id, {})
    if not isinstance(usage, dict):
        try:
            raw: dict[str, Any] = usage.model_dump()  # pydantic v2
        except Exception:
            try:
                raw = dict(usage)
            except Exception:
                return TokenUsage.missing(request_id, {})
    else:
        raw = dict(usage)
    if not raw:
        return TokenUsage.missing(request_id, raw)
    prompt = _int(raw.get("prompt_tokens"))
    if prompt is None:
        return TokenUsage.missing(request_id, raw)
    completion = _int(raw.get("completion_tokens"))
    total = _int(raw.get("total_tokens"))
    cached: int | None = None
    details = raw.get("prompt_tokens_details") or {}
    if isinstance(details, dict):
        cached = _int(details.get("cached_tokens"))
    uncached = max(0, prompt - cached) if cached is not None else None
    reasoning: int | None = None
    comp_details = raw.get("completion_tokens_details") or {}
    if isinstance(comp_details, dict):
        reasoning = _int(comp_details.get("reasoning_tokens"))
    if total is None and (prompt is not None or completion is not None):
        total = (prompt or 0) + (completion or 0)
    return TokenUsage(
        input_tokens=prompt,
        cached_input_tokens=cached,
        uncached_input_tokens=uncached,
        output_tokens=completion,
        reasoning_tokens=reasoning,
        total_tokens=total,
        usage_source="provider",
        provider_request_id=request_id,
        has_authoritative_input=True,
        raw=raw,
    )


def normalize_usage(
    provider: str, payload: Any | None, request_id: str = ""
) -> TokenUsage:
    """Dispatch to the provider-specific normalizer."""
    name = (provider or "").strip().lower()
    if name == "gemini":
        if payload is None:
            return TokenUsage.missing(request_id, {})
        return normalize_gemini_usage(
            payload if isinstance(payload, dict) else {}, request_id
        )
    return normalize_openai_usage(payload, request_id)


def as_token_usage(payload: Any | None, provider: str = "") -> TokenUsage:
    """Accept an already-normalized usage dict, else normalize provider payload.

    Guards against double normalization: adapters such as
    ``GeminiRestClient.generate`` return both ``usage`` (normalized) and
    ``raw_usage`` (raw provider metadata). Passing the normalized form back
    through ``normalize_usage`` marks it missing; this helper detects the
    normalized shape (``usage_source`` + ``has_authoritative_input`` keys)
    and returns it directly.
    """
    if isinstance(payload, dict) and "usage_source" in payload and "has_authoritative_input" in payload:
        try:
            return TokenUsage(
                input_tokens=_int(payload.get("input_tokens")),
                cached_input_tokens=_int(payload.get("cached_input_tokens")),
                uncached_input_tokens=_int(payload.get("uncached_input_tokens")),
                output_tokens=_int(payload.get("output_tokens")),
                reasoning_tokens=_int(payload.get("reasoning_tokens")),
                total_tokens=_int(payload.get("total_tokens")),
                usage_source=str(payload.get("usage_source") or "missing"),
                provider_request_id=str(payload.get("provider_request_id") or ""),
                has_authoritative_input=bool(payload.get("has_authoritative_input")),
                raw=dict(payload.get("raw") or {}),
            )
        except Exception:
            pass
    return normalize_usage(provider, payload)


def tokenizer_estimate_usage(    prompt_text: str, completion_text: str = "", request_id: str = ""
) -> TokenUsage:
    """Fallback estimate for anomaly detection only, never the main result."""
    prompt_est = estimate_tokens(prompt_text)
    completion_est = estimate_tokens(completion_text) if completion_text else 0
    return TokenUsage(
        input_tokens=prompt_est,
        cached_input_tokens=None,
        uncached_input_tokens=None,
        output_tokens=completion_est or None,
        reasoning_tokens=None,
        total_tokens=prompt_est + completion_est,
        usage_source="tokenizer_estimate",
        provider_request_id=request_id,
        has_authoritative_input=False,
        raw={"prompt_chars": len(prompt_text or "")},
    )


@dataclass(slots=True)
class RequestRecord:
    """One immutable model-request ledger row."""

    request_id: str
    experiment_id: str
    task_id: str
    arm: str
    attempt: int
    timestamp: str
    model: str
    provider: str
    ok: bool = True
    is_retry: bool = False
    error: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage.missing)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["usage"] = self.usage.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RequestRecord":
        data = dict(payload)
        usage_payload = data.get("usage") or {}
        data["usage"] = TokenUsage(**{
            key: usage_payload.get(key)
            for key in (
                "input_tokens", "cached_input_tokens", "uncached_input_tokens",
                "output_tokens", "reasoning_tokens", "total_tokens",
                "usage_source", "provider_request_id",
                "has_authoritative_input", "raw",
            )
        })
        return cls(**{k: data.get(k, "") for k in (
            "request_id", "experiment_id", "task_id", "arm", "attempt",
            "timestamp", "model", "provider", "ok", "is_retry", "error",
        )} | {"usage": data["usage"]})


def new_request_id() -> str:
    return f"req_{uuid.uuid4().hex[:16]}"


class RequestLedger:
    """Ordered per-request ledger for one attempt (or a whole experiment)."""

    def __init__(self) -> None:
        self.rows: list[RequestRecord] = []

    def append(self, record: RequestRecord) -> RequestRecord:
        self.rows.append(record)
        return record

    def log(
        self,
        *,
        experiment_id: str,
        task_id: str,
        arm: str,
        attempt: int,
        model: str,
        provider: str,
        usage: TokenUsage,
        ok: bool = True,
        is_retry: bool = False,
        error: str = "",
        request_id: str = "",
        timestamp: str = "",
    ) -> RequestRecord:
        record = RequestRecord(
            request_id=request_id or usage.provider_request_id or new_request_id(),
            experiment_id=experiment_id,
            task_id=task_id,
            arm=arm,
            attempt=attempt,
            timestamp=timestamp or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            model=model,
            provider=provider,
            ok=ok,
            is_retry=is_retry,
            error=error[:1000],
            usage=usage,
        )
        if not record.usage.provider_request_id:
            record.usage.provider_request_id = record.request_id
        return self.append(record)

    def attempt_rows(
        self, task_id: str, arm: str, attempt: int
    ) -> list[RequestRecord]:
        return [
            row for row in self.rows
            if row.task_id == task_id and row.arm == arm and row.attempt == attempt
        ]

    def attempt_totals(self, task_id: str, arm: str, attempt: int) -> dict[str, Any]:
        """Sum authoritative provider fields only; missing stays missing."""
        rows = self.attempt_rows(task_id, arm, attempt)
        authoritative = [row for row in rows if row.usage.is_complete]
        incomplete = len(rows) - len(authoritative)

        def _sum(field: str) -> int | None:
            values = [getattr(row.usage, field) for row in authoritative]
            values = [value for value in values if value is not None]
            if not values:
                return None
            return sum(values)

        totals = {
            "model_requests": len(rows),
            "authoritative_requests": len(authoritative),
            "incomplete_requests": incomplete,
            "has_complete_input_usage": bool(rows) and incomplete == 0,
            "input_tokens": _sum("input_tokens"),
            "cached_input_tokens": _sum("cached_input_tokens"),
            "uncached_input_tokens": _sum("uncached_input_tokens"),
            "output_tokens": _sum("output_tokens"),
            "reasoning_tokens": _sum("reasoning_tokens"),
            "total_tokens": _sum("total_tokens"),
        }
        return totals

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(row.to_dict(), sort_keys=True) for row in self.rows)

    @classmethod
    def from_jsonl(cls, text: str) -> "RequestLedger":
        ledger = cls()
        for line in (text or "").splitlines():
            line = line.strip()
            if line:
                ledger.append(RequestRecord.from_dict(json.loads(line)))
        return ledger


def check_resume_compatible(prior: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Compare locked provenance fields; non-empty means refuse resume."""
    keys = (
        "model", "provider", "temperature", "reasoning_effort",
        "max_tokens", "max_turns", "system_prompt_hash",
        "tool_schema_hash", "snapshot_hash", "code_rev", "experiment_id",
    )
    mismatched = [
        key for key in keys
        if key in prior or key in current
        if prior.get(key) != current.get(key)
    ]
    return mismatched


def prompt_anatomy(
    *,
    system: str = "",
    tool_schemas: Any = None,
    guidance: str = "",
    history: str = "",
    task_text: str = "",
) -> dict[str, Any]:
    """Summarize prompt composition with deterministic size estimates."""
    try:
        schemas_text = json.dumps(tool_schemas or [], sort_keys=True, default=str)
    except (TypeError, ValueError):
        schemas_text = str(tool_schemas or "")
    sections = {
        "system_and_agent_instructions": system or "",
        "provider_tool_schemas": schemas_text,
        "toolatlas_guidance": guidance or "",
        "conversation_and_tool_result_history": history or "",
        "current_task_text": task_text or "",
    }
    sizes = {name: estimate_tokens(text) for name, text in sections.items()}
    sizes["total_estimate"] = sum(sizes.values())
    return {"sections": sections, "estimated_tokens": sizes}


def cross_check_usage(
    usage: TokenUsage, *, prompt_text: str = "", completion_text: str = ""
) -> dict[str, Any]:
    """Compare provider totals against a tokenizer estimate for anomalies."""
    estimate = tokenizer_estimate_usage(prompt_text, completion_text)
    if not usage.is_complete:
        return {"comparable": False, "reason": "no authoritative provider usage"}
    ratio: float | None = None
    if usage.input_tokens and estimate.input_tokens:
        ratio = round(usage.input_tokens / max(1, estimate.input_tokens), 3)
    return {
        "comparable": True,
        "provider_input": usage.input_tokens,
        "estimate_input": estimate.input_tokens,
        "ratio": ratio,
        "anomaly": ratio is not None and (ratio < 0.25 or ratio > 4.0),
    }
