"""LLM trace reflection and capability aggregation (Phase 3).

Replaces the deterministic positional-rationale template and fixed tip list
with reflection over verified execution records. Everything the model emits
is schema-constrained and grounded before it may enter memory:

- one rationale per step, referencing only tools present in the trace;
- tips that survive secret/literal sanitization;
- aggregated capability statements that overlap an observed candidate
  (trigram check), so unsupported inventions are rejected, not stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .llm_client import LLMCall, LLMResponse
from .llm_proposer import ProposalError, _charge, _extract_json, scrub_proposal_text
from .similarity import trigram_similarity

REFLECT_SYSTEM = (
    "You are a memory distiller for tool-using agents. Given verified "
    "execution records (tool steps plus independent verifier verdicts), write "
    "agent-neutral memory: one short rationale per step describing the "
    "positional intent of that tool use (no chain-of-thought, no first-person "
    "reasoning, no file names, paths, numbers, or secrets), and task-level "
    "tips distilled from both successes and failures (each tip pairs an "
    "observed outcome with the reusable correction). Every statement must be "
    "grounded in the records — invent nothing. Respond with strict JSON only: "
    '{"rationales": ["..."], "tips": ["..."]}'
)

AGGREGATE_SYSTEM = (
    "You are a capability aggregator. Merge each candidate list into at most "
    "three canonical statements, preserving meaning without adding new "
    "claims. Drop duplicates and statements unsupported by the candidates. "
    "Respond with strict JSON only: "
    '{"affordances": ["..."], "boundaries": ["..."]}'
)

GROUNDING_TRIGRAM = 0.3
MAX_TIPS = 5
MAX_RATIONALE_CHARS = 300


def _tool_mentions(text: str, tool_names: list[str], exclude: str = "") -> list[str]:
    mentioned = []
    for name in tool_names:
        if name != exclude and re.search(rf"\b{re.escape(name)}\b", text or ""):
            mentioned.append(name)
    return mentioned


@dataclass(slots=True)
class ReflectedTrace:
    step_rationales: list[str] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)


def reflect_trace(
    summary: str,
    steps: list[dict[str, str]],
    outcomes: list[dict[str, Any]],
    allowed_tools: list[str],
    llm: LLMCall,
    *,
    cost: Any | None = None,
) -> ReflectedTrace:
    """Distill grounded rationales and tips from verified execution records."""
    if not steps:
        raise ProposalError("reflection requires at least one step")
    record_lines = []
    for position, step in enumerate(steps):
        record_lines.append(
            f"step {position}: tool={step.get('tool', '?')} "
            f"attempted={scrub_proposal_text(step.get('rationale', ''), 200)}")
    for index, outcome in enumerate(outcomes):
        record_lines.append(
            f"run {index}: resolved={bool(outcome.get('resolved'))} "
            f"verifier={outcome.get('verifier_type', 'external')} "
            f"observation={scrub_proposal_text(outcome.get('observation', ''), 200)}")
    user = f"TASK: {scrub_proposal_text(summary, 200)}\n" + "\n".join(record_lines)
    response: LLMResponse = llm(REFLECT_SYSTEM, user)
    _charge(cost, response)
    payload = _extract_json(response.text)
    if not isinstance(payload, dict):
        raise ProposalError("reflection must be a JSON object")
    rationales = payload.get("rationales")
    if not isinstance(rationales, list) or len(rationales) != len(steps):
        raise ProposalError(
            f"expected {len(steps)} rationales, got {rationales!r}"[:200])
    grounded: list[str] = []
    for position, (raw, step) in enumerate(zip(rationales, steps)):
        cleaned = scrub_proposal_text(raw, MAX_RATIONALE_CHARS)
        if not cleaned:
            raise ProposalError(f"rationale {position} is empty after sanitization")
        strangers = _tool_mentions(cleaned, allowed_tools, exclude=str(step.get("tool", "")))
        if strangers:
            raise ProposalError(
                f"rationale {position} references tools outside the trace: {strangers}")
        grounded.append(cleaned)
    tips: list[str] = []
    for raw_tip in (payload.get("tips") or [])[:MAX_TIPS]:
        cleaned = scrub_proposal_text(raw_tip, 300)
        if cleaned and all(cleaned != existing for existing in tips):
            tips.append(cleaned)
    return ReflectedTrace(step_rationales=grounded, tips=tips)


@dataclass(slots=True)
class AggregatedCapabilities:
    affordances: list[str] = field(default_factory=list)
    boundaries: list[str] = field(default_factory=list)


def aggregate_capabilities(
    candidates: dict[str, list[str]],
    llm: LLMCall,
    *,
    cost: Any | None = None,
) -> AggregatedCapabilities:
    """Merge candidate evidence into canonical statements (grounded only)."""
    known = [text for group in candidates.values() for text in group]
    user = "CANDIDATES:\n" + "\n".join(f"- {scrub_proposal_text(t, 300)}" for t in known)
    response: LLMResponse = llm(AGGREGATE_SYSTEM, user)
    _charge(cost, response)
    payload = _extract_json(response.text)
    if not isinstance(payload, dict):
        raise ProposalError("aggregation must be a JSON object")
    result = AggregatedCapabilities()
    for key in ("affordances", "boundaries"):
        for raw in (payload.get(key) or [])[:3]:
            cleaned = scrub_proposal_text(raw, 300)
            if not cleaned:
                continue
            if not any(trigram_similarity(cleaned, scrub_proposal_text(candidate, 300))
                       >= GROUNDING_TRIGRAM for candidate in known):
                continue  # unsupported invention: reject, do not store
            target = getattr(result, key)
            if all(cleaned != existing for existing in target):
                target.append(cleaned)
    return result
