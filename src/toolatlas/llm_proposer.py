"""LLM seed-task and probe proposal (paper Memory Bootstrapping, Phase 3).

Replaces the deterministic ``propose_seed_tasks`` template with a
task-designer LLM call. Outputs are schema-constrained and grounded:
unknown tool names, missing summaries, and environment literals are
rejected or sanitized — never ingested. Callers choose fallback behavior
explicitly via ``on_failure`` so silent degradation is always accounted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .llm_client import LLMCall, LLMResponse
from .memory import _generic_rationale
from .offline_builder import OfflineCost

SEED_DESIGNER_SYSTEM = (
    "You are a task designer for a benchmark that evaluates LLM agents using "
    "provider tools. Design small, verifiable filesystem-style tasks that each "
    "exercise the TARGET TOOL exactly once or twice inside a realistic "
    "multi-step workflow. Rules: use only tools from ALLOWED TOOLS; every "
    "step needs a concrete goal; summaries must be environment-invariant "
    "(no paths, filenames, numbers, secrets, or account IDs); no "
    "chain-of-thought, only the task. Respond with strict JSON only: "
    '{"tasks": [{"summary": "...", "steps": [{"tool": "...", "goal": "..."}]}]}'
)

PROBE_DESIGNER_SYSTEM = (
    "You are a capability explorer for provider tools. Given a tool and what "
    "is already known about it, propose ONE inward affordance probe (a "
    "realistic workflow use not yet covered) and ONE outward boundary probe "
    "(empty, malformed, unauthorized, or unusually large inputs). Rules: "
    "environment-invariant wording, no secrets or paths, no chain-of-thought. "
    'Respond with strict JSON only: {"affordance": "<task>", "boundary": "<task>"}'
)

SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password|bearer)\s*[:=]\s*\S+"),
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"nvapi-[A-Za-z0-9_.\-~+/=]+"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]+"),
)


class ProposalError(ValueError):
    """Schema, grounding, or sanitization failure for an LLM proposal."""


def scrub_proposal_text(text: str, limit: int = 500) -> str:
    """Strip secrets then environment literals from proposed text."""
    cleaned = str(text or "")
    for pattern in SECRET_PATTERNS:
        cleaned = pattern.sub("<secret>", cleaned)
    return _generic_rationale(cleaned)[:limit]


def _extract_json(text: str) -> Any:
    cleaned = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ProposalError(f"proposal is not strict JSON: {exc}") from exc


def _charge(cost: OfflineCost | None, response: LLMResponse) -> None:
    if cost is None:
        return
    usage = response.usage
    cost.llm_calls += 1
    if usage and usage.has_authoritative_input:
        cost.llm_input_tokens += usage.input_tokens or 0
        cost.llm_output_tokens += usage.output_tokens or 0


@dataclass(slots=True)
class SeedTask:
    task_id: str
    summary: str
    steps: list[dict[str, str]] = field(default_factory=list)


def propose_seed_tasks(
    target_tool: str,
    allowed_tools: list[str],
    llm: LLMCall,
    *,
    count: int = 3,
    cost: OfflineCost | None = None,
) -> list[SeedTask]:
    """Design ``count`` grounded seed tasks exercising ``target_tool``."""
    if target_tool not in allowed_tools:
        raise ProposalError(f"target tool not in allowlist: {target_tool!r}")
    user = (
        f"TARGET TOOL: {target_tool}\n"
        f"ALLOWED TOOLS: {', '.join(sorted(allowed_tools))}\n"
        f"Design exactly {count} tasks."
    )
    response = llm(SEED_DESIGNER_SYSTEM, user)
    _charge(cost, response)
    payload = _extract_json(response.text)
    items = payload.get("tasks") if isinstance(payload, dict) else None
    if not isinstance(items, list) or len(items) != count:
        raise ProposalError(f"expected exactly {count} tasks, got {items!r}"[:200])
    seeds: list[SeedTask] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ProposalError(f"task {index} is not an object")
        summary = scrub_proposal_text(item.get("summary", ""))
        if not summary:
            raise ProposalError(f"task {index} has an empty summary after sanitization")
        raw_steps = item.get("steps") or []
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ProposalError(f"task {index} has no steps")
        steps: list[dict[str, str]] = []
        for position, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                raise ProposalError(f"task {index} step {position} is not an object")
            tool = str(step.get("tool", "")).strip()
            if tool not in allowed_tools:
                raise ProposalError(
                    f"task {index} step {position} uses unknown tool {tool!r}")
            goal = scrub_proposal_text(step.get("goal", ""))
            if not goal:
                raise ProposalError(f"task {index} step {position} has an empty goal")
            steps.append({"tool": tool, "goal": goal})
        if target_tool not in {step["tool"] for step in steps}:
            raise ProposalError(f"task {index} never exercises {target_tool}")
        seeds.append(SeedTask(
            task_id=f"seed_{target_tool}_{index + 1}",
            summary=summary, steps=steps))
    return seeds


@dataclass(slots=True)
class ProbeProposal:
    tool: str
    affordance: str
    boundary: str


def propose_probes(
    tool: str,
    known_affordances: list[str],
    known_boundaries: list[str],
    llm: LLMCall,
    *,
    cost: OfflineCost | None = None,
) -> ProbeProposal:
    """Propose one novel affordance and one boundary probe for ``tool``."""
    user = (
        f"TOOL: {tool}\n"
        f"ALREADY COVERED AFFORDANCES: {' | '.join(known_affordances) or 'none'}\n"
        f"ALREADY COVERED BOUNDARIES: {' | '.join(known_boundaries) or 'none'}\n"
        "Propose probes not covered above."
    )
    response = llm(PROBE_DESIGNER_SYSTEM, user)
    _charge(cost, response)
    payload = _extract_json(response.text)
    if not isinstance(payload, dict):
        raise ProposalError("probe proposal must be a JSON object")
    affordance = scrub_proposal_text(payload.get("affordance", ""))
    boundary = scrub_proposal_text(payload.get("boundary", ""))
    if not affordance or not boundary:
        raise ProposalError("probe proposal is missing affordance or boundary text")
    return ProbeProposal(tool=tool, affordance=affordance, boundary=boundary)
