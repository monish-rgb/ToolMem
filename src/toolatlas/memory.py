from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from .models import (
    EvidenceEntry,
    ExecutionRecord,
    ExecutionStep,
    Rollout,
    StrategyNode,
    ToolNode,
    ToolSpec,
    TraceNode,
    utc_now,
)
from .similarity import cosine_text, tokens, trigram_similarity
from .storage import SCHEMA_VERSION, SQLiteStore

ALLOWED_STATUSES = {"active", "stale", "invalid", "quarantined"}
GOVERNANCE_STATUSES = ALLOWED_STATUSES - {"active"}
DEFAULT_MAX_AGE_DAYS = 30
TRIGRAM_FALLBACK_THRESHOLD = 0.15
MAX_PLAYBOOK_STEPS = 8
MAX_AVOID_NOTES = 2
MAX_CONVENTIONS = 3


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _generic_rationale(text: str) -> str:
    """Remove common environment-specific literals from reusable memory."""
    text = re.sub(r"(?:[A-Za-z]:)?[/\\][\w./\\-]+", "<path>", text)
    text = re.sub(r"(['\"]).*?\1", "<value>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<value>", text)
    return " ".join(text.split()).strip()[:1000]


STRUCTURAL_RATIONALE_RE = re.compile(
    r"^(invoke|run|call|execute|use)\s+[a-z0-9_.\-]+\s*(tool)?(\s+during\s+.*|\s+for\s+.*)?$"
)


def _intent_rationale(tool: str, rationale: str, position: int, total: int) -> str:
    """Upgrade structural rationales to positional tool-use intent.

    Provided rationales carry real intent and are kept (sanitized). Only
    empty or structural placeholders (e.g. "invoke X during the rollout")
    are replaced with deterministic, agent-neutral intent derived from the
    step position — never invented chain-of-thought.
    """
    cleaned = _generic_rationale(rationale or "")
    if cleaned and not STRUCTURAL_RATIONALE_RE.match(cleaned.lower()):
        return cleaned
    if total <= 1:
        return f"apply {tool} to the task input and verify the result"
    if position == 0:
        return f"establish working scope and inputs with {tool}"
    if position == total - 1:
        return f"produce and verify the task output with {tool}"
    return f"advance the intermediate result toward the task goal with {tool}"


def _normalize_tool_name(name: str) -> str:
    """Canonical tool identity for grouping (case, separators, versions)."""
    normalized = re.sub(r"[-.\s]+", "_", name.strip().lower())
    normalized = re.sub(r"_v\d+$", "", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or name.strip().lower()


_ENTRY_BOILERPLATE_RE = re.compile(
    r"^(reliably supports workflows that|avoid or validate when|compose with)\s+",
    re.IGNORECASE,
)


def _entry_core(text: str) -> str:
    """Comparable core of an evidence entry without boilerplate framing."""
    core = _ENTRY_BOILERPLATE_RE.sub("", text.strip().lower())
    return re.sub(r"\s+", " ", core).strip(" .")


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        length += 1
    return length


def _entries_match(left: str, right: str) -> bool:
    """Exact match after normalization, shared-stem match, else trigram cores."""
    if left == right:
        return True
    if left.strip().lower() == right.strip().lower():
        return True
    left_core, right_core = _entry_core(left), _entry_core(right)
    if len(left_core) < 8 or len(right_core) < 8:
        return False
    if _common_prefix_length(left_core, right_core) >= 12:
        return True
    return trigram_similarity(left_core, right_core) >= 0.75


def estimate_tokens(text: str) -> int:
    """Deterministic guidance-size estimate (characters/4, no model needed)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def _fingerprint(spec: ToolSpec) -> str:
    canonical = json.dumps(
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
            "version": spec.version,
            "provider": spec.provider,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _confidence(successes: int, failures: int, *, boundary: bool = False) -> float:
    supporting = failures if boundary else successes
    return round((supporting + 1) / (successes + failures + 2), 4)


class ToolMemory:
    """Lifecycle-aware, provider-side ToolAtlas graph stored in SQLite."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.RLock()
        self._store = SQLiteStore(self.path) if self.path else None
        self.tools: dict[str, ToolNode] = {}
        self.traces: dict[str, TraceNode] = {}
        self.strategies: dict[str, StrategyNode] = {}
        self.executions: dict[str, ExecutionRecord] = {}
        if self._store:
            self._load()

    def register_tools(self, specs: Iterable[ToolSpec]) -> None:
        """Register schemas and invalidate knowledge learned against older ones."""
        with self._lock:
            for spec in specs:
                fingerprint = _fingerprint(spec)
                node = self.tools.get(spec.name)
                if node and node.schema_fingerprint and node.schema_fingerprint != fingerprint:
                    self._mark_tool_memory_stale(spec.name)
                if node is None:
                    node = ToolNode(name=spec.name)
                    self.tools[spec.name] = node
                node.description = spec.description
                node.input_schema = spec.input_schema
                node.version = spec.version
                node.provider = spec.provider
                node.schema_fingerprint = fingerprint
                node.status = "active"
            self.save()

    def induce(self, task_id: str, summary: str, rollouts: list[Rollout]) -> TraceNode:
        """Compress a verified rollout batch and update all three graph layers."""
        if not rollouts:
            raise ValueError("at least one rollout is required")
        if not task_id.strip() or not summary.strip():
            raise ValueError("task_id and summary are required")
        with self._lock:
            for rollout in rollouts:
                if not rollout.verifier_type.strip():
                    raise ValueError("every rollout requires a verifier_type")
                if not rollout.execution_id:
                    rollout.execution_id = f"run_{uuid.uuid4().hex}"
                self.executions[rollout.execution_id] = ExecutionRecord(
                    execution_id=rollout.execution_id,
                    task_id=task_id,
                    resolved=rollout.resolved,
                    verifier_type=rollout.verifier_type,
                    verified_at=rollout.verified_at,
                    observation=_generic_rationale(rollout.observation),
                )

            successful = [rollout for rollout in rollouts if rollout.resolved]
            backbone = min(successful or rollouts, key=lambda rollout: len(rollout.steps))
            total_steps = len(backbone.steps)
            neutral_steps = [
                ExecutionStep(
                    step.tool,
                    _intent_rationale(step.tool, step.rationale, position, total_steps),
                )
                for position, step in enumerate(backbone.steps)
            ]
            used_tools = _unique(step.tool for rollout in rollouts for step in rollout.steps)
            success_count = len(successful)
            failure_count = len(rollouts) - success_count
            fingerprints = {
                name: self.tools.get(name, ToolNode(name=name)).schema_fingerprint
                for name in used_tools
            }
            trace = TraceNode(
                qid=task_id,
                summary=summary,
                agent_neutral_trace=neutral_steps,
                task_level_tips=self._distill_tips(rollouts),
                tools=used_tools,
                source_executions=[rollout.execution_id for rollout in rollouts],
                success_count=success_count,
                failure_count=failure_count,
                confidence=_confidence(success_count, failure_count),
                status="active",
                tool_fingerprints=fingerprints,
                last_verified_at=max(rollout.verified_at for rollout in rollouts),
            )
            self.traces[task_id] = trace

            for tool_name in used_tools:
                node = self.tools.setdefault(tool_name, ToolNode(name=tool_name))
                if task_id not in node.traces:
                    node.traces.append(task_id)
                successful_steps = [
                    step for rollout in successful for step in rollout.steps if step.tool == tool_name
                ]
                sources = [
                    rollout.execution_id for rollout in successful
                    if tool_name in {candidate.tool for candidate in rollout.steps}
                ]
                for step in successful_steps:
                    self._merge_entry(
                        node.affordances,
                        f"Reliably supports workflows that {_generic_rationale(step.rationale).lower()}",
                        task_id,
                        sources,
                        successes=1,
                        failures=0,
                        fingerprint=node.schema_fingerprint,
                    )
                failed = [
                    rollout for rollout in rollouts
                    if not rollout.resolved and tool_name in {step.tool for step in rollout.steps}
                ]
                for rollout in failed:
                    caution = _generic_rationale(
                        rollout.observation or "the attempted inputs were not supported"
                    )
                    self._merge_entry(
                        node.boundaries,
                        f"Avoid or validate when {caution.lower()}",
                        task_id,
                        [rollout.execution_id],
                        successes=0,
                        failures=1,
                        fingerprint=node.schema_fingerprint,
                        boundary=True,
                    )

            successful_tools = _unique(
                step.tool for rollout in successful for step in rollout.steps
            )
            if len(successful_tools) > 1:
                successful_ids = [rollout.execution_id for rollout in successful]
                for tool_name in successful_tools:
                    peers = [name for name in successful_tools if name != tool_name]
                    self._merge_entry(
                        self.tools[tool_name].co_usage,
                        f"Compose with {', '.join(peers)} for multi-step workflows",
                        task_id,
                        successful_ids,
                        successes=max(1, success_count),
                        failures=0,
                        fingerprint=self.tools[tool_name].schema_fingerprint,
                        related_tools=peers,
                    )

            self._rebuild_trace_edges()
            self._induce_strategies()
            self.save()
            return trace

    def guide(
        self,
        task: str,
        top_k: int = 3,
        read_budget: int = 8,
        max_age_days: int = DEFAULT_MAX_AGE_DAYS,
        token_budget: int | None = None,
    ) -> dict[str, Any]:
        """Perform an auditable bounded traversal over currently valid memory.

        Seed retrieval is lexical first; when nothing matches, a trigram
        fallback rescues paraphrased queries (still empty when truly
        unrelated — no generic advice is ever invented). ``token_budget``
        optionally caps the rendered guidance size.
        """
        if top_k < 1 or read_budget < 1 or max_age_days < 1:
            raise ValueError("top_k, read_budget, and max_age_days must be positive")
        if token_budget is not None and token_budget < 1:
            raise ValueError("token_budget must be positive when set")
        with self._lock:
            candidates = [
                node for node in self.traces.values()
                if self._trace_is_current(node, max_age_days=max_age_days)
            ]
            ranked = sorted(
                ((cosine_text(task, node.summary), node.confidence, node.qid) for node in candidates),
                reverse=True,
            )
            seeds = [(sim, confidence, qid) for sim, confidence, qid in ranked[:top_k] if sim > 0]
            retrieval = "lexical"
            if not seeds:
                fallback = sorted(
                    (
                        (trigram_similarity(task, node.summary), node.confidence, node.qid)
                        for node in candidates
                    ),
                    reverse=True,
                )
                seeds = [
                    (sim, confidence, qid) for sim, confidence, qid in fallback[:top_k]
                    if sim >= TRIGRAM_FALLBACK_THRESHOLD
                ]
                retrieval = "trigram-fallback"
            if not seeds:
                return self._empty_guidance(read_budget)

            operations: list[dict[str, str]] = []
            selected: list[str] = []
            for _, _, qid in seeds:
                if len(operations) >= read_budget:
                    break
                selected.append(qid)
                operations.append({"action": "ReadTrace", "target": qid})

            if len(operations) + 2 <= read_budget:
                seed = self.traces[selected[0]]
                neighbors = [
                    qid for qid in seed.neighbors
                    if qid not in selected
                    and self._trace_is_current(self.traces[qid], max_age_days=max_age_days)
                ]
                neighbors.sort(key=lambda qid: cosine_text(task, self.traces[qid].summary), reverse=True)
                if neighbors and cosine_text(task, self.traces[neighbors[0]].summary) > 0:
                    operations.append({"action": "Expand", "target": seed.qid})
                    selected.append(neighbors[0])
                    operations.append({"action": "ReadTrace", "target": neighbors[0]})

            relevant_tools = _unique(
                tool for qid in selected for tool in self.traces[qid].tools
                if tool in self.tools and self.tools[tool].status == "active"
            )
            successful_selected = [
                self.traces[qid] for qid in selected if self.traces[qid].success_count > 0
            ]
            playbook: list[dict[str, Any]] = []
            seen_steps: set[tuple[str, str]] = set()
            for trace in successful_selected:
                for step in trace.agent_neutral_trace:
                    key = (step.tool, step.rationale)
                    if step.tool in relevant_tools and key not in seen_steps:
                        playbook.append(
                            {
                                "tool": step.tool,
                                "rationale": step.rationale,
                                "confidence": trace.confidence,
                                "source_trace": trace.qid,
                            }
                        )
                        seen_steps.add(key)
                    if len(playbook) >= MAX_PLAYBOOK_STEPS:
                        break
                if len(playbook) >= MAX_PLAYBOOK_STEPS:
                    break

            task_terms = set(tokens(task))
            tool_tips: list[dict[str, Any]] = []
            avoid: list[dict[str, Any]] = []
            # Once the playbook is sufficient, read at most two more tools and
            # only for boundary cautions — extra affordance reads rarely change
            # the plan but always cost traversal budget.
            max_tool_reads = 2 if len(playbook) >= 6 else len(relevant_tools)
            tools_read = 0
            for tool_name in relevant_tools:
                if len(operations) >= read_budget or tools_read >= max_tool_reads:
                    break
                tools_read += 1
                operations.append({"action": "ReadTool", "target": tool_name})
                entries = [
                    entry for entry in (
                        self.tools[tool_name].affordances
                        + self.tools[tool_name].boundaries
                        + self.tools[tool_name].co_usage
                    ) if entry.status == "active"
                ]
                entries.sort(
                    key=lambda entry: (
                        len(task_terms & set(tokens(entry.text))),
                        entry.confidence,
                        len(entry.source_executions),
                    ),
                    reverse=True,
                )
                for entry in entries[:2]:
                    tool_tips.append(
                        {
                            "tool": tool_name,
                            "tip": entry.text,
                            "confidence": entry.confidence,
                            "evidence_count": len(entry.source_executions),
                            "source_queries": entry.source_queries,
                        }
                    )
                for entry in self.tools[tool_name].boundaries:
                    if entry.status == "active" and len(avoid) < MAX_AVOID_NOTES:
                        avoid.append(
                            {
                                "tool": tool_name,
                                "caution": entry.text,
                                "confidence": entry.confidence,
                                "evidence_count": len(entry.source_executions),
                            }
                        )

            strategy: list[dict[str, Any]] = []
            for node in self.strategies.values():
                if len(operations) >= read_budget:
                    break
                if node.status == "active" and any(qid in selected for qid in node.source_queries):
                    operations.append({"action": "ReadStrategy", "target": node.sid})
                    strategy.append(
                        {
                            "text": node.text,
                            "source_strategy_id": node.sid,
                            "confidence": node.confidence,
                        }
                    )

            conventions: list[dict[str, str]] = []
            for node in strategy[:2]:
                conventions.append({"text": node["text"], "source": node["source_strategy_id"]})
            for _, _, qid in seeds:
                for tip in self.traces[qid].task_level_tips:
                    if len(conventions) >= MAX_CONVENTIONS:
                        break
                    if all(tip != existing["text"] for existing in conventions):
                        conventions.append({"text": tip, "source": qid})
                if len(conventions) >= MAX_CONVENTIONS:
                    break

            truncated = False
            if token_budget is not None:
                playbook, avoid, conventions, truncated = self._fit_token_budget(
                    playbook, avoid, conventions, token_budget
                )

            operations.append({"action": "Done", "target": "guidance"})
            guidance = {
                "schema_version": SCHEMA_VERSION,
                "seed_candidates": [
                    {
                        "qid": qid,
                        "sim": round(sim, 4),
                        "confidence": confidence,
                        "summary": self.traces[qid].summary,
                    }
                    for sim, confidence, qid in seeds
                ],
                "playbook": playbook,
                "strategy": strategy,
                "tool_tips": tool_tips,
                "avoid": avoid,
                "conventions": conventions,
                "truncated": truncated,
                "traversal": {
                    "operations": operations,
                    "reads_used": sum(op["action"] != "Done" for op in operations),
                    "read_budget": read_budget,
                    "retrieval": retrieval,
                },
                "coverage": {
                    "seed_count": len(seeds),
                    "tools_covered": relevant_tools,
                    "retrieval": retrieval,
                },
            }
            guidance["guidance_tokens_estimate"] = estimate_tokens(
                json.dumps(
                    {
                        "seed_candidates": guidance["seed_candidates"],
                        "playbook": playbook,
                        "avoid": avoid,
                        "conventions": conventions,
                    },
                    sort_keys=True,
                )
            )
            return guidance

    def reverify_trace(
        self, task_id: str, resolved: bool, verifier_type: str, observation: str = ""
    ) -> TraceNode:
        """Refresh a trace against current tool schemas after external verification."""
        if not verifier_type.strip():
            raise ValueError("verifier_type is required")
        with self._lock:
            trace = self.traces.get(task_id)
            if not trace:
                raise ValueError(f"unknown trace: {task_id}")
            execution_id = f"run_{uuid.uuid4().hex}"
            now = utc_now()
            self.executions[execution_id] = ExecutionRecord(
                execution_id, task_id, resolved, verifier_type, now,
                _generic_rationale(observation),
            )
            trace.source_executions.append(execution_id)
            trace.success_count += int(resolved)
            trace.failure_count += int(not resolved)
            trace.confidence = _confidence(trace.success_count, trace.failure_count)
            trace.last_verified_at = now
            trace.status = "active" if resolved else "invalid"
            trace.status_reason = "" if resolved else (observation or "re-verification failed")
            trace.tool_fingerprints = {
                tool: self.tools[tool].schema_fingerprint
                for tool in trace.tools if tool in self.tools
            }
            for tool_name in trace.tools:
                node = self.tools.get(tool_name)
                if not node:
                    continue
                for entry in node.affordances + node.boundaries + node.co_usage:
                    if task_id in entry.source_queries:
                        entry.source_executions.append(execution_id)
                        entry.success_count += int(resolved)
                        entry.failure_count += int(not resolved)
                        entry.confidence = _confidence(
                            entry.success_count,
                            entry.failure_count,
                            boundary=entry in node.boundaries,
                        )
                        entry.status = "active" if resolved else "invalid"
                        entry.last_verified_at = now
                        entry.tool_fingerprint = node.schema_fingerprint
            self._induce_strategies()
            self.save()
            return trace

    def set_trace_status(self, task_id: str, status: str, reason: str) -> TraceNode:
        """Governance hook to quarantine or invalidate memory with an audit reason."""
        if status not in GOVERNANCE_STATUSES:
            raise ValueError(f"status must be one of {sorted(GOVERNANCE_STATUSES)}")
        if not reason.strip():
            raise ValueError("a governance reason is required")
        with self._lock:
            if task_id not in self.traces:
                raise ValueError(f"unknown trace: {task_id}")
            self.traces[task_id].status = status
            self.traces[task_id].status_reason = _generic_rationale(reason)
            self._induce_strategies()
            self.save()
            return self.traces[task_id]

    def refresh_status(self, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> dict[str, Any]:
        if max_age_days < 1:
            raise ValueError("max_age_days must be positive")
        stale_traces = [
            qid for qid, trace in self.traces.items()
            if not self._trace_is_current(trace, max_age_days=max_age_days)
        ]
        return {
            "stale_or_blocked_traces": stale_traces,
            "stale_tools": [name for name, node in self.tools.items() if node.status == "stale"],
            "refresh_candidates": [
                {
                    "qid": qid,
                    "summary": self.traces[qid].summary,
                    "reason": self._refresh_reason(self.traces[qid], max_age_days),
                }
                for qid in stale_traces
            ],
        }

    def stats(self) -> dict[str, int]:
        return {
            "tools": len(self.tools),
            "traces": len(self.traces),
            "strategies": len(self.strategies),
            "trace_edges": sum(len(trace.neighbors) for trace in self.traces.values()) // 2,
            "executions": len(self.executions),
            "stale_traces": sum(not self._trace_is_current(trace) for trace in self.traces.values()),
            "refresh_due": sum(
                not self._trace_is_current(trace, max_age_days=DEFAULT_MAX_AGE_DAYS)
                for trace in self.traces.values()
            ),
        }

    def tool_details(self, name: str) -> dict[str, Any] | None:
        node = self.tools.get(name)
        return asdict(node) if node else None

    def suggest_probes(self, name: str) -> dict[str, Any]:
        node = self.tools.get(name)
        if not node:
            raise ValueError(f"unknown tool: {name}")
        peers = _unique(peer for entry in node.co_usage for peer in entry.related_tools)
        return {
            "tool_fingerprint": node.schema_fingerprint,
            "boundary": [{
                "direction": "outward",
                "task": f"Test {name} with empty, malformed, unauthorized, and unusually large inputs.",
            }],
            "affordance": [{
                "direction": "inward",
                "task": (
                    f"Verify a realistic workflow that combines {name} with {peers[0]}."
                    if peers else f"Verify a representative use of {name} not covered by existing traces."
                ),
            }],
        }

    def save(self) -> None:
        if not self._store:
            return
        self._store.save(
            {
                "tools": {key: asdict(value) for key, value in self.tools.items()},
                "traces": {key: asdict(value) for key, value in self.traces.items()},
                "strategies": {key: asdict(value) for key, value in self.strategies.items()},
                "executions": {key: asdict(value) for key, value in self.executions.items()},
            }
        )

    def _load(self) -> None:
        assert self._store is not None
        payload = self._store.load()
        self.tools = {
            key: ToolNode(
                name=value["name"],
                description=value.get("description", ""),
                affordances=[EvidenceEntry(**entry) for entry in value.get("affordances", [])],
                boundaries=[EvidenceEntry(**entry) for entry in value.get("boundaries", [])],
                co_usage=[EvidenceEntry(**entry) for entry in value.get("co_usage", [])],
                traces=value.get("traces", []),
                input_schema=value.get("input_schema", {}),
                version=value.get("version", "unknown"),
                provider=value.get("provider", "local"),
                schema_fingerprint=value.get("schema_fingerprint", ""),
                status=value.get("status", "active"),
            )
            for key, value in payload["tools"].items()
        }
        self.traces = {
            key: TraceNode(
                qid=value["qid"],
                summary=value["summary"],
                agent_neutral_trace=[ExecutionStep(**step) for step in value["agent_neutral_trace"]],
                task_level_tips=value.get("task_level_tips", []),
                tools=value.get("tools", []),
                neighbors=value.get("neighbors", []),
                source_executions=value.get("source_executions", []),
                success_count=value.get("success_count", 0),
                failure_count=value.get("failure_count", 0),
                confidence=value.get("confidence", 0.5),
                status=value.get("status", "active"),
                status_reason=value.get("status_reason", ""),
                tool_fingerprints=value.get("tool_fingerprints", {}),
                last_verified_at=value.get("last_verified_at", utc_now()),
            )
            for key, value in payload["traces"].items()
        }
        self.strategies = {key: StrategyNode(**value) for key, value in payload["strategies"].items()}
        self.executions = {
            key: ExecutionRecord(**value) for key, value in payload["executions"].items()
        }

    def _mark_tool_memory_stale(self, tool_name: str) -> None:
        node = self.tools[tool_name]
        for entry in node.affordances + node.boundaries + node.co_usage:
            entry.status = "stale"
        for trace in self.traces.values():
            if tool_name in trace.tools:
                trace.status = "stale"
                trace.status_reason = f"tool schema changed: {tool_name}"
        for strategy in self.strategies.values():
            if tool_name in strategy.tool_sequence:
                strategy.status = "stale"

    def _trace_is_current(self, trace: TraceNode, max_age_days: int | None = None) -> bool:
        if trace.status != "active":
            return False
        schemas_current = all(
            tool in self.tools
            and self.tools[tool].status == "active"
            and trace.tool_fingerprints.get(tool) == self.tools[tool].schema_fingerprint
            for tool in trace.tools
        )
        if not schemas_current:
            return False
        if max_age_days is None:
            return True
        try:
            verified_at = datetime.fromisoformat(trace.last_verified_at)
            if verified_at.tzinfo is None:
                verified_at = verified_at.replace(tzinfo=UTC)
        except ValueError:
            return False
        return verified_at >= datetime.now(UTC) - timedelta(days=max_age_days)

    def _refresh_reason(self, trace: TraceNode, max_age_days: int) -> str:
        if trace.status != "active":
            return trace.status_reason or trace.status
        if not self._trace_is_current(trace):
            return "tool_schema_changed"
        if not self._trace_is_current(trace, max_age_days=max_age_days):
            return "verification_expired"
        return "current"

    @staticmethod
    def _merge_entry(
        entries: list[EvidenceEntry], text: str, task_id: str, execution_ids: list[str], *,
        successes: int, failures: int, fingerprint: str, boundary: bool = False,
        related_tools: list[str] | None = None,
    ) -> None:
        entry = next(
            (candidate for candidate in entries if _entries_match(candidate.text, text)),
            None,
        )
        if entry is None:
            entry = EvidenceEntry(text=text, related_tools=related_tools or [])
            entries.append(entry)
        if task_id not in entry.source_queries:
            entry.source_queries.append(task_id)
        entry.source_executions = _unique(entry.source_executions + execution_ids)
        entry.success_count += successes
        entry.failure_count += failures
        entry.confidence = _confidence(entry.success_count, entry.failure_count, boundary=boundary)
        entry.status = "active"
        entry.last_verified_at = utc_now()
        entry.tool_fingerprint = fingerprint

    @staticmethod
    def _distill_tips(rollouts: list[Rollout]) -> list[str]:
        """Distill reusable tips from successes and failures alike.

        Successes yield planning tips (scope discovery, ordered verification);
        failures yield fix tips pairing what went wrong with the correction, so
        future rollouts skip the same wasted calls.
        """
        tips: list[str] = []
        successful = [rollout for rollout in rollouts if rollout.resolved]
        failed = [rollout for rollout in rollouts if not rollout.resolved]
        if successful:
            tips.append("Verify the final tool result against the task requirement.")
            longest = max(len(rollout.steps) for rollout in successful)
            if longest > 1:
                tips.append(
                    "Discover the working scope first, then apply tools in order "
                    "and verify each intermediate result."
                )
        for rollout in failed:
            if not rollout.observation:
                continue
            cause = _generic_rationale(rollout.observation)
            tips.append(f"Validate inputs first: {cause}")
            tips.append(
                f"If the verifier reports {cause}, correct the inputs and retry "
                "instead of repeating the failed call."
            )
        return _unique(tips)

    def _rebuild_trace_edges(self) -> None:
        for node in self.traces.values():
            node.neighbors = []
        ids = list(self.traces)
        for index, left_id in enumerate(ids):
            scores: list[tuple[float, str]] = []
            for right_id in ids[index + 1:]:
                left, right = self.traces[left_id], self.traces[right_id]
                score = cosine_text(left.summary, right.summary)
                if set(left.tools) & set(right.tools):
                    score += 0.25
                if score > 0.2:
                    scores.append((score, right_id))
            for _, right_id in sorted(scores, reverse=True)[:3]:
                self.traces[left_id].neighbors.append(right_id)
                self.traces[right_id].neighbors.append(left_id)

    @staticmethod
    def _is_near_sequence(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
        """Same tool order modulo at most one inserted or missing step."""
        if not (len(longer) - len(shorter) == 1 and len(shorter) >= 1):
            return False
        index = 0
        skipped = False
        for tool in longer:
            if index < len(shorter) and tool == shorter[index]:
                index += 1
            elif not skipped:
                skipped = True
            else:
                return False
        return index == len(shorter)

    def _induce_strategies(self) -> None:
        sequences: dict[tuple[str, ...], list[str]] = {}
        canonical_of: dict[str, tuple[str, ...]] = {}
        for qid, trace in self.traces.items():
            if len(trace.agent_neutral_trace) <= 1 or trace.success_count == 0:
                continue
            if not self._trace_is_current(trace):
                continue
            canonical = tuple(_normalize_tool_name(step.tool) for step in trace.agent_neutral_trace)
            canonical_of[qid] = canonical
            sequences.setdefault(canonical, []).append(qid)
        # Merge near-duplicate groups (same order, at most one step apart) so
        # strategies form across paraphrased rollouts instead of exact repeats.
        groups = list(sequences.items())
        merged: list[tuple[tuple[str, ...], list[str]]] = []
        consumed: set[int] = set()
        for index, (canonical, qids) in enumerate(groups):
            if index in consumed:
                continue
            if len(qids) >= 2:
                merged.append((canonical, list(qids)))
                consumed.add(index)
                continue
            for other_index, (other_canonical, other_qids) in enumerate(groups):
                if other_index in consumed or other_index == index or len(other_qids) < 1:
                    continue
                short, long = sorted([canonical, other_canonical], key=len)
                if self._is_near_sequence(short, long):
                    merged.append((long, _unique(qids + other_qids)))
                    consumed.add(index)
                    consumed.add(other_index)
                    break
            else:
                merged.append((canonical, list(qids)))
                consumed.add(index)
        self.strategies = {}
        for canonical, qids in merged:
            if len(_unique(qids)) >= 2:
                unique_qids = _unique(qids)
                longest = max(
                    unique_qids,
                    key=lambda qid: len(self.traces[qid].agent_neutral_trace),
                )
                display = [
                    step.tool for step in self.traces[longest].agent_neutral_trace
                ]
                confidence = round(
                    sum(self.traces[qid].confidence for qid in unique_qids) / len(unique_qids), 4
                )
                sid = f"strategy_{len(self.strategies) + 1}"
                self.strategies[sid] = StrategyNode(
                    sid=sid,
                    text=f"Apply {' then '.join(display)} and verify the composed result.",
                    source_queries=unique_qids,
                    tool_sequence=list(canonical),
                    confidence=confidence,
                    status="active",
                )

    @staticmethod
    def _fit_token_budget(
        playbook: list[dict[str, Any]],
        avoid: list[dict[str, Any]],
        conventions: list[dict[str, str]],
        token_budget: int,
    ) -> tuple[list, list, list, bool]:
        """Shrink guidance to a token budget: conventions, avoid, then playbook.

        Order preserves the highest call-saving density first: planning
        conventions, then failure-avoidance cautions, then the step sequence
        (kept in order, trimmed from the tail).
        """
        def size(play: list, avo: list, con: list) -> int:
            return estimate_tokens(json.dumps({"p": play, "a": avo, "c": con}, sort_keys=True))

        truncated = False
        sections: list[list] = [conventions, avoid, playbook]
        while size(playbook, avoid, conventions) > token_budget and any(sections):
            for section in sections:
                if section:
                    section.pop()
                    truncated = True
                    break
        return playbook, avoid, conventions, truncated

    @staticmethod
    def _empty_guidance(read_budget: int) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "seed_candidates": [], "playbook": [], "strategy": [], "tool_tips": [],
            "avoid": [], "conventions": [], "truncated": False,
            "guidance_tokens_estimate": 0,
            "traversal": {
                "operations": [{"action": "Done", "target": "no_relevant_current_memory"}],
                "reads_used": 0,
                "read_budget": read_budget,
                "retrieval": "none",
            },
            "coverage": {"seed_count": 0, "tools_covered": [], "retrieval": "none"},
        }

    def reverification_due(self, max_age_days: int = DEFAULT_MAX_AGE_DAYS) -> dict[str, Any]:
        """List traces needing a re-verification run, most overdue first.

        Covers expired verification, schema drift, and governance-blocked
        traces — the paper's lifecycle gap: stale guidance causes wasted
        inference calls, so refresh scheduling is a call-saving mechanism.
        """
        if max_age_days < 1:
            raise ValueError("max_age_days must be positive")
        now = datetime.now(UTC)
        due: list[dict[str, Any]] = []
        for qid, trace in self.traces.items():
            reason = self._refresh_reason(trace, max_age_days=max_age_days)
            if reason == "current":
                continue
            try:
                verified_at = datetime.fromisoformat(trace.last_verified_at)
                if verified_at.tzinfo is None:
                    verified_at = verified_at.replace(tzinfo=UTC)
                overdue = max(0, (now - verified_at).days - max_age_days)
            except ValueError:
                overdue = max_age_days
            due.append(
                {
                    "qid": qid,
                    "summary": trace.summary,
                    "reason": reason,
                    "days_overdue": overdue,
                    "success_count": trace.success_count,
                    "failure_count": trace.failure_count,
                }
            )
        due.sort(key=lambda item: (item["reason"] != "verification_expired", -item["days_overdue"]))
        return {"traces_due": due, "count": len(due)}
