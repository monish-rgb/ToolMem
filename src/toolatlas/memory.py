from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .models import (
    EvidenceEntry,
    ExecutionStep,
    Rollout,
    StrategyNode,
    ToolNode,
    ToolSpec,
    TraceNode,
)
from .similarity import cosine_text, tokens


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _generic_rationale(text: str) -> str:
    """Remove environment-specific literals while preserving tool-use intent."""
    text = re.sub(r"(?:[A-Za-z]:)?[/\\][\w./\\-]+", "<path>", text)
    text = re.sub(r"(['\"]).*?\1", "<value>", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\b", "<value>", text)
    return " ".join(text.split()).strip()


class ToolMemory:
    """Persistent three-layer ToolAtlas graph with deterministic induction."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.tools: dict[str, ToolNode] = {}
        self.traces: dict[str, TraceNode] = {}
        self.strategies: dict[str, StrategyNode] = {}
        if self.path and self.path.exists():
            self._load()

    def register_tools(self, specs: Iterable[ToolSpec]) -> None:
        for spec in specs:
            node = self.tools.setdefault(spec.name, ToolNode(name=spec.name))
            node.description = spec.description
        self.save()

    def induce(self, task_id: str, summary: str, rollouts: list[Rollout]) -> TraceNode:
        """Compress verified rollouts and update trace, capability, and strategy layers."""
        if not rollouts:
            raise ValueError("at least one rollout is required")
        successful = [r for r in rollouts if r.resolved]
        backbone = min(successful or rollouts, key=lambda r: len(r.steps))
        neutral_steps = [
            ExecutionStep(step.tool, _generic_rationale(step.rationale))
            for step in backbone.steps
        ]
        used_tools = _unique(step.tool for r in rollouts for step in r.steps)
        tips = self._distill_tips(rollouts)
        trace = TraceNode(task_id, summary, neutral_steps, tips, used_tools)
        self.traces[task_id] = trace

        for tool_name in used_tools:
            node = self.tools.setdefault(tool_name, ToolNode(name=tool_name))
            if task_id not in node.traces:
                node.traces.append(task_id)
            successful_steps = [
                step for r in successful for step in r.steps if step.tool == tool_name
            ]
            for step in successful_steps:
                self._merge_entry(
                    node.affordances,
                    f"Reliably supports workflows that {_generic_rationale(step.rationale).lower()}",
                    task_id,
                )
            failed = [r for r in rollouts if not r.resolved and tool_name in {s.tool for s in r.steps}]
            for rollout in failed:
                caution = _generic_rationale(rollout.observation or "the attempted inputs were not supported")
                self._merge_entry(node.boundaries, f"Avoid or validate when {caution.lower()}", task_id)

        if len(used_tools) > 1:
            for tool_name in used_tools:
                peers = [name for name in used_tools if name != tool_name]
                self._merge_entry(
                    self.tools[tool_name].co_usage,
                    f"Compose with {', '.join(peers)} for multi-step workflows",
                    task_id,
                    peers,
                )

        self._rebuild_trace_edges()
        self._induce_strategies()
        self.save()
        return trace

    def guide(self, task: str, top_k: int = 3, read_budget: int = 8) -> dict[str, Any]:
        """Perform a bounded graph traversal and return task-conditioned guidance."""
        if top_k < 1 or read_budget < 1:
            raise ValueError("top_k and read_budget must be positive")
        ranked = sorted(
            ((cosine_text(task, node.summary), qid) for qid, node in self.traces.items()),
            reverse=True,
        )
        seeds = [(sim, qid) for sim, qid in ranked[:top_k] if sim > 0]
        if not seeds:
            return {"seed_candidates": [], "playbook": [], "strategy": [], "tool_tips": []}

        selected: list[str] = []
        reads = 0
        frontier = [qid for _, qid in seeds]
        while frontier and reads < read_budget:
            qid = frontier.pop(0)
            if qid in selected:
                continue
            selected.append(qid)  # ReadTrace
            reads += 1
            if reads >= read_budget:
                break
            for neighbor in self.traces[qid].neighbors:  # Expand
                if neighbor not in selected and neighbor not in frontier:
                    frontier.append(neighbor)

        task_terms = set(tokens(task))
        relevant_tools = _unique(
            tool for qid in selected for tool in self.traces[qid].tools
        )
        playbook: list[dict[str, str]] = []
        seen_steps: set[tuple[str, str]] = set()
        for qid in selected:
            for step in self.traces[qid].agent_neutral_trace:
                key = (step.tool, step.rationale)
                if key not in seen_steps:
                    playbook.append({"tool": step.tool, "rationale": step.rationale})
                    seen_steps.add(key)

        strategy = []
        for node in self.strategies.values():
            if any(qid in selected for qid in node.source_queries):
                strategy.append({"text": node.text, "source_strategy_id": node.sid})

        tool_tips: list[dict[str, str]] = []
        for tool_name in relevant_tools:
            if reads >= read_budget:
                break
            reads += 1  # ReadTool
            node = self.tools[tool_name]
            entries = node.affordances + node.boundaries + node.co_usage
            ranked_entries = sorted(
                entries,
                key=lambda e: (len(task_terms & set(tokens(e.text))), len(e.source_queries)),
                reverse=True,
            )
            for entry in ranked_entries[:2]:
                tool_tips.append({"tool": tool_name, "tip": entry.text})

        return {
            "seed_candidates": [
                {"qid": qid, "sim": round(sim, 4), "summary": self.traces[qid].summary}
                for sim, qid in seeds
            ],
            "playbook": playbook,
            "strategy": strategy,
            "tool_tips": tool_tips,
        }

    def stats(self) -> dict[str, int]:
        return {
            "tools": len(self.tools),
            "traces": len(self.traces),
            "strategies": len(self.strategies),
            "trace_edges": sum(len(t.neighbors) for t in self.traces.values()) // 2,
        }

    def tool_details(self, name: str) -> dict[str, Any] | None:
        node = self.tools.get(name)
        return asdict(node) if node else None

    def suggest_probes(self, name: str) -> dict[str, list[dict[str, str]]]:
        """Propose cheap outward/inward probes from current capability coverage."""
        node = self.tools.get(name)
        if not node:
            raise ValueError(f"unknown tool: {name}")
        peers = _unique(peer for entry in node.co_usage for peer in entry.related_tools)
        composition = (
            f"Verify a realistic workflow that combines {name} with {peers[0]}."
            if peers
            else f"Verify a representative successful use of {name} not covered by existing traces."
        )
        return {
            "boundary": [
                {
                    "direction": "outward",
                    "task": f"Test {name} with empty, malformed, and unusually large inputs; verify graceful failure.",
                }
            ],
            "affordance": [{"direction": "inward", "task": composition}],
        }

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "tools": {key: asdict(value) for key, value in self.tools.items()},
            "traces": {key: asdict(value) for key, value in self.traces.items()},
            "strategies": {key: asdict(value) for key, value in self.strategies.items()},
        }
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.tools = {
            key: ToolNode(
                name=value["name"],
                description=value.get("description", ""),
                affordances=[EvidenceEntry(**entry) for entry in value.get("affordances", [])],
                boundaries=[EvidenceEntry(**entry) for entry in value.get("boundaries", [])],
                co_usage=[EvidenceEntry(**entry) for entry in value.get("co_usage", [])],
                traces=value.get("traces", []),
            )
            for key, value in payload.get("tools", {}).items()
        }
        self.traces = {
            key: TraceNode(
                qid=value["qid"],
                summary=value["summary"],
                agent_neutral_trace=[ExecutionStep(**step) for step in value["agent_neutral_trace"]],
                task_level_tips=value.get("task_level_tips", []),
                tools=value.get("tools", []),
                neighbors=value.get("neighbors", []),
            )
            for key, value in payload.get("traces", {}).items()
        }
        self.strategies = {
            key: StrategyNode(**value) for key, value in payload.get("strategies", {}).items()
        }

    @staticmethod
    def _merge_entry(
        entries: list[EvidenceEntry], text: str, task_id: str, related_tools: list[str] | None = None
    ) -> None:
        for entry in entries:
            if entry.text == text:
                if task_id not in entry.source_queries:
                    entry.source_queries.append(task_id)
                return
        entries.append(EvidenceEntry(text, [task_id], related_tools or []))

    @staticmethod
    def _distill_tips(rollouts: list[Rollout]) -> list[str]:
        tips: list[str] = []
        if any(r.resolved for r in rollouts):
            tips.append("Verify the final tool result against the task requirement.")
        for rollout in rollouts:
            if not rollout.resolved and rollout.observation:
                tips.append(f"Validate inputs first: {_generic_rationale(rollout.observation)}")
        return _unique(tips)

    def _rebuild_trace_edges(self) -> None:
        for node in self.traces.values():
            node.neighbors = []
        ids = list(self.traces)
        for index, left_id in enumerate(ids):
            scores = []
            for right_id in ids[index + 1 :]:
                left, right = self.traces[left_id], self.traces[right_id]
                semantic = cosine_text(left.summary, right.summary)
                shared_tools = len(set(left.tools) & set(right.tools))
                score = semantic + (0.25 if shared_tools else 0.0)
                if score > 0.2:
                    scores.append((score, right_id))
            for _, right_id in sorted(scores, reverse=True)[:3]:
                self.traces[left_id].neighbors.append(right_id)
                self.traces[right_id].neighbors.append(left_id)

    def _induce_strategies(self) -> None:
        sequences: dict[tuple[str, ...], list[str]] = {}
        for qid, trace in self.traces.items():
            sequence = tuple(step.tool for step in trace.agent_neutral_trace)
            if len(sequence) > 1:
                sequences.setdefault(sequence, []).append(qid)
        self.strategies = {}
        for sequence, qids in sequences.items():
            if len(qids) >= 2:
                sid = f"strategy_{len(self.strategies) + 1}"
                self.strategies[sid] = StrategyNode(
                    sid=sid,
                    text=f"Apply {' then '.join(sequence)} and verify the composed result.",
                    source_queries=qids,
                    tool_sequence=list(sequence),
                )
