from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ExecutionStep:
    tool: str
    rationale: str


@dataclass(slots=True)
class Rollout:
    task_id: str
    summary: str
    steps: list[ExecutionStep]
    resolved: bool
    observation: str = ""


@dataclass(slots=True)
class EvidenceEntry:
    text: str
    source_queries: list[str] = field(default_factory=list)
    related_tools: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TraceNode:
    qid: str
    summary: str
    agent_neutral_trace: list[ExecutionStep]
    task_level_tips: list[str]
    tools: list[str]
    neighbors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ToolNode:
    name: str
    description: str = ""
    affordances: list[EvidenceEntry] = field(default_factory=list)
    boundaries: list[EvidenceEntry] = field(default_factory=list)
    co_usage: list[EvidenceEntry] = field(default_factory=list)
    traces: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StrategyNode:
    sid: str
    text: str
    source_queries: list[str]
    tool_sequence: list[str]


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)

