from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    version: str = "unknown"
    provider: str = "local"


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
    execution_id: str = ""
    verifier_type: str = "external"
    verified_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class EvidenceEntry:
    text: str
    source_queries: list[str] = field(default_factory=list)
    related_tools: list[str] = field(default_factory=list)
    source_executions: list[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.5
    status: str = "active"
    last_verified_at: str = field(default_factory=utc_now)
    tool_fingerprint: str = ""


@dataclass(slots=True)
class TraceNode:
    qid: str
    summary: str
    agent_neutral_trace: list[ExecutionStep]
    task_level_tips: list[str]
    tools: list[str]
    neighbors: list[str] = field(default_factory=list)
    source_executions: list[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    confidence: float = 0.5
    status: str = "active"
    status_reason: str = ""
    tool_fingerprints: dict[str, str] = field(default_factory=dict)
    last_verified_at: str = field(default_factory=utc_now)


@dataclass(slots=True)
class ToolNode:
    name: str
    description: str = ""
    affordances: list[EvidenceEntry] = field(default_factory=list)
    boundaries: list[EvidenceEntry] = field(default_factory=list)
    co_usage: list[EvidenceEntry] = field(default_factory=list)
    traces: list[str] = field(default_factory=list)
    input_schema: dict[str, Any] = field(default_factory=dict)
    version: str = "unknown"
    provider: str = "local"
    schema_fingerprint: str = ""
    status: str = "active"


@dataclass(slots=True)
class StrategyNode:
    sid: str
    text: str
    source_queries: list[str]
    tool_sequence: list[str]
    confidence: float = 0.5
    status: str = "active"


@dataclass(slots=True)
class ExecutionRecord:
    execution_id: str
    task_id: str
    resolved: bool
    verifier_type: str
    verified_at: str
    observation: str = ""


def to_dict(value: Any) -> dict[str, Any]:
    return asdict(value)
