"""Offline capability exploration for ToolAtlas memory (paper §3.2, deterministic).

Seed tasks cover only a small region of tool capability. The explorer pushes
outward with boundary probes (empty / mistyped / oversized / missing inputs)
and fills inward with minimal affordance probes, executes them against a
caller-provided client, and ingests only verified outcomes. Verified
rejections become boundary cautions; verified successes become affordances —
so inference-time agents skip the trial-and-error calls exploration already
paid for once, offline.

Safety: exploration only runs when explicitly invoked, only against the tools
named in ``allow_tools``, never against destructive tools unless each one is
explicitly opted in, and a dry run lists every probe without executing
anything. Probes run against the caller's client root (a sandbox in
production); this module never chooses a root itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .memory import _generic_rationale
from .models import ExecutionStep, Rollout, ToolSpec

DESTRUCTIVE_HINTS = (
    "delete",
    "drop",
    "remove",
    "destroy",
    "revoke",
    "kill",
    "shutdown",
    "terminate",
    "uninstall",
    "reset",
    "truncate",
    "format",
)

MAX_PROBE_FIELDS = 3
OVERSIZED_LENGTH = 20000


@dataclass
class Probe:
    """One executable probe against a single tool."""

    name: str
    tool: str
    args: dict[str, Any]
    direction: str  # "outward" (boundary) or "inward" (affordance)
    expect_success: bool


@dataclass
class ExplorationReport:
    """Auditable outcome of one exploration run (JSON-serializable)."""

    tools_explored: list[str] = field(default_factory=list)
    probes_executed: int = 0
    affordances_verified: int = 0
    boundaries_verified: int = 0
    unexpected: list[dict[str, str]] = field(default_factory=list)
    traces_ingested: list[str] = field(default_factory=list)


def is_destructive(tool_name: str) -> bool:
    lowered = tool_name.lower()
    return any(hint in lowered for hint in DESTRUCTIVE_HINTS)


def _schema_properties(spec: ToolSpec) -> tuple[dict[str, Any], list[str]]:
    schema = spec.input_schema if isinstance(spec.input_schema, dict) else {}
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required", [])
    if not isinstance(required, list):
        required = []
    return properties, [str(name) for name in required]


def _benign_value(prop: Any) -> Any:
    if not isinstance(prop, dict):
        return "probe"
    kind = str(prop.get("type", "string")).lower()
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return False
    if kind == "array":
        return []
    if kind == "object":
        return {}
    return "probe"


def affordance_probes(spec: ToolSpec) -> list[Probe]:
    """Minimal valid calls: one per tool, built from required schema fields."""
    properties, required = _schema_properties(spec)
    args = {name: _benign_value(properties.get(name, {})) for name in required}
    return [Probe("minimal-valid-call", spec.name, args, "inward", True)]


def boundary_probes(spec: ToolSpec) -> list[Probe]:
    """Outward probes over required fields: missing, mistyped, empty, oversized."""
    properties, required = _schema_properties(spec)
    probes: list[Probe] = []
    targets = required[:MAX_PROBE_FIELDS] or list(properties)[:MAX_PROBE_FIELDS]
    for name in targets:
        prop = properties.get(name, {})
        kind = str(prop.get("type", "string")).lower() if isinstance(prop, dict) else "string"
        base = {other: _benign_value(properties.get(other, {})) for other in required if other != name}
        probes.append(Probe(f"missing-{name}", spec.name, dict(base), "outward", False))
        mistyped = dict(base)
        mistyped[name] = [1, 2] if kind in ("string", "boolean") else {"unexpected": "object"}
        probes.append(Probe(f"mistyped-{name}", spec.name, mistyped, "outward", False))
        if kind == "string":
            empty = dict(base)
            empty[name] = ""
            probes.append(Probe(f"empty-{name}", spec.name, empty, "outward", False))
            oversized = dict(base)
            oversized[name] = "x" * OVERSIZED_LENGTH
            probes.append(Probe(f"oversized-{name}", spec.name, oversized, "outward", False))
    return probes


def plan_exploration(specs: list[ToolSpec], allow_tools: list[str]) -> list[Probe]:
    """List every probe that would run (dry run — executes nothing)."""
    allowed = set(allow_tools)
    probes: list[Probe] = []
    for spec in specs:
        if spec.name not in allowed:
            continue
        probes.extend(affordance_probes(spec))
        probes.extend(boundary_probes(spec))
    return probes


ProbeOutcome = tuple[bool, str]
Prober = Callable[[str, dict[str, Any]], Awaitable[ProbeOutcome]]


async def run_exploration(
    memory: Any,
    specs: list[ToolSpec],
    prober: Prober,
    allow_tools: list[str],
    task_prefix: str = "explore",
    allow_destructive: bool = False,
) -> ExplorationReport:
    """Execute probes and ingest verified outcomes into memory.

    Only tools in ``allow_tools`` run; destructive tools additionally require
    ``allow_destructive``. A probe whose outcome contradicts its expectation
    is recorded in ``unexpected`` and never ingested.
    """
    report = ExplorationReport()
    allowed = set(allow_tools)
    for spec in specs:
        if spec.name not in allowed:
            continue
        if is_destructive(spec.name) and not allow_destructive:
            report.unexpected.append({"tool": spec.name, "reason": "destructive-refused"})
            continue
        memory.register_tools([spec])
        probes = affordance_probes(spec) + boundary_probes(spec)
        rollouts: list[Rollout] = []
        for probe in probes:
            try:
                ok, summary = await prober(probe.tool, probe.args)
            except Exception as exc:  # noqa: BLE001 — probe failure is evidence
                ok, summary = False, f"{type(exc).__name__}: {exc}"
            clean_summary = _generic_rationale(summary or "no result detail")
            report.probes_executed += 1
            if ok == probe.expect_success:
                if probe.direction == "inward":
                    report.affordances_verified += 1
                    rollouts.append(Rollout(
                        task_id=f"{task_prefix}-{spec.name}",
                        summary=f"Verify a representative use of {spec.name}",
                        steps=[ExecutionStep(spec.name, f"apply {spec.name} to valid inputs")],
                        resolved=True,
                        observation=f"verified: {clean_summary}",
                        verifier_type="explorer",
                    ))
                else:
                    report.boundaries_verified += 1
                    rollouts.append(Rollout(
                        task_id=f"{task_prefix}-{spec.name}-boundary",
                        summary=f"Probe rejection behavior of {spec.name}",
                        steps=[ExecutionStep(spec.name, f"probe rejection with {probe.name}")],
                        resolved=False,
                        observation=f"{spec.name} rejects {probe.name}: {clean_summary}",
                        verifier_type="explorer",
                    ))
            else:
                report.unexpected.append({
                    "tool": spec.name,
                    "probe": probe.name,
                    "reason": f"expected success={probe.expect_success} but observed ok={ok}",
                })
        grouped: dict[str, list[Rollout]] = {}
        for rollout in rollouts:
            grouped.setdefault(rollout.task_id, []).append(rollout)
        for task_id, batch in grouped.items():
            memory.induce(task_id, batch[0].summary, batch)
            report.traces_ingested.append(task_id)
        report.tools_explored.append(spec.name)
    return report


async def explore_with_client(
    memory: Any,
    client: Any,
    allow_tools: list[str],
    task_prefix: str = "explore",
    allow_destructive: bool = False,
) -> ExplorationReport:
    """Adapter: run exploration through a live MCP client (sandbox root)."""
    listed = await client.list_tools()
    items = getattr(listed, "tools", listed) or []

    def _field(item: Any, *names: str, default: Any = "") -> Any:
        if isinstance(item, dict):
            for name in names:
                if name in item:
                    return item[name]
            return default
        for name in names:
            value = getattr(item, name, None)
            if value is not None:
                return value
        return default

    specs = [
        ToolSpec(
            name=str(_field(item, "name")),
            description=str(_field(item, "description", default="")),
            input_schema=_field(item, "inputSchema", "input_schema", default={}) or {},
        )
        for item in items
    ]

    async def _prober(tool: str, args: dict[str, Any]) -> ProbeOutcome:
        try:
            result = await client.call_tool(tool, args)
        except Exception as exc:  # noqa: BLE001 — probe failure is evidence
            return False, f"{type(exc).__name__}: {exc}"
        if bool(getattr(result, "is_error", False)):
            return False, str(getattr(result, "content", "tool reported an error"))
        structured = getattr(result, "structured_content", None)
        return True, str(structured if structured is not None else getattr(result, "content", "ok"))

    return await run_exploration(
        memory, specs, _prober, allow_tools, task_prefix, allow_destructive
    )


def exploration_summary(report: ExplorationReport) -> dict[str, Any]:
    """JSON-serializable summary with amortized-cost framing."""
    return {
        "tools_explored": report.tools_explored,
        "probes_executed": report.probes_executed,
        "affordances_verified": report.affordances_verified,
        "boundaries_verified": report.boundaries_verified,
        "unexpected": report.unexpected,
        "traces_ingested": report.traces_ingested,
        "note": "offline construction cost; amortize over evaluation runs",
    }
