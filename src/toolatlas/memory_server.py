from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from .memory import ToolMemory
from .models import ExecutionStep, Rollout, ToolSpec

SCHEMA_VERSION = "2"


def _rollout(task_id: str, summary: str, item: dict[str, Any]) -> Rollout:
    return Rollout(
        task_id=task_id,
        summary=summary,
        steps=[ExecutionStep(**step) for step in item["steps"]],
        resolved=item["resolved"],
        observation=item.get("observation", ""),
        execution_id=item.get("execution_id", ""),
        verifier_type=item.get("verifier_type", "external"),
        **({"verified_at": item["verified_at"]} if item.get("verified_at") else {}),
    )


def create_memory_server(path: str | Path | None = None) -> MCPServer:
    store = ToolMemory(path)
    server = MCPServer("toolatlas-memory")

    @server.tool()
    def register_tools(tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Register provider tool specifications before learning executions."""
        store.register_tools(
            ToolSpec(
                name=item["name"],
                description=item.get("description", ""),
                input_schema=item.get("input_schema", {}),
                version=item.get("version", "unknown"),
                provider=item.get("provider", "local"),
            )
            for item in tools
        )
        return store.stats()

    @server.tool()
    def remember_execution(
        task_id: str,
        summary: str,
        steps: list[dict[str, str]],
        resolved: bool,
        observation: str = "",
        verifier_type: str = "external",
    ) -> dict[str, Any]:
        """Induce graph memory from one execution-verified agent rollout."""
        rollout = _rollout(
            task_id, summary,
            {
                "steps": steps,
                "resolved": resolved,
                "observation": observation,
                "verifier_type": verifier_type,
            },
        )
        trace = store.induce(task_id, summary, [rollout])
        return {
            "schema_version": SCHEMA_VERSION,
            "qid": trace.qid,
            "tools": trace.tools,
            "confidence": trace.confidence,
            "stats": store.stats(),
        }

    @server.tool()
    def remember_rollouts(
        task_id: str, summary: str, rollouts: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Induce one trace from multiple independently verified attempts."""
        trace = store.induce(
            task_id,
            summary,
            [_rollout(task_id, summary, item) for item in rollouts],
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "qid": trace.qid,
            "confidence": trace.confidence,
            "success_count": trace.success_count,
            "failure_count": trace.failure_count,
            "execution_ids": trace.source_executions,
        }

    @server.tool()
    def get_guidance(
        task: str,
        top_k: int = 3,
        read_budget: int = 8,
        max_age_days: int = 30,
    ) -> dict[str, Any]:
        """Traverse tool-side memory and return task-conditioned tool guidance."""
        return store.guide(
            task,
            top_k=top_k,
            read_budget=read_budget,
            max_age_days=max_age_days,
        )

    @server.tool()
    def inspect_tool(tool_name: str) -> dict[str, Any]:
        """Inspect learned affordances, boundaries, co-usage, and linked traces."""
        return store.tool_details(tool_name) or {"error": f"unknown tool: {tool_name}"}

    @server.tool()
    def suggest_probes(tool_name: str) -> dict[str, Any]:
        """Suggest outward boundary and inward affordance probes from graph coverage."""
        return store.suggest_probes(tool_name)

    @server.tool()
    def refresh_status(max_age_days: int = 30) -> dict[str, Any]:
        """List stale, invalid, or quarantined memories requiring review."""
        return store.refresh_status(max_age_days=max_age_days)

    @server.tool()
    def reverify_trace(
        task_id: str,
        resolved: bool,
        verifier_type: str,
        observation: str = "",
    ) -> dict[str, Any]:
        """Record external re-verification against the current tool schemas."""
        trace = store.reverify_trace(task_id, resolved, verifier_type, observation)
        return {
            "qid": trace.qid,
            "status": trace.status,
            "confidence": trace.confidence,
            "last_verified_at": trace.last_verified_at,
        }

    @server.tool()
    def set_trace_status(task_id: str, status: str, reason: str) -> dict[str, Any]:
        """Govern a trace by setting it stale, invalid, or quarantined with a reason."""
        trace = store.set_trace_status(task_id, status, reason)
        return {"qid": trace.qid, "status": trace.status, "reason": trace.status_reason}

    @server.tool()
    def memory_stats() -> dict[str, int]:
        """Return counts for the three graph layers."""
        return store.stats()

    @server.resource("toolatlas://graph")
    def graph_snapshot() -> str:
        """Return a JSON summary of the persistent tool memory graph."""
        return json.dumps(store.stats())

    return server


memory_path = Path(os.environ.get("TOOLATLAS_MEMORY_PATH", ".toolatlas/memory.db"))
mcp = create_memory_server(memory_path)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
