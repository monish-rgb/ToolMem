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


READ_ONLY_TOOLS = frozenset({
    "get_guidance",
    "inspect_tool",
    "suggest_probes",
    "refresh_status",
    "reverification_due",
    "memory_stats",
})
"""Tools exposed by the read-only evaluation profile.

Excluded from evaluation: ingestion (register_tools, remember_execution,
remember_rollouts), re-verification (reverify_trace), and governance or
status-change mutation (set_trace_status). refresh_status only lists review
candidates and is safe to expose.
"""


def create_memory_server(
    path: str | Path | None = None, read_only: bool = False
) -> MCPServer:
    store = ToolMemory(path, read_only=read_only)
    server = MCPServer("toolatlas-memory")

    if not read_only:
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
        def induce_reflected(
            task_id: str,
            summary: str,
            rollouts: list[dict[str, Any]],
            task_level_tips: list[str] | None = None,
            step_rationales: list[str] | None = None,
            known_tools: list[str] | None = None,
        ) -> dict[str, Any]:
            """Induce one trace with LLM-reflected rationales and tips.

            Write path only: excluded from the read-only evaluation profile.
            Reflected content is re-sanitized and grounded by the store;
            ungrounded entries are rejected, never stored.
            """
            trace = store.induce(
                task_id,
                summary,
                [_rollout(task_id, summary, item) for item in rollouts],
                task_level_tips=task_level_tips,
                step_rationales=step_rationales,
                induction="llm-reflected",
                known_tools=known_tools,
            )
            return {
                "schema_version": SCHEMA_VERSION,
                "qid": trace.qid,
                "confidence": trace.confidence,
                "induction": trace.induction,
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
        token_budget: int = 384,
        embed_mode: str = "lexical",
    ) -> dict[str, Any]:
        """Traverse tool-side memory and return task-conditioned tool guidance.

        ``token_budget`` caps the rendered guidance size so online input
        cost stays bounded (plan Phase 2). Empty retrieval stays empty:
        no generic advice is ever invented. ``embed_mode="embedding"`` ranks
        seeds with the deterministic credential-free hash embedder blended
        with lexical scores; the mode is recorded in the traversal audit.
        """
        embedder: Any = None
        if embed_mode == "embedding":
            from .embeddings import HashEmbedder
            embedder = HashEmbedder()
        return store.guide(
            task,
            top_k=top_k,
            read_budget=read_budget,
            max_age_days=max_age_days,
            token_budget=token_budget,
            embed_mode=embed_mode,
            embedder=embedder,
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

    if not read_only:
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
    def reverification_due(max_age_days: int = 30) -> dict[str, Any]:
        """List traces needing a re-verification run, most overdue first."""
        return store.reverification_due(max_age_days=max_age_days)

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
memory_read_only = os.environ.get("TOOLATLAS_READ_ONLY", "").strip().lower() in (
    "1", "true", "yes", "on",
)
mcp = create_memory_server(memory_path, read_only=memory_read_only)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
