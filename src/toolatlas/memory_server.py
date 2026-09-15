from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from .memory import ToolMemory
from .models import ExecutionStep, Rollout, ToolSpec


def create_memory_server(path: str | Path | None = None) -> MCPServer:
    store = ToolMemory(path)
    server = MCPServer("toolatlas-memory")

    @server.tool()
    def register_tools(tools: list[dict[str, Any]]) -> dict[str, int]:
        """Register provider tool specifications before learning executions."""
        store.register_tools(
            ToolSpec(
                name=item["name"],
                description=item.get("description", ""),
                input_schema=item.get("input_schema", {}),
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
    ) -> dict[str, Any]:
        """Induce graph memory from one execution-verified agent rollout."""
        rollout = Rollout(
            task_id=task_id,
            summary=summary,
            steps=[ExecutionStep(**step) for step in steps],
            resolved=resolved,
            observation=observation,
        )
        trace = store.induce(task_id, summary, [rollout])
        return {"qid": trace.qid, "tools": trace.tools, "stats": store.stats()}

    @server.tool()
    def get_guidance(task: str, top_k: int = 3, read_budget: int = 8) -> dict[str, Any]:
        """Traverse tool-side memory and return task-conditioned tool guidance."""
        return store.guide(task, top_k=top_k, read_budget=read_budget)

    @server.tool()
    def inspect_tool(tool_name: str) -> dict[str, Any]:
        """Inspect learned affordances, boundaries, co-usage, and linked traces."""
        return store.tool_details(tool_name) or {"error": f"unknown tool: {tool_name}"}

    @server.tool()
    def suggest_probes(tool_name: str) -> dict[str, list[dict[str, str]]]:
        """Suggest outward boundary and inward affordance probes from graph coverage."""
        return store.suggest_probes(tool_name)

    @server.tool()
    def memory_stats() -> dict[str, int]:
        """Return counts for the three graph layers."""
        return store.stats()

    @server.resource("toolatlas://graph")
    def graph_snapshot() -> str:
        """Return a JSON summary of the persistent tool memory graph."""
        return json.dumps(store.stats())

    return server


memory_path = Path(os.environ.get("TOOLATLAS_MEMORY_PATH", ".toolatlas/memory.json"))
mcp = create_memory_server(memory_path)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
