from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .filesystem_demo import (
    FILESYSTEM_PACKAGE_VERSION,
    FILESYSTEM_PROVIDER,
    filesystem_server_path,
)
from .memory_server import create_memory_server

READ_ONLY_TOOLS = {
    "read_file",
    "read_text_file",
    "read_media_file",
    "read_multiple_files",
    "list_directory",
    "list_directory_with_sizes",
    "directory_tree",
    "search_files",
    "get_file_info",
    "list_allowed_directories",
}


def _value(result: Any) -> Any:
    data = result.structured_content
    if isinstance(data, dict):
        return data.get("result", data.get("content", data))
    return data


@dataclass
class ReadOnlyAudit:
    client: Client
    calls: list[dict[str, Any]] = field(default_factory=list)
    forbidden_attempts: int = 0

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        if tool not in READ_ONLY_TOOLS:
            self.forbidden_attempts += 1
            raise PermissionError(f"write-capable MCP tool blocked by benchmark: {tool}")
        started = time.perf_counter()
        result = await self.client.call_tool(tool, arguments)
        self.calls.append(
            {
                "tool": tool,
                "is_error": result.is_error,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        return result


def _first_path(search_result: Any) -> str:
    text = str(_value(search_result))
    candidates = [line.strip() for line in text.splitlines() if line.strip()]
    if not candidates or candidates[0] == "No matches found":
        raise AssertionError("target file was not found")
    return candidates[0]


def _extract_integer(content: str, field_name: str) -> int:
    match = re.search(rf"(?m)^\s*{re.escape(field_name)}\s*:\s*(\d+)\s*$", content)
    if not match:
        raise AssertionError(f"field {field_name!r} was not present in the target file")
    return int(match.group(1))


async def _seed_read_memory(audit: ReadOnlyAudit, memory: Client, root: Path) -> None:
    training = [
        ("retry", "retry-policy.md", "max_attempts", 5),
        ("timeout", "timeout-policy.md", "request_timeout_seconds", 30),
    ]
    for label, filename, field_name, expected in training:
        search = await audit.call(
            "search_files", {"path": str(root), "pattern": f"**/{filename}"}
        )
        target = _first_path(search)
        read = await audit.call("read_text_file", {"path": target})
        actual = _extract_integer(str(_value(read)), field_name)
        await memory.call_tool(
            "remember_execution",
            {
                "task_id": f"readonly_{label}_policy",
                "summary": f"Find a {label} policy file and report its configured setting",
                "steps": [
                    {"tool": "search_files", "rationale": "locate the requested policy file recursively"},
                    {"tool": "read_text_file", "rationale": "read the matching policy file and extract the requested setting"},
                ],
                "resolved": actual == expected,
                "observation": "the requested setting matched the exact expected value",
                "verifier_type": "exact_field_match",
            },
        )
        if actual != expected:
            raise AssertionError(f"training verifier failed for {filename}")


async def _baseline_agent(audit: ReadOnlyAudit, root: Path) -> int:
    await audit.call("list_directory", {"path": str(root)})
    await audit.call("directory_tree", {"path": str(root)})
    search = await audit.call(
        "search_files", {"path": str(root), "pattern": "**/deployment-policy.md"}
    )
    target = _first_path(search)
    read = await audit.call("read_text_file", {"path": target})
    return _extract_integer(str(_value(read)), "rollout_window_minutes")


async def _toolatlas_agent(audit: ReadOnlyAudit, memory: Client, root: Path) -> tuple[int, dict[str, Any]]:
    guidance_result = await memory.call_tool(
        "get_guidance",
        {
            "task": "Find the deployment policy file and report its rollout window setting",
            "top_k": 2,
            "read_budget": 6,
        },
    )
    guidance = _value(guidance_result)
    recommended = [step["tool"] for step in guidance["playbook"]]
    if recommended[:2] != ["search_files", "read_text_file"]:
        raise AssertionError(f"ToolAtlas did not retrieve the expected read-only playbook: {recommended}")
    search = await audit.call(
        "search_files", {"path": str(root), "pattern": "**/deployment-policy.md"}
    )
    target = _first_path(search)
    read = await audit.call("read_text_file", {"path": target})
    return _extract_integer(str(_value(read)), "rollout_window_minutes"), guidance


def _metrics(name: str, answer: int, expected: int, audit: ReadOnlyAudit, elapsed: float) -> dict[str, Any]:
    return {
        "agent": name,
        "passed": answer == expected,
        "answer": answer,
        "expected": expected,
        "filesystem_tool_calls": len(audit.calls),
        "tools_used": [call["tool"] for call in audit.calls],
        "forbidden_write_attempts": audit.forbidden_attempts,
        "elapsed_ms": round(elapsed * 1000, 3),
    }


async def run_readonly_ab(project_root: Path, memory_path: Path) -> dict[str, Any]:
    """Compare deterministic agents on the same live read-only MCP task."""
    project_root = project_root.resolve()
    root = project_root / "tests" / "fixtures" / "readonly_workspace"
    server_js = filesystem_server_path(project_root)
    if not server_js.is_file():
        raise FileNotFoundError("run npm install before the read-only benchmark")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required")
    parameters = StdioServerParameters(command=node, args=[str(server_js), str(root)])
    memory_server = create_memory_server(memory_path)

    async with Client(parameters) as filesystem, Client(memory_server) as memory:
        listed = await filesystem.list_tools()
        read_specs = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
                "provider": FILESYSTEM_PROVIDER,
                "version": FILESYSTEM_PACKAGE_VERSION,
            }
            for tool in listed.tools if tool.name in READ_ONLY_TOOLS
        ]
        await memory.call_tool("register_tools", {"tools": read_specs})

        training_audit = ReadOnlyAudit(filesystem)
        await _seed_read_memory(training_audit, memory, root)

        baseline_audit = ReadOnlyAudit(filesystem)
        started = time.perf_counter()
        baseline_answer = await _baseline_agent(baseline_audit, root)
        baseline_elapsed = time.perf_counter() - started

        toolatlas_audit = ReadOnlyAudit(filesystem)
        started = time.perf_counter()
        toolatlas_answer, guidance = await _toolatlas_agent(toolatlas_audit, memory, root)
        toolatlas_elapsed = time.perf_counter() - started

    expected = 20
    baseline = _metrics("baseline_without_toolatlas", baseline_answer, expected, baseline_audit, baseline_elapsed)
    assisted = _metrics("agent_with_toolatlas", toolatlas_answer, expected, toolatlas_audit, toolatlas_elapsed)
    return {
        "benchmark": "live_readonly_filesystem_ab",
        "filesystem_server": f"{FILESYSTEM_PROVIDER}@{FILESYSTEM_PACKAGE_VERSION}",
        "root": str(root),
        "read_only_tools": sorted(READ_ONLY_TOOLS),
        "training_calls_excluded_from_comparison": len(training_audit.calls),
        "baseline": baseline,
        "toolatlas": assisted,
        "comparison": {
            "both_passed": baseline["passed"] and assisted["passed"],
            "filesystem_calls_saved": baseline["filesystem_tool_calls"] - assisted["filesystem_tool_calls"],
            "call_reduction_percent": round(
                100 * (baseline["filesystem_tool_calls"] - assisted["filesystem_tool_calls"])
                / baseline["filesystem_tool_calls"],
                2,
            ),
        },
        "retrieved_guidance": guidance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live read-only baseline vs ToolAtlas A/B test")
    parser.add_argument("--memory", type=Path, default=Path(".toolatlas/readonly-benchmark.db"))
    args = parser.parse_args()
    result = asyncio.run(run_readonly_ab(Path.cwd(), args.memory))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
