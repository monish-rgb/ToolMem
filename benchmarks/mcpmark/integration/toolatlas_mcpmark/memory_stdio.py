"""Stdio ToolAtlas memory sessions for the MCPMark integration.

Memory always runs as a real stdio MCP subprocess (never in-process) in
reportable runs. The child receives only its database path (plus the
read-only flag); model or service credentials are never inherited. The
session factory is injectable so unit tests can substitute a fake client.
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional

CREDENTIAL_ENV_KEYS = frozenset({
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "MOONSHOT_API_KEY",
    "OPENROUTER_API_KEY",
    "DASHSCOPE_API_KEY",
    "GROK_API_KEY",
    "NVIDIA_API_KEY",
    "LLM_API_KEY",
    "GITHUB_TOKEN",
    "GITHUB_TOKENS",
    "GITHUB_PERSONAL_ACCESS_TOKEN",
    "NOTION_API_KEY",
    "SOURCE_NOTION_API_KEY",
    "EVAL_NOTION_API_KEY",
    "POSTGRES_PASSWORD",
})


def memory_env(db_path: str | Path, read_only: bool = False) -> dict[str, str]:
    """Minimal child environment: database path and mode flag only."""
    env = {
        "TOOLATLAS_MEMORY_PATH": str(Path(db_path)),
        "PATH": os.environ.get("PATH", ""),
    }
    if read_only:
        env["TOOLATLAS_READ_ONLY"] = "1"
    for key in CREDENTIAL_ENV_KEYS:
        assert key not in env, f"credential leak into memory env: {key}"
    return env


def _result_value(result: Any) -> Any:
    data = getattr(result, "structured_content", None)
    if isinstance(data, dict):
        return data.get("result", data)
    return data


class StdioMemorySession:
    """One memory subprocess + client pair."""

    def __init__(self, client: Any, recorder: Any = None) -> None:
        self._client = client
        self._recorder = recorder
        self.calls: list[dict[str, Any]] = []

    async def call(self, tool: str, args: dict, kind: str = "memory") -> Any:
        event = None
        if self._recorder is not None:
            event = self._recorder.begin_call("toolatlas-memory", tool, args, kind)
        try:
            result = await self._client.call_tool(tool, args)
        except Exception as exc:
            if event is not None:
                self._recorder.end_call(event, False, error=f"{type(exc).__name__}: {exc}")
            raise
        value = _result_value(result)
        if event is not None:
            self._recorder.end_call(event, True, result=value)
        self.calls.append({"tool": tool, "ok": True})
        return value

    async def guidance(self, task: str, top_k: int, read_budget: int) -> dict:
        value = await self.call(
            "get_guidance", {"task": task, "top_k": top_k, "read_budget": read_budget}
        )
        return value if isinstance(value, dict) else {}

    async def learn(
        self,
        task_id: str,
        summary: str,
        tool_specs: list[dict],
        steps: list[dict],
        resolved: bool,
        observation: str,
    ) -> dict:
        if tool_specs:
            await self.call("register_tools", {"tools": tool_specs})
        value = await self.call(
            "remember_execution",
            {
                "task_id": task_id,
                "summary": summary,
                "steps": steps,
                "resolved": resolved,
                "observation": observation,
                "verifier_type": "mcpmark-verify-py",
            },
        )
        return value if isinstance(value, dict) else {}


@asynccontextmanager
async def open_stdio_memory(
    db_path: str | Path,
    read_only: bool = False,
    package_dir: Optional[str | Path] = None,
    python_exe: Optional[str] = None,
    recorder: Any = None,
) -> AsyncIterator[StdioMemorySession]:
    """Launch the ToolAtlas memory server over stdio with a sanitized env."""
    from mcp import Client, StdioServerParameters  # lazy: keeps module import-light

    env = memory_env(db_path, read_only)
    if package_dir is not None:
        env["PYTHONPATH"] = str(Path(package_dir)) + os.pathsep + env.get("PYTHONPATH", "")
    params = StdioServerParameters(
        command=python_exe or sys.executable,
        args=["-m", "toolatlas.memory_server"],
        env=env,
    )
    async with Client(params) as client:
        yield StdioMemorySession(client, recorder)


SessionFactory = Callable[..., Any]
"""Injectable session opener: open(db_path, read_only, recorder) -> async CM."""
