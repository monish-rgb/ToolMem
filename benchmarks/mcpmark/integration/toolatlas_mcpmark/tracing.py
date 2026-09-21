"""MCP-boundary tracing for the ToolAtlas MCPMark integration.

The wrapper instruments the MCP server object (not model-specific response
parsing): every attempted call is recorded *before* awaiting the provider and
updated with its result or exception afterwards, so raised calls are counted
with their error instead of silently omitted.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from .sanitize import fingerprint_tool, sanitize_args, summarize_result


class TraceRecorder:
    """Ordered, JSON-serializable MCP event log for one attempt."""

    def __init__(
        self,
        task_id: str = "",
        arm: str = "",
        attempt: int = 0,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self.task_id = task_id
        self.arm = arm
        self.attempt = attempt
        self._clock = clock or time.monotonic
        self._seq = 0
        self.events: list[dict[str, Any]] = []
        self.tool_specs: list[dict[str, Any]] = []

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def record_discovery(self, tool_count: int, tool_fingerprints: dict) -> dict:
        event = {
            "seq": self._next_seq(),
            "kind": "discovery",
            "task_id": self.task_id,
            "arm": self.arm,
            "attempt": self.attempt,
            "tool_count": tool_count,
            "tool_fingerprints": tool_fingerprints,
            "t_start": self._clock(),
        }
        event["t_end"] = event["t_start"]
        self.events.append(event)
        return event

    def begin_call(self, provider: str, tool: str, args: Any, kind: str) -> dict:
        event = {
            "seq": self._next_seq(),
            "kind": kind,  # "provider" or "memory"
            "provider": provider,
            "tool": tool,
            "args": sanitize_args(args),
            "task_id": self.task_id,
            "arm": self.arm,
            "attempt": self.attempt,
            "t_start": self._clock(),
            "t_end": None,
            "ok": None,
            "error": None,
            "result_summary": None,
        }
        self.events.append(event)
        return event

    def end_call(self, event: dict, ok: bool, result: Any = None, error: str = "") -> dict:
        event["t_end"] = self._clock()
        event["ok"] = ok
        if ok:
            event["result_summary"] = summarize_result(result)
        else:
            event["error"] = str(error)[:1000]
        return event

    def counts(self) -> dict[str, int]:
        provider = sum(1 for e in self.events if e["kind"] == "provider")
        memory = sum(1 for e in self.events if e["kind"] == "memory")
        failed = sum(1 for e in self.events if e["kind"] in ("provider", "memory") and e["ok"] is False)
        return {
            "provider_calls": provider,
            "memory_calls": memory,
            "total_mcp_calls": provider + memory,
            "failed_calls": failed,
            "discovery_events": sum(1 for e in self.events if e["kind"] == "discovery"),
        }


class TracedMCPServer:
    """Delegate everything to the provider server; trace tool invocations.

    ``list_tools`` is recorded as a discovery event (kept separate from tool
    invocation counts). ``call_tool`` is recorded before awaiting and updated
    with the result or the raised exception.
    """

    def __init__(self, inner: Any, recorder: TraceRecorder, provider: str = "mcp") -> None:
        self._inner = inner
        self._recorder = recorder
        self._provider = provider

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def __aenter__(self) -> "TracedMCPServer":
        inner = self._inner
        enter = getattr(inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(self, *args: Any) -> Any:
        inner = self._inner
        exit = getattr(inner, "__aexit__", None)
        if exit is not None:
            return await exit(*args)
        return False

    async def list_tools(self) -> Any:
        tools = await self._inner.list_tools()
        items = getattr(tools, "tools", tools) or []
        fingerprints = {}
        specs: list[dict[str, Any]] = []
        try:
            for item in items:
                if isinstance(item, dict):
                    name = item.get("name", "?")
                    description = item.get("description", "")
                    schema = item.get("inputSchema") or item.get("input_schema") or {}
                else:
                    name = getattr(item, "name", "?")
                    description = getattr(item, "description", "") or ""
                    schema = getattr(item, "inputSchema", getattr(item, "input_schema", None)) or {}
                spec = {"name": name, "description": description, "input_schema": schema}
                fingerprints[str(name)] = fingerprint_tool(spec)
                specs.append(
                    {
                        "name": str(name)[:128],
                        "description": sanitize_args(description),
                        "input_schema": schema if isinstance(schema, dict) else {},
                    }
                )
        except Exception:
            fingerprints = {}
            specs = []
        if specs:
            self._recorder.tool_specs = specs
        count = len(items) if hasattr(items, "__len__") else -1
        self._recorder.record_discovery(count, fingerprints)
        return tools

    async def call_tool(self, name: str, args: Any = None) -> Any:
        event = self._recorder.begin_call(self._provider, name, args if args is not None else {}, "provider")
        try:
            if args is None:
                result = await self._inner.call_tool(name)
            else:
                result = await self._inner.call_tool(name, args)
        except Exception as exc:
            self._recorder.end_call(event, False, error=f"{type(exc).__name__}: {exc}")
            raise
        self._recorder.end_call(event, True, result=result)
        return result
