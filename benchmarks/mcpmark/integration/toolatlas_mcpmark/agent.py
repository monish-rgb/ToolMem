"""ToolAtlas-aware MCPMark agent logic.

``ToolAtlasAgentMixin`` carries all ToolAtlas behavior without importing any
MCPMark module, so it unit-tests with a fake base. Production wiring binds it
to the real ``MCPMarkAgent`` via :func:`create_agent_class`, reusing the
official model and provider-tool execution path instead of copying it.

Modes (set per run via :meth:`configure_toolatlas`):

- ``off``: identical code path to the official agent — same prompt, same
  listed provider tools, same budgets and provider calls; zero memory calls.
- ``read``: exactly one read-only ``get_guidance`` call per attempt before the
  provider loop; the fixed guidance block is injected only when retrieval is
  non-empty. Memory mutation is impossible (read-only server profile).
- ``learn``: no memory calls during execution; after the independent official
  verifier runs, :meth:`on_verified_rollout` ingests the sanitized trace as
  affordance evidence (verified success) or boundary evidence (failure).
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Optional

from .memory_stdio import open_stdio_memory
from .sanitize import (
    format_guidance_block,
    guidance_is_empty,
    retrieval_query,
    sanitize_text,
)
from .tracing import TraceRecorder, TracedMCPServer

MODES = ("off", "learn", "read")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]+", "_", text).strip("_")[:120]


class ToolAtlasAgentMixin:
    """ToolAtlas behavior layered over an MCPMark-compatible base agent."""

    def configure_toolatlas(
        self,
        *,
        mode: str = "off",
        memory_path: Optional[str | Path] = None,
        trace_dir: Optional[str | Path] = None,
        top_k: int = 3,
        read_budget: int = 8,
        token_budget: int = 384,
        embed_mode: str = "lexical",
        task_id: str = "",
        arm: str = "",
        attempt: int = 0,
        package_dir: Optional[str | Path] = None,
        python_exe: Optional[str] = None,
        session_factory: Any = None,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown toolatlas mode: {mode!r} (expected one of {MODES})")
        if mode in ("learn", "read") and not memory_path:
            raise ValueError(f"toolatlas mode '{mode}' requires a memory path")
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        if embed_mode not in ("lexical", "embedding"):
            raise ValueError("embed_mode must be 'lexical' or 'embedding'")
        self._tl_mode = mode
        self._tl_memory = str(memory_path) if memory_path else ""
        self._tl_trace_dir = Path(trace_dir) if trace_dir else None
        self._tl_top_k = int(top_k)
        self._tl_read_budget = int(read_budget)
        self._tl_token_budget = int(token_budget)
        self._tl_embed_mode = embed_mode
        self._tl_package_dir = str(package_dir) if package_dir else None
        self._tl_python_exe = python_exe
        self._tl_session_factory = session_factory or open_stdio_memory
        self._tl_recorder = TraceRecorder(task_id=task_id, arm=arm, attempt=attempt)
        self._tl_task_id = task_id
        self._tl_arm = arm
        self._tl_attempt = attempt
        self._tl_guidance_raw: dict = {}
        self._tl_guidance_block = ""
        self._tl_pending_rollout: Optional[dict] = None

    # -- per-attempt context (called by the evaluator patch) -----------------
    def begin_attempt(self, task_id: str, arm: str = "", attempt: int = 0) -> None:
        if not task_id.strip():
            raise ValueError("begin_attempt requires an explicit task_id")
        if arm not in ("baseline", "toolatlas"):
            raise ValueError(f"begin_attempt requires arm 'baseline'|'toolatlas', got {arm!r}")
        if attempt < 1:
            raise ValueError(f"begin_attempt requires attempt >= 1, got {attempt!r}")
        mode = getattr(self, "_tl_mode", "off")
        cfg = {
            "mode": mode,
            "memory_path": getattr(self, "_tl_memory", ""),
            "trace_dir": getattr(self, "_tl_trace_dir", None),
            "top_k": getattr(self, "_tl_top_k", 3),
            "read_budget": getattr(self, "_tl_read_budget", 8),
            "token_budget": getattr(self, "_tl_token_budget", 384),
            "embed_mode": getattr(self, "_tl_embed_mode", "lexical"),
            "package_dir": getattr(self, "_tl_package_dir", None),
            "python_exe": getattr(self, "_tl_python_exe", None),
            "session_factory": getattr(self, "_tl_session_factory", None),
        }
        self.configure_toolatlas(task_id=task_id, arm=arm, attempt=attempt, **cfg)

    # -- interception points (signatures identical to the official agent) -----
    async def _create_mcp_server(self) -> Any:
        server = await super()._create_mcp_server()  # type: ignore[misc]
        recorder = getattr(self, "_tl_recorder", None)
        if recorder is None:
            return server
        return TracedMCPServer(server, recorder)

    async def _execute_litellm_with_tools(
        self, instruction: str, tool_call_log_file: Any = None
    ) -> dict:
        original = instruction
        instruction = await self._prepare_instruction(instruction)
        result = await super()._execute_litellm_with_tools(instruction, tool_call_log_file)  # type: ignore[misc]
        return self._finish_attempt(original, result)

    async def _execute_claude_native_with_tools(
        self, instruction: str, tool_call_log_file: Any = None
    ) -> dict:
        original = instruction
        instruction = await self._prepare_instruction(instruction)
        result = await super()._execute_claude_native_with_tools(instruction, tool_call_log_file)  # type: ignore[misc]
        return self._finish_attempt(original, result)

    async def _prepare_instruction(self, instruction: str) -> str:
        if getattr(self, "_tl_mode", "off") != "read":
            return instruction
        block = await self._fetch_guidance(instruction)
        if block:
            return block + "\n\n" + instruction
        return instruction

    def _finish_attempt(self, original_instruction: str, result: Any) -> Any:
        if getattr(self, "_tl_mode", "off") == "learn":
            recorder = getattr(self, "_tl_recorder", None) or TraceRecorder()
            self._tl_pending_rollout = {
                "instruction": original_instruction,
                "provider_steps": [
                    {"tool": e["tool"], "rationale": "invoke %s during the rollout" % e["tool"]}
                    for e in recorder.events
                    if e["kind"] == "provider" and e["ok"] is not False
                ],
            }
        self._write_trace()
        if isinstance(result, dict):
            result.setdefault("toolatlas", self.toolatlas_attempt_record())
        return result

    async def _fetch_guidance(self, instruction: str) -> str:
        query = retrieval_query(instruction)
        opener = self._tl_session_factory
        async with opener(self._tl_memory, True, self._tl_package_dir, self._tl_python_exe, self._tl_recorder) as session:
            raw = await session.guidance(
                query, self._tl_top_k, self._tl_read_budget,
                getattr(self, "_tl_token_budget", 384),
                getattr(self, "_tl_embed_mode", "lexical"))
        self._tl_guidance_raw = raw if isinstance(raw, dict) else {}
        self._tl_guidance_block = format_guidance_block(
            self._tl_guidance_raw,
            token_budget=getattr(self, "_tl_token_budget", 384))
        return self._tl_guidance_block

    # -- post-verifier learning hook (evaluator calls this, learn mode only) --
    def on_verified_rollout(
        self,
        *,
        task_id: str = "",
        instruction: str = "",
        verifier_success: bool = False,
        verification_output: str = "",
    ) -> dict:
        mode = getattr(self, "_tl_mode", "off")
        if mode != "learn":
            return {"ingested": False, "reason": f"mode is '{mode}', hook only acts in 'learn'"}
        pending = getattr(self, "_tl_pending_rollout", None) or {}
        steps = pending.get("provider_steps") or []
        recorder = getattr(self, "_tl_recorder", None)
        tool_specs = list(getattr(recorder, "tool_specs", None) or [])
        summary = sanitize_text(instruction or task_id, limit=200) or sanitize_text(task_id, limit=200)
        observation = sanitize_text(
            ("verified success. " if verifier_success else "verifier failure. ")
            + (verification_output or ""),
            limit=500,
        )
        opener = self._tl_session_factory
        outcome = asyncio.run(_learn_once(opener, self._tl_memory, self._tl_package_dir, self._tl_python_exe, task_id or self._tl_task_id, summary, tool_specs, steps, verifier_success, observation))
        self._tl_pending_rollout = None
        return {"ingested": True, "resolved": verifier_success, "detail": outcome}

    # -- records --------------------------------------------------------------
    def toolatlas_attempt_record(self) -> dict:
        recorder = getattr(self, "_tl_recorder", None) or TraceRecorder()
        return {
            "mode": getattr(self, "_tl_mode", "off"),
            "task_id": getattr(self, "_tl_task_id", ""),
            "arm": getattr(self, "_tl_arm", ""),
            "attempt": getattr(self, "_tl_attempt", 0),
            "counts": recorder.counts(),
            "retrieved_guidance": getattr(self, "_tl_guidance_raw", {}),
            "injected_guidance": getattr(self, "_tl_guidance_block", ""),
            "guidance_empty": guidance_is_empty(getattr(self, "_tl_guidance_raw", {})),
        }

    def _write_trace(self) -> Optional[str]:
        trace_dir = getattr(self, "_tl_trace_dir", None)
        if trace_dir is None:
            return None
        trace_dir = Path(trace_dir)
        trace_dir.mkdir(parents=True, exist_ok=True)
        name = "%s__%s__attempt%s.json" % (
            _slug(getattr(self, "_tl_task_id", "task")),
            _slug(getattr(self, "_tl_arm", "arm")),
            getattr(self, "_tl_attempt", 0),
        )
        path = trace_dir / name
        recorder = getattr(self, "_tl_recorder", None) or TraceRecorder()
        payload = {
            "attempt_record": self.toolatlas_attempt_record(),
            "events": recorder.events,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return str(path)


async def _learn_once(opener, memory_path, package_dir, python_exe, task_id, summary, tool_specs, steps, resolved, observation):
    async with opener(memory_path, False, package_dir, python_exe, None) as session:
        return await session.learn(task_id, summary, tool_specs, steps, resolved, observation)


def create_agent_class(base: Any) -> Any:
    """Bind the mixin to a concrete MCPMark agent base class."""

    class ToolAtlasMCPMarkAgent(ToolAtlasAgentMixin, base):
        """Same official agent execution in both arms, plus ToolAtlas modes."""

    ToolAtlasMCPMarkAgent.__name__ = "ToolAtlasMCPMarkAgent"
    ToolAtlasMCPMarkAgent.__qualname__ = "ToolAtlasMCPMarkAgent"
    return ToolAtlasMCPMarkAgent
