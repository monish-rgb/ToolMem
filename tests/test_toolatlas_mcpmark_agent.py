"""Unit tests for the ToolAtlas MCPMark integration (Phase 2 gate).

Uses a fake agent base, fake provider MCP server, and fake memory sessions —
no MCPMark checkout, model, or subprocess required. Proves the gate
properties: off-mode equivalence, exception counting, exactly-one guidance
call, empty-guidance passthrough, write-impossible evaluation, and
verification-gated learning with sanitized evidence.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "mcpmark" / "integration")
)

from toolatlas_mcpmark.agent import MODES, create_agent_class  # noqa: E402
from toolatlas_mcpmark.memory_stdio import CREDENTIAL_ENV_KEYS, memory_env  # noqa: E402
from toolatlas_mcpmark.sanitize import (  # noqa: E402
    format_guidance_block,
    guidance_is_empty,
    sanitize_args,
    sanitize_text,
)
from toolatlas_mcpmark.tracing import TraceRecorder, TracedMCPServer  # noqa: E402
from toolatlas_mcpmark.wiring import (  # noqa: E402
    maybe_run_learning_hook,
    register_agent,
    toolatlas_options_from_args,
)

WRITE_TOOLS = {
    "register_tools",
    "remember_execution",
    "remember_rollouts",
    "reverify_trace",
    "set_trace_status",
}


class FakeProviderServer:
    """Fake provider MCP server with one raising tool."""

    def __init__(self, fail_on=()):
        self.fail_on = set(fail_on)
        self.calls = []

    async def list_tools(self):
        return FakeToolList(["alpha", "beta"])

    async def call_tool(self, name, args=None):
        self.calls.append((name, args or {}))
        if name in self.fail_on:
            raise RuntimeError(f"provider denied {name}")
        return {"ok": True, "tool": name}


class FakeToolList:
    def __init__(self, names):
        self.tools = [
            {"name": n, "description": f"do {n}", "inputSchema": {"type": "object"}}
            for n in names
        ]


class FakeBase:
    """Fake MCPMark-compatible base mirroring the official entry shapes."""

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        self.server_factory = FakeProviderServer

    async def _create_mcp_server(self):
        return self.server_factory()

    async def _execute_litellm_with_tools(self, instruction, tool_call_log_file=None):
        self.seen_instruction = instruction
        mcp_server = await self._create_mcp_server()
        tools = await mcp_server.list_tools()
        names = [t["name"] for t in tools.tools]
        for name in names:
            try:
                await mcp_server.call_tool(name, {"path": "/secret/x", "q": "hi"})
            except Exception:
                pass
        return {"success": True, "output": [], "token_usage": {}, "turn_count": 1}


FakeAgent = create_agent_class(FakeBase)


class FakeSession:
    def __init__(self, guidance, forbid_writes=False, recorder=None):
        self._guidance = guidance
        self.forbid_writes = forbid_writes
        self._recorder = recorder
        self.calls = []

    async def _call(self, tool, args, func):
        event = None
        if self._recorder is not None:
            event = self._recorder.begin_call("toolatlas-memory", tool, args, "memory")
        try:
            result = await func()
        except Exception as exc:
            if event is not None:
                self._recorder.end_call(event, False, error=f"{type(exc).__name__}: {exc}")
            raise
        if event is not None:
            self._recorder.end_call(event, True, result=result)
        return result

    async def guidance(self, task, top_k, read_budget):
        args = {"task": task, "top_k": top_k, "read_budget": read_budget}
        self.calls.append({"tool": "get_guidance", **args})

        async def _do():
            return self._guidance

        return await self._call("get_guidance", args, _do)

    async def learn(self, task_id, summary, tool_specs, steps, resolved, observation):
        args = {"task_id": task_id, "summary": summary, "steps": steps,
                "resolved": resolved, "observation": observation}
        self.calls.append({"tool": "remember_execution", **args, "specs": tool_specs})

        async def _do():
            if self.forbid_writes:
                raise AssertionError("write tool called in evaluation mode")
            return {"qid": task_id}

        return await self._call("remember_execution", args, _do)


def factory_with(guidance, sessions, forbid_writes=False):
    @asynccontextmanager
    async def _open(db_path, read_only=False, package_dir=None, python_exe=None, recorder=None):
        session = FakeSession(guidance, forbid_writes, recorder)
        sessions.append(session)
        yield session
    return _open


FULL_GUIDANCE = {
    "seed_candidates": [{"summary": "Find a thing", "confidence": 0.6}],
    "playbook": [{"tool": "alpha", "rationale": "locate files first"}],
}


def make_agent(mode, **kwargs):
    agent = FakeAgent()
    params = {"mode": mode, "memory_path": "mem.db", "trace_dir": None,
              "top_k": 3, "read_budget": 8, "task_id": "cat/task",
              "arm": "test", "attempt": 1}
    params.update(kwargs)
    agent.configure_toolatlas(**params)
    return agent


def test_modes_known():
    assert MODES == ("off", "learn", "read")


def test_off_mode_is_equivalent_and_silent():
    sessions = []
    agent = make_agent("off", session_factory=factory_with({}, sessions))
    result =     asyncio.run(agent._execute_litellm_with_tools("Do the thing", None))
    assert agent.seen_instruction == "Do the thing"
    assert sessions == []
    assert result["toolatlas"]["counts"]["memory_calls"] == 0
    assert result["toolatlas"]["counts"]["provider_calls"] == 2
    assert result["toolatlas"]["guidance_empty"] is True


def test_provider_exception_is_counted_with_error(tmp_path):
    agent = make_agent("off", trace_dir=tmp_path)
    agent.server_factory = lambda: FakeProviderServer(fail_on={"alpha"})
    asyncio.run(agent._execute_litellm_with_tools("Do it", None))
    record = agent.toolatlas_attempt_record()
    assert record["counts"]["provider_calls"] == 2
    assert record["counts"]["failed_calls"] == 1
    errors = [e for e in agent._tl_recorder.events if e.get("ok") is False]
    assert errors and "provider denied alpha" in errors[0]["error"]
    trace_file = tmp_path / "cat_task__test__attempt1.json"
    assert trace_file.exists()
    payload = json.loads(trace_file.read_text())
    assert payload["attempt_record"]["counts"]["failed_calls"] == 1


def test_read_mode_single_guidance_and_injection():
    sessions = []
    agent = make_agent("read", session_factory=factory_with(dict(FULL_GUIDANCE), sessions))
    asyncio.run(agent._execute_litellm_with_tools("Do the thing", None))
    assert len(sessions) == 1
    assert [c["tool"] for c in sessions[0].calls] == ["get_guidance"]
    assert agent.seen_instruction.startswith("[ToolAtlas memory guidance")
    assert agent.seen_instruction.endswith("Do the thing")
    record = agent.toolatlas_attempt_record()
    assert record["counts"]["memory_calls"] == 1
    assert record["guidance_empty"] is False
    assert "alpha" in record["injected_guidance"]


def test_read_mode_empty_guidance_injects_nothing():
    sessions = []
    agent = make_agent("read", session_factory=factory_with(
        {"seed_candidates": [], "playbook": []}, sessions))
    asyncio.run(agent._execute_litellm_with_tools("Do the thing", None))
    assert agent.seen_instruction == "Do the thing"
    assert len(sessions) == 1
    assert agent.toolatlas_attempt_record()["guidance_empty"] is True


def test_read_mode_cannot_write():
    sessions = []
    agent = make_agent("read", session_factory=factory_with(dict(FULL_GUIDANCE), sessions, forbid_writes=True))
    asyncio.run(agent._execute_litellm_with_tools("Do it", None))
    assert agent.toolatlas_attempt_record()["counts"]["memory_calls"] == 1


def test_learn_mode_defers_writes_until_verified_hook():
    sessions = []
    agent = make_agent("learn", session_factory=factory_with({}, sessions))
    asyncio.run(agent._execute_litellm_with_tools("Solve C:\\work\\x with key sk-123", None))
    assert sessions == []
    assert agent.toolatlas_attempt_record()["counts"]["memory_calls"] == 0
    outcome = agent.on_verified_rollout(
        task_id="cat/task", instruction="Solve C:\\work\\x with key sk-123",
        verifier_success=True, verification_output="All 8 files ok in /tmp/out")
    assert outcome["ingested"] is True
    assert len(sessions) == 1
    learned = sessions[0].calls[0]
    assert learned["tool"] == "remember_execution"
    assert learned["resolved"] is True
    blob = json.dumps(learned)
    assert "C:\\work" not in blob and "sk-123" not in blob and "/tmp/out" not in blob


def test_learn_mode_failure_becomes_boundary_evidence():
    sessions = []
    agent = make_agent("learn", session_factory=factory_with({}, sessions))
    asyncio.run(agent._execute_litellm_with_tools("Try it", None))
    outcome = agent.on_verified_rollout(
        task_id="cat/task", instruction="Try it",
        verifier_success=False, verification_output="small_files missing")
    assert outcome["ingested"] is True and outcome["resolved"] is False
    assert sessions[0].calls[0]["resolved"] is False


def test_hook_skipped_outside_learn_mode():
    for mode in ("off", "read"):
        agent = make_agent(mode)
        outcome = agent.on_verified_rollout(task_id="t", instruction="i", verifier_success=True)
        assert outcome["ingested"] is False


def test_hook_failure_never_breaks_run():
    class Exploding:
        def on_verified_rollout(self, **kwargs):
            raise RuntimeError("boom")
    record = maybe_run_learning_hook(Exploding(), task_id="t", instruction="i",
                                     verifier_success=True, verification_output="")
    assert record["ingested"] is False and "boom" in record["reason"]


def test_sanitizer_strips_secrets_paths_numbers():
    dirty = 'Use KEY=sk-abc123XYZ at C:\\data\\f quota 300 in "small_files" now'
    clean = sanitize_text(dirty)
    assert "sk-abc123XYZ" not in clean and "C:\\data" not in clean
    assert "<secret>" in clean and "<path>" in clean and "<value>" in clean
    assert sanitize_text(123) == "<value>"


def test_sanitize_args_preserves_structure():
    args = {"path": "/a/b", "n": 7, "items": ["x", 2], "ok": True}
    clean = sanitize_args(args)
    assert clean["path"] == "<path>" and clean["n"] == "<value>"
    assert clean["items"] == ["x", "<value>"] and clean["ok"] is True


def test_guidance_block_empty_when_no_evidence():
    assert guidance_is_empty({}) is True
    assert guidance_is_empty({"seed_candidates": [], "playbook": []}) is True
    assert format_guidance_block({}) == ""
    assert format_guidance_block(dict(FULL_GUIDANCE)).startswith("[ToolAtlas")


def test_memory_env_has_no_credentials():
    env = memory_env("mem.db", read_only=True)
    assert env["TOOLATLAS_MEMORY_PATH"] == "mem.db"
    assert env["TOOLATLAS_READ_ONLY"] == "1"
    assert not (set(env) & set(CREDENTIAL_ENV_KEYS))
    env2 = memory_env("mem.db")
    assert "TOOLATLAS_READ_ONLY" not in env2


def test_wiring_helpers():
    registry = register_agent({"mcpmark": object}, object)
    assert set(registry) == {"mcpmark", "toolatlas"}
    with pytest.raises(ValueError):
        register_agent(registry, object)

    class Args:
        pass
    assert toolatlas_options_from_args(Args())["mode"] == "off"
    assert maybe_run_learning_hook(object(), task_id="t", instruction="i",
                                   verifier_success=True, verification_output="")["ingested"] is False


def test_traced_server_delegates_and_records():
    recorder = TraceRecorder(task_id="t", arm="a", attempt=1)
    server = TracedMCPServer(FakeProviderServer(), recorder, provider="fs")
    assert asyncio.run(server.list_tools()).tools[0]["name"] == "alpha"
    assert asyncio.run(server.call_tool("beta", {"x": 1})) == {"ok": True, "tool": "beta"}
    counts = recorder.counts()
    assert counts == {"provider_calls": 1, "memory_calls": 0, "total_mcp_calls": 1,
                      "failed_calls": 0, "discovery_events": 1}
    assert recorder.tool_specs and recorder.tool_specs[0]["name"] == "alpha"
