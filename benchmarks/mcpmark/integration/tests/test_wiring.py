"""Wiring tests executed inside the pinned MCPMark image.

Proves the Phase 2 gate on the real integration: the same ToolAtlas-aware
agent class is registered, CLI flags exist, off-mode construction matches the
official agent path, evaluation modes expose no memory writes, and the
evaluator forwards options plus writes the hook record. No model calls.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("OPENAI_API_KEY", "wiring-test-dummy-key")

from src.agents import AGENT_REGISTRY  # noqa: E402
from src.agents.base_agent import BaseMCPAgent  # noqa: E402
from src.evaluator import MCPEvaluator  # noqa: E402
from src.toolatlas_mcpmark.wiring import toolatlas_options_from_args  # noqa: E402


def test_registry_contains_toolatlas_agent():
    assert set(AGENT_REGISTRY) >= {"mcpmark", "react", "toolatlas"}
    cls = AGENT_REGISTRY["toolatlas"]
    assert cls.__name__ == "ToolAtlasMCPMarkAgent"
    assert issubclass(cls, BaseMCPAgent)
    assert issubclass(cls, AGENT_REGISTRY["mcpmark"])


def test_off_mode_construction_matches_official_signature():
    import inspect

    cls = AGENT_REGISTRY["toolatlas"]
    base = AGENT_REGISTRY["mcpmark"]
    assert str(inspect.signature(cls.__init__)) == str(inspect.signature(base.__init__))
    agent = cls(
        litellm_input_model_name="openai/dummy",
        api_key="dummy",
        base_url=None,
        mcp_service="filesystem",
        timeout=60,
        service_config={},
        service_config_provider=lambda: {},
        reasoning_effort="default",
        compaction_token=999999999,
    )
    assert getattr(agent, "_tl_mode", "off") == "off"
    assert agent.toolatlas_attempt_record()["counts"] == {
        "provider_calls": 0,
        "memory_calls": 0,
        "total_mcp_calls": 0,
        "failed_calls": 0,
        "discovery_events": 0,
    }


def test_evaluator_forwards_toolatlas_options():
    with tempfile.TemporaryDirectory() as tmp:
        evaluator = MCPEvaluator(
            mcp_service="filesystem",
            model="openai/dummy",
            timeout=60,
            exp_name="wiring-test",
            output_dir=Path(tmp),
            agent_name="toolatlas",
            toolatlas_mode="read",
            toolatlas_memory=str(Path(tmp) / "mem.db"),
            toolatlas_trace_dir=str(Path(tmp) / "traces"),
            toolatlas_top_k=2,
            toolatlas_read_budget=5,
        )
    assert evaluator.agent._tl_mode == "read"
    assert evaluator.agent._tl_top_k == 2
    assert evaluator.agent._tl_read_budget == 5
    assert evaluator._toolatlas_options["mode"] == "read"


def test_evaluator_defaults_to_off_without_options():
    with tempfile.TemporaryDirectory() as tmp:
        evaluator = MCPEvaluator(
            mcp_service="filesystem",
            model="openai/dummy",
            timeout=60,
            exp_name="wiring-defaults",
            output_dir=Path(tmp),
            agent_name="toolatlas",
        )
    assert evaluator.agent._tl_mode == "off"


def test_cli_exposes_toolatlas_flags():
    proc = subprocess.run(
        [sys.executable, "-m", "pipeline", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0
    for flag in (
        "--agent",
        "--toolatlas-mode",
        "--toolatlas-memory",
        "--toolatlas-trace-dir",
        "--toolatlas-top-k",
        "--toolatlas-read-budget",
    ):
        assert flag in proc.stdout, flag


def test_toolatlas_options_defaults():
    class Args:
        pass

    assert toolatlas_options_from_args(Args()) == {
        "mode": "off",
        "memory_path": None,
        "trace_dir": None,
        "top_k": 3,
        "read_budget": 8,
    }


def test_hook_record_is_json(tmp_path):
    record = {"ingested": False, "reason": "mode is 'off', hook only acts in 'learn'"}
    path = tmp_path / "toolatlas_hook.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    assert json.loads(path.read_text())["ingested"] is False


class _StubMessage:
    def __init__(self):
        self.content = "done"
        self.tool_calls = None
        self.function_call = None

    def model_dump(self):
        return {"role": "assistant", "content": self.content}


class _StubResponse:
    def __init__(self):
        from types import SimpleNamespace

        self.choices = [SimpleNamespace(message=_StubMessage(), finish_reason="stop")]
        self.model = "stub/stub-model"
        self.usage = SimpleNamespace(
            prompt_tokens=10, total_tokens=15, completion_tokens=5,
            completion_tokens_details=None,
        )


class _FakeMCPServer:
    def __init__(self):
        self.listed = 0
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def list_tools(self):
        self.listed += 1
        return [
            {"name": "alpha", "description": "do alpha",
             "inputSchema": {"type": "object"}},
            {"name": "beta", "description": "do beta",
             "inputSchema": {"type": "object"}},
        ]

    async def call_tool(self, name, args=None):
        self.calls.append((name, args))
        return {"ok": True}


def _make_agent(agent_cls, tmp_path):
    return agent_cls(
        litellm_input_model_name="openai/dummy",
        api_key="dummy",
        base_url=None,
        mcp_service="filesystem",
        timeout=60,
        service_config={"test_directory": str(tmp_path)},
        service_config_provider=lambda: {},
        reasoning_effort="default",
        compaction_token=999999999,
    )


def test_off_mode_matches_official_loop(tmp_path, monkeypatch):
    """Deterministic gate: toolatlas --mode off behaves like the stock agent."""
    import asyncio

    import litellm
    from src.toolatlas_mcpmark.wiring import maybe_configure_agent

    seen = {}

    async def _stub_acompletion(**kwargs):
        seen.setdefault("instructions", []).append(kwargs["messages"][1]["content"])
        seen["tools"] = kwargs.get("tools")
        seen["temperature"] = kwargs.get("temperature")
        seen["max_tokens"] = kwargs.get("max_tokens")
        return _StubResponse()

    monkeypatch.setattr(litellm, "acompletion", _stub_acompletion)

    async def _fake_create(self):
        return _FakeMCPServer()

    monkeypatch.setattr(AGENT_REGISTRY["mcpmark"], "_create_mcp_server", _fake_create)
    monkeypatch.setattr(AGENT_REGISTRY["toolatlas"], "_create_mcp_server", _fake_create)

    async def _run(agent_cls, mode):
        agent = _make_agent(agent_cls, tmp_path)
        maybe_configure_agent(agent, {"mode": mode, "memory_path": None,
                                      "trace_dir": None, "top_k": 3, "read_budget": 8})
        result = await agent._execute_litellm_with_tools("Do the thing", None)
        return agent, result

    stock_agent, stock_result = asyncio.run(_run(AGENT_REGISTRY["mcpmark"], "off"))
    assisted_agent, assisted_result = asyncio.run(_run(AGENT_REGISTRY["toolatlas"], "off"))

    assert stock_result["success"] is True and assisted_result["success"] is True
    instructions = seen["instructions"]
    assert instructions[0] == instructions[1] == "Do the thing"
    assert seen["temperature"] == 1.0 and seen["max_tokens"] == 32768
    assert assisted_agent.toolatlas_attempt_record()["counts"]["memory_calls"] == 0
    assert assisted_agent.toolatlas_attempt_record()["counts"]["provider_calls"] == 0
