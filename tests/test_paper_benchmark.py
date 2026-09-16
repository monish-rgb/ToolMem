from pathlib import Path

import pytest

from toolatlas.filesystem_demo import filesystem_server_path
from toolatlas.paper_benchmark import _arm_metrics, run_paper_protocol_benchmark


def test_paper_pass_metrics():
    tasks = [
        {"runs": [{"passed": True, "provider_tool_calls": 2, "memory_tool_calls": 1,
                    "total_mcp_calls": 3, "elapsed_ms": 1.0}] * 4},
        {"runs": [{"passed": False, "provider_tool_calls": 2, "memory_tool_calls": 1,
                    "total_mcp_calls": 3, "elapsed_ms": 2.0}] * 3
                   + [{"passed": True, "provider_tool_calls": 2, "memory_tool_calls": 1,
                       "total_mcp_calls": 3, "elapsed_ms": 2.0}]},
    ]
    metrics = _arm_metrics(tasks, 4)
    assert metrics["pass_at_1"] == 0.625
    assert metrics["pass_at_k"] == 1.0
    assert metrics["k"] == 4


@pytest.mark.asyncio
async def test_paper_protocol_filesystem_control(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    if not filesystem_server_path(project_root).is_file():
        pytest.skip("run npm install to install the real Filesystem MCP server")

    monkeypatch.chdir(tmp_path)
    result = await run_paper_protocol_benchmark(project_root, runs=1)

    assert result["claim_scope"] == "local_filesystem_control_not_full_paper_reproduction"
    assert len(result["splits"]) == 2
    for split in result["splits"]:
        assert split["memory_frozen_during_evaluation"] is True
        assert split["baseline"]["metrics"]["pass_at_1"] == 1.0
        assert split["toolatlas"]["metrics"]["pass_at_1"] == 1.0
        assert split["comparison"]["provider_tool_calls"]["percent"] == 50.0
        assert split["comparison"]["total_mcp_calls_including_memory"]["percent"] == 25.0
        assert split["training_cost"]["break_even_evaluation_runs_for_total_mcp_calls"] == 10
