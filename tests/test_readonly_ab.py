from pathlib import Path

import pytest

from toolatlas.filesystem_demo import filesystem_server_path
from toolatlas.readonly_benchmark import READ_ONLY_TOOLS, run_readonly_ab


@pytest.mark.asyncio
async def test_live_readonly_agent_ab_comparison(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    if not filesystem_server_path(project_root).is_file():
        pytest.skip("run npm install to install the real Filesystem MCP server")

    result = await run_readonly_ab(project_root, tmp_path / "memory.db")

    assert result["comparison"]["both_passed"] is True
    assert result["baseline"]["filesystem_tool_calls"] == 4
    assert result["toolatlas"]["filesystem_tool_calls"] == 2
    assert result["comparison"]["call_reduction_percent"] == 50.0
    assert result["baseline"]["forbidden_write_attempts"] == 0
    assert result["toolatlas"]["forbidden_write_attempts"] == 0
    assert set(result["baseline"]["tools_used"]) <= READ_ONLY_TOOLS
    assert set(result["toolatlas"]["tools_used"]) <= READ_ONLY_TOOLS
