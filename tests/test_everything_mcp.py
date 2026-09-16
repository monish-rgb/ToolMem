from pathlib import Path

import pytest

from toolatlas.everything_demo import everything_server_path, run_everything_demo


@pytest.mark.asyncio
async def test_everything_mcp_all_discovered_tools(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    if not everything_server_path(project_root).is_file():
        pytest.skip("run npm install to install the Everything MCP server")

    result = await run_everything_demo(project_root, tmp_path, tmp_path / "memory.db")

    assert result["discovered_tool_count"] >= 14
    assert result["all_tools_passed"] is True
    assert len(result["tools"]) == result["discovered_tool_count"]
    assert result["environment_sanitized"] is True
    assert [step["tool"] for step in result["guidance"]["playbook"]][:2] == [
        "echo",
        "get-sum",
    ]
    assert result["stats"]["strategies"] == 1
