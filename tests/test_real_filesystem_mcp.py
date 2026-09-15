from pathlib import Path

import pytest

from toolatlas.filesystem_demo import filesystem_server_path, run_filesystem_demo


@pytest.mark.asyncio
async def test_official_filesystem_mcp_end_to_end(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    if not filesystem_server_path(project_root).is_file():
        pytest.skip("run npm install to install the real Filesystem MCP server")

    result = await run_filesystem_demo(
        project_root,
        tmp_path / "allowed-files",
        tmp_path / "memory.db",
    )

    assert result["discovered_tool_count"] >= 14
    assert result["out_of_sandbox_read_denied"] is True
    assert result["stats"]["traces"] == 4
    assert result["stats"]["strategies"] == 1
    assert [step["tool"] for step in result["guidance"]["playbook"]][:2] == [
        "write_file",
        "read_text_file",
    ]
