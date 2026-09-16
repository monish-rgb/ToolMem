import os
from pathlib import Path

import pytest

from toolatlas.github_demo import github_server_path, run_github_demo


@pytest.mark.asyncio
async def test_official_github_mcp_end_to_end(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    if not github_server_path(project_root).is_file():
        pytest.skip("run npm install to install the GitHub MCP server")
    if not os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN"):
        pytest.skip("set GITHUB_PERSONAL_ACCESS_TOKEN to run the GitHub integration")
    repo = os.environ.get("GITHUB_TEST_REPOSITORY", "")
    if "/" in repo:
        owner, name = repo.split("/", 1)
    else:
        owner, name = os.environ.get("GITHUB_TEST_OWNER", ""), os.environ.get("GITHUB_TEST_REPO", "")
    if not owner or not name:
        pytest.skip("set GITHUB_TEST_REPOSITORY='owner/repo' to run the GitHub integration")

    result = await run_github_demo(project_root, tmp_path / "memory.db", owner.strip(), name.strip())

    assert result["discovered_tool_count"] >= 20
    assert result["missing_file_denied"] is True
    assert result["stats"]["strategies"] == 1
    assert [step["tool"] for step in result["guidance"]["playbook"]][:2] == [
        "list_issues",
        "list_commits",
    ]
