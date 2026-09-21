from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .memory_server import create_memory_server
from .memory_paths import fresh_memory_path

GITHUB_PACKAGE_VERSION = "2025.4.8"
GITHUB_PROVIDER = "io.github.modelcontextprotocol/server-github"

# Read-only subset; write-capable tools (create/update/merge/push/...) are never called.
READ_ONLY_GITHUB_TOOLS = {
    "search_repositories",
    "get_file_contents",
    "list_commits",
    "list_issues",
    "search_code",
    "search_issues",
    "search_users",
    "get_issue",
    "get_pull_request",
    "list_pull_requests",
    "get_pull_request_files",
    "get_pull_request_status",
    "get_pull_request_comments",
    "get_pull_request_reviews",
}


def github_server_path(project_root: Path) -> Path:
    return (
        project_root
        / "node_modules"
        / "@modelcontextprotocol"
        / "server-github"
        / "dist"
        / "index.js"
    )


def _value(result: Any) -> Any:
    data = result.structured_content
    if isinstance(data, dict):
        return data.get("result", data.get("content", data))
    return data


def _test_repo() -> tuple[str, str]:
    repo = os.environ.get("GITHUB_TEST_REPOSITORY", "").strip()
    if "github.com/" in repo:
        repo = repo.split("github.com/")[-1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    repo = repo.strip("/")
    if "/" in repo:
        parts = [p.strip() for p in repo.split("/") if p.strip()]
        if len(parts) >= 2:
            return parts[-2], parts[-1]
    owner = os.environ.get("GITHUB_TEST_OWNER", "").strip()
    name = os.environ.get("GITHUB_TEST_REPO", "").strip()
    if owner and name:
        return owner, name
    raise RuntimeError(
        "set GITHUB_TEST_REPOSITORY='owner/repo' (or OWNER+REPO) before running the GitHub demo"
    )


async def run_github_demo(
    project_root: Path,
    memory_path: Path,
    owner: str,
    repo: str,
) -> dict[str, Any]:
    """Exercise ToolAtlas against the pinned official GitHub MCP server (read-only)."""
    project_root = project_root.resolve()
    memory_path = memory_path.resolve()
    server_js = github_server_path(project_root)
    if not server_js.is_file():
        raise FileNotFoundError("GitHub MCP server is not installed; run npm install")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required for the GitHub MCP integration")
    token = os.environ.get("GITHUB_PERSONAL_ACCESS_TOKEN", "")
    if not token:
        raise RuntimeError("set $env:GITHUB_PERSONAL_ACCESS_TOKEN before running the GitHub demo")

    # Inherit only PATH + token so the server cannot see unrelated secrets.
    server_env = {"PATH": os.environ.get("PATH", ""), "GITHUB_PERSONAL_ACCESS_TOKEN": token}
    params = StdioServerParameters(command=node, args=[str(server_js)], env=server_env)
    memory_server = create_memory_server(memory_path)

    async with Client(params) as github, Client(memory_server) as memory:
        listed = await github.list_tools()
        tool_names = [t.name for t in listed.tools]
        for required in ("list_issues", "list_commits"):
            if required not in tool_names:
                raise AssertionError(f"GitHub server missing expected read tool: {required}")
        specs = [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.input_schema,
                "version": GITHUB_PACKAGE_VERSION,
                "provider": GITHUB_PROVIDER,
            }
            for t in listed.tools
            if t.name in READ_ONLY_GITHUB_TOOLS
        ]
        await memory.call_tool("register_tools", {"tools": specs})

        async def remember(task_id: str, summary: str, steps: list[dict[str, str]], resolved: bool, obs: str, verifier: str) -> None:
            await memory.call_tool(
                "remember_execution",
                {"task_id": task_id, "summary": summary, "steps": steps,
                 "resolved": resolved, "observation": obs, "verifier_type": verifier},
            )

        # 1. Repository access check via direct read (search index misses
        # private/forked repos, so don't gate on search_repositories).
        commits_probe = await github.call_tool("list_commits", {"owner": owner, "repo": repo})
        repo_verified = not commits_probe.is_error
        await remember(
            "github_repo_access",
            "Review recent commits for context",
            [{"tool": "list_commits", "rationale": "review recent commits for context"}],
            repo_verified,
            "the commit history was retrieved for the expected repository",
            "list_succeeded",
        )
        if not repo_verified:
            raise AssertionError(
                "GitHub repository access failed: check PAT scopes and GITHUB_TEST_REPOSITORY='owner/repo'"
            )

        # 2 + 3. Two composed overview runs with the SAME 2-tool sequence so a
        # strategy forms (strategies require the sequence in >=2 traces).
        for idx in (1, 2):
            issues = await github.call_tool(
                "list_issues", {"owner": owner, "repo": repo, "state": "open", "perPage": 5}
            )
            commits = await github.call_tool("list_commits", {"owner": owner, "repo": repo})
            composed_ok = not issues.is_error and not commits.is_error
            await remember(
                f"github_repo_overview_{idx}",
                "Inspect repository issues and recent commits for triage",
                [
                    {"tool": "list_issues", "rationale": "list open issues for triage review"},
                    {"tool": "list_commits", "rationale": "review recent commits for context"},
                ],
                composed_ok,
                "the issue list and commit history were both retrieved",
                "both_tools_succeeded",
            )
            if not composed_ok:
                raise AssertionError("GitHub composed overview verification failed")

        # 4. Boundary probe: nonexistent file path must error; stored as caution.
        # Note: this server raises MCPError instead of returning is_error.
        try:
            denied = await github.call_tool(
                "get_file_contents", {"owner": owner, "repo": repo, "path": ".toolatlas-no-such-file-xyz"}
            )
            missing_denied = denied.is_error
        except Exception:
            missing_denied = True
        await remember(
            "github_missing_file_boundary",
            "Read a file path that does not exist in the repository",
            [{"tool": "get_file_contents", "rationale": "read a file path outside known repository content"}],
            missing_denied,
            "paths without repository content are rejected",
            "expected_mcp_error",
        )
        if not missing_denied:
            raise AssertionError("GitHub server unexpectedly returned content for a missing path")

        guidance_result = await memory.call_tool(
            "get_guidance",
            {"task": "Inspect repository issues and recent commits for triage", "top_k": 3, "read_budget": 8},
        )
        stats_result = await memory.call_tool("memory_stats", {})
        guidance = _value(guidance_result)
        playbook_tools = [step["tool"] for step in guidance["playbook"]]
        if playbook_tools[:2] != ["list_issues", "list_commits"]:
            raise AssertionError(f"Unexpected learned playbook: {playbook_tools}")
        return {
            "github_server": f"{GITHUB_PROVIDER}@{GITHUB_PACKAGE_VERSION}",
            "test_repository": f"{owner}/{repo}",
            "discovered_tool_count": len(tool_names),
            "missing_file_denied": missing_denied,
            "stats": _value(stats_result),
            "guidance": guidance,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test ToolAtlas with the official GitHub MCP server (read-only)")
    parser.add_argument("--memory", type=Path, help="database to reuse; omit for a fresh database")
    parser.add_argument("--owner", default="")
    parser.add_argument("--repo", default="")
    args = parser.parse_args()
    owner, repo = (args.owner, args.repo) if args.owner and args.repo else _test_repo()
    memory_path = args.memory or fresh_memory_path("github-demo")
    output = asyncio.run(run_github_demo(Path.cwd(), memory_path, owner, repo))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
