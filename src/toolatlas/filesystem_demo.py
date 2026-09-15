from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

FILESYSTEM_PACKAGE_VERSION = "2026.8.31"
FILESYSTEM_PROVIDER = "io.github.modelcontextprotocol/server-filesystem"


def _value(result: Any) -> Any:
    data = result.structured_content
    if isinstance(data, dict):
        return data.get("result", data.get("content", data))
    return data


def filesystem_server_path(project_root: Path) -> Path:
    return project_root / "node_modules" / "@modelcontextprotocol" / "server-filesystem" / "dist" / "index.js"


async def run_filesystem_demo(
    project_root: Path,
    sandbox: Path,
    memory_path: Path,
) -> dict[str, Any]:
    """Exercise ToolAtlas against the installed official Filesystem MCP server."""
    project_root = project_root.resolve()
    sandbox = sandbox.resolve()
    memory_path = memory_path.resolve()
    sandbox.mkdir(parents=True, exist_ok=True)
    server_js = filesystem_server_path(project_root)
    if not server_js.is_file():
        raise FileNotFoundError("Filesystem MCP server is not installed; run npm install")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required for the Filesystem MCP integration")

    filesystem_params = StdioServerParameters(
        command=node,
        args=[str(server_js), str(sandbox)],
    )
    memory_params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "toolatlas.memory_server"],
        env={"TOOLATLAS_MEMORY_PATH": str(memory_path)},
    )

    async with Client(filesystem_params) as filesystem, Client(memory_params) as memory:
        listed = await filesystem.list_tools()
        specs = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
                "version": FILESYSTEM_PACKAGE_VERSION,
                "provider": FILESYSTEM_PROVIDER,
            }
            for tool in listed.tools
        ]
        await memory.call_tool("register_tools", {"tools": specs})

        allowed = await filesystem.call_tool("list_allowed_directories", {})
        allowed_text = str(_value(allowed))
        if str(sandbox) not in allowed_text:
            raise AssertionError("Filesystem server did not enforce the requested sandbox root")

        async def write_read_task(task_id: str, filename: str, content: str) -> None:
            target = sandbox / filename
            write_result = await filesystem.call_tool(
                "write_file", {"path": str(target), "content": content}
            )
            read_result = await filesystem.call_tool("read_text_file", {"path": str(target)})
            actual = str(_value(read_result))
            verified = not write_result.is_error and not read_result.is_error and actual == content
            await memory.call_tool(
                "remember_rollouts",
                {
                    "task_id": task_id,
                    "summary": "Write a text file and verify its contents",
                    "rollouts": [
                        {
                            "steps": [
                                {"tool": "write_file", "rationale": "write required content to an allowed file"},
                                {"tool": "read_text_file", "rationale": "read the file back to verify exact contents"},
                            ],
                            "resolved": verified,
                            "observation": "read-back content matched the expected content" if verified else "read-back verification failed",
                            "verifier_type": "exact_match",
                        }
                    ],
                },
            )
            if not verified:
                raise AssertionError(f"Filesystem MCP write/read verification failed for {filename}")

        await write_read_task("fs_write_read_1", "first.txt", "ToolAtlas real MCP integration\n")
        await write_read_task("fs_write_read_2", "second.txt", "Provider-side memory is reusable\n")

        listing = await filesystem.call_tool("list_directory", {"path": str(sandbox)})
        listing_text = str(_value(listing))
        list_verified = "first.txt" in listing_text and "second.txt" in listing_text
        await memory.call_tool(
            "remember_execution",
            {
                "task_id": "fs_list_directory",
                "summary": "List files in an allowed directory",
                "steps": [{"tool": "list_directory", "rationale": "inspect an allowed directory for expected files"}],
                "resolved": list_verified,
                "observation": "directory listing contained both expected files",
                "verifier_type": "membership_check",
            },
        )
        if not list_verified:
            raise AssertionError("Filesystem MCP directory listing verification failed")

        outside = project_root / "README.md"
        denied = await filesystem.call_tool("read_text_file", {"path": str(outside)})
        await memory.call_tool(
            "remember_execution",
            {
                "task_id": "fs_outside_sandbox_boundary",
                "summary": "Read a file outside the allowed directory",
                "steps": [{"tool": "read_text_file", "rationale": "read a text file outside the allowed root"}],
                "resolved": not denied.is_error,
                "observation": "paths outside the configured allowed directory are rejected",
                "verifier_type": "expected_mcp_error",
            },
        )
        if not denied.is_error:
            raise AssertionError("Filesystem MCP server unexpectedly allowed an out-of-sandbox read")

        guidance_result = await memory.call_tool(
            "get_guidance",
            {
                "task": "Create a text file and confirm that its contents were written correctly",
                "top_k": 3,
                "read_budget": 8,
            },
        )
        stats_result = await memory.call_tool("memory_stats", {})
        guidance = _value(guidance_result)
        playbook_tools = [step["tool"] for step in guidance["playbook"]]
        if playbook_tools[:2] != ["write_file", "read_text_file"]:
            raise AssertionError(f"Unexpected learned playbook: {playbook_tools}")
        return {
            "filesystem_server": f"{FILESYSTEM_PROVIDER}@{FILESYSTEM_PACKAGE_VERSION}",
            "sandbox": str(sandbox),
            "discovered_tool_count": len(listed.tools),
            "out_of_sandbox_read_denied": denied.is_error,
            "stats": _value(stats_result),
            "guidance": guidance,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Test ToolAtlas with the official Filesystem MCP server")
    parser.add_argument("--sandbox", type=Path, default=Path("mcp-sandbox"))
    parser.add_argument("--memory", type=Path, default=Path("filesystem-memory.db"))
    args = parser.parse_args()
    output = asyncio.run(run_filesystem_demo(Path.cwd(), args.sandbox, args.memory))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
