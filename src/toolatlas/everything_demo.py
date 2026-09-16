from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters, types

from .memory_paths import fresh_memory_path
from .memory_server import create_memory_server

EVERYTHING_PACKAGE_VERSION = "2026.8.31"
EVERYTHING_PROVIDER = "io.github.modelcontextprotocol/server-everything"


def everything_server_path(project_root: Path) -> Path:
    return (
        project_root
        / "node_modules"
        / "@modelcontextprotocol"
        / "server-everything"
        / "dist"
        / "index.js"
    )


def _text(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    return "\n".join(str(getattr(block, "text", "")) for block in blocks)


def _arguments(tool_name: str) -> dict[str, Any]:
    arguments: dict[str, dict[str, Any]] = {
        "echo": {"message": "ToolAtlas verification"},
        "get-annotated-message": {"messageType": "success", "includeImage": True},
        "get-env": {},
        "get-resource-links": {"count": 2},
        "get-resource-reference": {"resourceType": "Text", "resourceId": 1},
        "get-roots-list": {},
        "gzip-file-as-resource": {
            "name": "toolatlas.txt.gz",
            "data": "data:text/plain,ToolAtlas%20verification",
            "outputType": "resource",
        },
        "get-structured-content": {"location": "New York"},
        "get-sum": {"a": 19, "b": 23},
        "get-tiny-image": {},
        "trigger-long-running-operation": {"duration": 0.01, "steps": 1},
        "toggle-simulated-logging": {},
        "toggle-subscriber-updates": {},
        "trigger-elicitation-request": {},
        "trigger-url-elicitation": {
            "url": "https://modelcontextprotocol.io",
            "message": "Deterministic integration test; no browser will be opened",
            "errorPath": False,
        },
        "trigger-sampling-request": {
            "prompt": "Return a deterministic integration-test response",
            "maxTokens": 16,
        },
        "simulate-research-query": {"topic": "ToolAtlas", "ambiguous": False},
    }
    if tool_name not in arguments:
        raise AssertionError(f"no deterministic test arguments for discovered tool {tool_name!r}")
    return arguments[tool_name]


async def run_everything_demo(
    project_root: Path,
    client_root: Path,
    memory_path: Path,
) -> dict[str, Any]:
    """Run and verify every tool exposed by the pinned Everything MCP server."""
    project_root = project_root.resolve()
    client_root = client_root.resolve()
    memory_path = memory_path.resolve()
    if not client_root.is_dir():
        raise FileNotFoundError(f"client root does not exist: {client_root}")
    server_js = everything_server_path(project_root)
    if not server_js.is_file():
        raise FileNotFoundError("Everything MCP server is not installed; run npm install")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required for the Everything MCP integration")

    async def roots_callback(_context: Any) -> types.ListRootsResult:
        return types.ListRootsResult(
            roots=[types.Root(uri=client_root.as_uri(), name=client_root.name)]
        )

    async def sampling_callback(
        _context: Any, _params: types.CreateMessageRequestParams
    ) -> types.CreateMessageResult:
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(text="Deterministic ToolAtlas test response"),
            model="toolatlas-deterministic-test",
            stopReason="endTurn",
        )

    async def elicitation_callback(
        _context: Any, _params: types.ElicitRequestParams
    ) -> types.ElicitResult:
        # Exercise the protocol without opening a browser or collecting user data.
        return types.ElicitResult(action="decline")

    # Explicitly replace the inherited environment so get-env cannot reveal credentials.
    server_env = {
        "PATH": os.environ.get("PATH", ""),
        "TOOLATLAS_TEST_ROOT": str(client_root),
        "GZIP_ALLOWED_DOMAINS": "modelcontextprotocol.io",
    }
    params = StdioServerParameters(
        command=node,
        args=[str(server_js), "stdio"],
        cwd=client_root,
        env=server_env,
    )
    memory_server = create_memory_server(memory_path)

    async def call_discovered_tool(
        client: Client, tool_name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        if tool_name != "simulate-research-query":
            return await client.call_tool(tool_name, arguments)

        request = types.CallToolRequest(
            method="tools/call",
            params=types.CallToolRequestParams(
                name=tool_name,
                arguments=arguments,
                task=types.TaskMetadata(ttl=60_000),
            ),
        )
        # mcp-python 2.2.0 serializes task augmentation but its high-level
        # tools/call result validator still accepts only CallToolResult. Use the
        # session transport for the initial CreateTaskResult, then return to the
        # public typed request path for polling and result retrieval.
        request_data = request.model_dump(by_alias=True, mode="json", exclude_none=True)
        call_options: dict[str, Any] = {"timeout": 90}
        client.session._stamp(request_data, call_options)
        created_raw = await client.session._dispatcher.send_raw_request(
            "tools/call", request_data["params"], call_options
        )
        created = types.CreateTaskResult.model_validate(created_raw, by_name=False)
        task_id = created.task.task_id
        while True:
            status = await client.session.send_request(
                types.GetTaskRequest(
                    params=types.GetTaskRequestParams(taskId=task_id)
                ),
                types.GetTaskResult,
                request_read_timeout_seconds=90,
            )
            if status.status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep((status.poll_interval or 100) / 1000)
        if status.status != "completed":
            raise AssertionError(
                f"Everything MCP task {tool_name} ended with {status.status}: "
                f"{status.status_message or ''}"
            )
        return await client.session.send_request(
            types.GetTaskPayloadRequest(
                params=types.GetTaskPayloadRequestParams(taskId=task_id)
            ),
            types.CallToolResult,
            request_read_timeout_seconds=90,
        )

    async with Client(
        params,
        sampling_callback=sampling_callback,
        elicitation_callback=elicitation_callback,
        list_roots_callback=roots_callback,
        read_timeout_seconds=90,
    ) as everything, Client(memory_server) as memory:
        listed = await everything.list_tools()
        tool_names = [tool.name for tool in listed.tools]
        specs = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
                "version": EVERYTHING_PACKAGE_VERSION,
                "provider": EVERYTHING_PROVIDER,
            }
            for tool in listed.tools
        ]
        await memory.call_tool("register_tools", {"tools": specs})

        results: list[dict[str, Any]] = []
        for index, tool_name in enumerate(tool_names, start=1):
            result = await call_discovered_tool(
                everything, tool_name, _arguments(tool_name)
            )
            verified = not result.is_error
            text = _text(result)
            if tool_name == "get-roots-list":
                verified = verified and client_root.as_uri() in text
            elif tool_name == "get-sum":
                verified = verified and "42" in text
            elif tool_name == "get-env":
                verified = (
                    verified
                    and "TOOLATLAS_TEST_ROOT" in text
                    and "NVIDIA_API_KEY" not in text
                    and "GITHUB_PERSONAL_ACCESS_TOKEN" not in text
                )

            await memory.call_tool(
                "remember_execution",
                {
                    "task_id": f"everything_tool_{index}",
                    "summary": f"Exercise the {tool_name} capability with deterministic input",
                    "steps": [
                        {
                            "tool": tool_name,
                            "rationale": "exercise the capability with deterministic test input",
                        }
                    ],
                    "resolved": verified,
                    "observation": (
                        "the tool returned the expected successful response"
                        if verified
                        else "the tool response failed deterministic verification"
                    ),
                    "verifier_type": "deterministic_protocol_check",
                },
            )
            results.append({"tool": tool_name, "passed": verified})
            if not verified:
                raise AssertionError(
                    f"Everything MCP verification failed for {tool_name}: {_text(result)}"
                )

            # Stop interval-based simulations immediately after proving they start.
            if tool_name in {"toggle-simulated-logging", "toggle-subscriber-updates"}:
                stopped = await everything.call_tool(tool_name, {})
                if stopped.is_error:
                    raise AssertionError(f"failed to stop {tool_name}")

        for task_index in range(1, 3):
            echo_result = await everything.call_tool(
                "echo", {"message": f"composed verification {task_index}"}
            )
            sum_result = await everything.call_tool(
                "get-sum", {"a": 20 + task_index, "b": 22 - task_index}
            )
            composed_verified = (
                not echo_result.is_error
                and not sum_result.is_error
                and "42" in _text(sum_result)
            )
            await memory.call_tool(
                "remember_execution",
                {
                    "task_id": f"everything_composed_{task_index}",
                    "summary": "Echo a label and calculate the sum of two numbers",
                    "steps": [
                        {
                            "tool": "echo",
                            "rationale": "echo the requested label for confirmation",
                        },
                        {
                            "tool": "get-sum",
                            "rationale": "calculate the requested numeric sum",
                        },
                    ],
                    "resolved": composed_verified,
                    "observation": "the label was echoed and the exact sum matched",
                    "verifier_type": "exact_composed_result",
                },
            )
            if not composed_verified:
                raise AssertionError("Everything MCP composed verification failed")

        guidance_result = await memory.call_tool(
            "get_guidance",
            {
                "task": "Echo a message and calculate the sum of two values",
                "top_k": 3,
                "read_budget": 8,
            },
        )
        guidance = guidance_result.structured_content
        if isinstance(guidance, dict):
            guidance = guidance.get("result", guidance)
        playbook = [step["tool"] for step in guidance["playbook"]]
        if playbook[:2] != ["echo", "get-sum"]:
            raise AssertionError(f"unexpected Everything MCP playbook: {playbook}")

        stats_result = await memory.call_tool("memory_stats", {})
        stats = stats_result.structured_content
        if isinstance(stats, dict):
            stats = stats.get("result", stats)
        return {
            "everything_server": f"{EVERYTHING_PROVIDER}@{EVERYTHING_PACKAGE_VERSION}",
            "client_root": str(client_root),
            "memory_db": str(memory_path),
            "discovered_tool_count": len(tool_names),
            "tools": results,
            "all_tools_passed": all(item["passed"] for item in results),
            "environment_sanitized": True,
            "stats": stats,
            "guidance": guidance,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test ToolAtlas with every discovered Everything MCP tool"
    )
    parser.add_argument("--root", type=Path, required=True, help="root advertised to the MCP server")
    parser.add_argument(
        "--memory",
        type=Path,
        help="database to reuse; omit for a fresh, uniquely named database",
    )
    args = parser.parse_args()
    memory_path = args.memory or fresh_memory_path("everything-demo")
    output = asyncio.run(run_everything_demo(Path.cwd(), args.root, memory_path))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
