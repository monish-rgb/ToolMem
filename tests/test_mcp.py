import pytest
from mcp import Client

from toolatlas.memory_server import create_memory_server
from toolatlas.text_tools_server import mcp as text_server


@pytest.mark.asyncio
async def test_both_local_mcp_servers(tmp_path):
    memory_server = create_memory_server(tmp_path / "memory.json")
    async with Client(text_server) as tools, Client(memory_server) as memory:
        normalized = await tools.call_tool("normalize_text", {"text": "Hello,   MCP!"})
        assert normalized.structured_content == {"result": "hello mcp"}

        await memory.call_tool(
            "remember_execution",
            {
                "task_id": "hello",
                "summary": "Normalize text",
                "steps": [{"tool": "normalize_text", "rationale": "normalize noisy text"}],
                "resolved": True,
            },
        )
        result = await memory.call_tool("get_guidance", {"task": "normalize text"})
        assert result.structured_content["playbook"][0]["tool"] == "normalize_text"

