from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters


def _value(result):
    data = result.structured_content
    return data.get("result", data) if isinstance(data, dict) else data


async def run_demo(memory_path: Path) -> dict:
    python = sys.executable
    tool_params = StdioServerParameters(
        command=python, args=["-m", "toolatlas.text_tools_server"]
    )
    memory_params = StdioServerParameters(
        command=python,
        args=["-m", "toolatlas.memory_server"],
        env={"TOOLATLAS_MEMORY_PATH": str(memory_path.resolve())},
    )
    async with Client(tool_params) as tools, Client(memory_params) as memory:
        listed = await tools.list_tools()
        specs = [
            {
                "name": item.name,
                "description": item.description or "",
                "input_schema": item.input_schema,
            }
            for item in listed.tools
        ]
        await memory.call_tool("register_tools", {"tools": specs})

        async def learn(task_id, summary, text, keyword):
            normalized_result = await tools.call_tool("normalize_text", {"text": text})
            normalized = _value(normalized_result)
            count_result = await tools.call_tool(
                "keyword_count", {"text": normalized, "keyword": keyword}
            )
            count = _value(count_result)
            expected = normalized.split().count(keyword.lower())
            await memory.call_tool(
                "remember_execution",
                {
                    "task_id": task_id,
                    "summary": summary,
                    "steps": [
                        {"tool": "normalize_text", "rationale": "normalize text before analysis"},
                        {"tool": "keyword_count", "rationale": "count a target keyword in normalized text"},
                    ],
                    "resolved": count == expected,
                    "observation": f"verified expected count {expected}",
                },
            )

        await learn("q_normalize_count", "Normalize text and count a keyword", "Tools, tools; MEMORY!", "tools")
        await learn("q_clean_frequency", "Clean noisy text and compute keyword frequency", "Memory... makes memory reusable", "memory")

        # Execution-verified boundary probe from the paper's outward exploration idea.
        bad = await tools.call_tool("keyword_count", {"text": "anything", "keyword": ""})
        await memory.call_tool(
            "remember_execution",
            {
                "task_id": "q_empty_keyword_boundary",
                "summary": "Count an empty keyword in text",
                "steps": [{"tool": "keyword_count", "rationale": "count a keyword in text"}],
                "resolved": not bad.is_error,
                "observation": "an empty keyword is rejected and must be validated",
            },
        )

        guidance = await memory.call_tool(
            "get_guidance",
            {"task": "Clean a sentence and count how often a keyword occurs", "top_k": 3, "read_budget": 8},
        )
        stats = await memory.call_tool("memory_stats", {})
        return {"stats": _value(stats), "guidance": _value(guidance)}


def main() -> None:
    output = asyncio.run(run_demo(Path("demo-memory.json")))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

