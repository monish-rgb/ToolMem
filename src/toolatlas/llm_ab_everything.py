from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters, types

from .everything_demo import EVERYTHING_PACKAGE_VERSION, EVERYTHING_PROVIDER, everything_server_path
from .llm_ab_nim import DEFAULT_BASE_URL, DEFAULT_MODEL, DEFAULT_TEMPERATURE, _openai_client, _parse_answer_int
from .memory_server import create_memory_server
from .readonly_benchmark import _value

DEFAULT_MAX_STEPS = 8

SYSTEM_PROMPT = (
    "You are an Everything MCP agent. Use only the provided tools. "
    "Answer with the echoed label and the exact numeric sum."
)
# Neutral baseline: no echo->sum hint. Arm B gets it via memory.
TASKS = [
    {"label": "production verification", "a": 19, "b": 23, "expected": 42},
    {"label": "staging checklist", "a": 10, "b": 15, "expected": 25},
    {"label": "cache audit", "a": 7, "b": 8, "expected": 15},
]


@dataclass
class EverythingAudit:
    client: Client
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        started = time.perf_counter()
        result = await self.client.call_tool(tool, arguments)
        self.calls.append(
            {"tool": tool, "is_error": result.is_error, "latency_ms": round((time.perf_counter() - started) * 1000, 3)}
        )
        return result


def mcp_to_openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema or {"type": "object", "properties": {}},
            },
        }
        for tool in mcp_tools
    ]


def _text(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    return "\n".join(str(getattr(block, "text", "")) for block in blocks)


async def _seed_training_memory(audit: EverythingAudit, memory: Client) -> None:
    for idx, (label, a, b) in enumerate([("training alpha", 20, 22), ("training beta", 21, 21)], start=1):
        echo = await audit.call("echo", {"message": label})
        total = await audit.call("get-sum", {"a": a, "b": b})
        ok = not echo.is_error and not total.is_error and str(a + b) in _text(total)
        await memory.call_tool(
            "remember_execution",
            {
                "task_id": f"everything_composed_train_{idx}",
                "summary": "Echo a label and calculate the sum of two numbers",
                "steps": [
                    {"tool": "echo", "rationale": "echo the requested label for confirmation"},
                    {"tool": "get-sum", "rationale": "calculate the requested numeric sum"},
                ],
                "resolved": ok,
                "observation": "the label was echoed and the exact sum matched",
                "verifier_type": "exact_composed_result",
            },
        )
        if not ok:
            raise AssertionError("training verifier failed for Everything MCP")


async def _run_llm_agent(
    model: str, temperature: float, max_steps: int, system_prompt: str,
    openai_tools: list[dict[str, Any]], audit: EverythingAudit, task: dict[str, Any],
) -> dict[str, Any]:
    client = _openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Echo the label '{task['label']}' and calculate the sum of {task['a']} and {task['b']}. Report the label and the integer sum.",
        },
    ]
    steps = 0
    while steps < max_steps:
        steps += 1
        response = await asyncio.to_thread(
            client.chat.completions.create, model=model, messages=messages,
            tools=openai_tools or None, tool_choice="auto" if openai_tools else "none",
            temperature=temperature, max_tokens=512,
        )
        choice = response.choices[0].message
        if not choice.tool_calls:
            text = choice.content or ""
            answer = _parse_answer_int(text)
            passed = answer == task["expected"] and task["label"].split()[0].lower() in text.lower()
            return {"final_text": text, "answer": answer, "passed": passed, "steps": steps}
        messages.append(
            {
                "role": "assistant", "content": choice.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in choice.tool_calls
                ],
            }
        )
        for tc in choice.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = await audit.call(tc.function.name, args)
                payload = _text(result) or str(_value(result))
                if result.is_error:
                    payload = f"MCP tool error: {payload}"
            except Exception as exc:  # keep loop alive on tool errors
                payload = f"Blocked: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": payload[:8000]})
    return {"final_text": "", "answer": None, "passed": False, "steps": steps}


async def run_everything_nim_ab(
    project_root: Path, client_root: Path, memory_path: Path,
    model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE, max_steps: int = DEFAULT_MAX_STEPS,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    client_root = client_root.resolve()
    if not client_root.is_dir():
        raise FileNotFoundError(f"client root does not exist: {client_root}")
    server_js = everything_server_path(project_root)
    if not server_js.is_file():
        raise FileNotFoundError("run npm install before the Everything NIM A/B test")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required")

    async def roots_callback(_ctx: Any) -> types.ListRootsResult:
        return types.ListRootsResult(roots=[types.Root(uri=client_root.as_uri(), name=client_root.name)])

    async def sampling_callback(_ctx: Any, _p: types.CreateMessageRequestParams) -> types.CreateMessageResult:
        return types.CreateMessageResult(
            role="assistant", content=types.TextContent(text="Deterministic ToolAtlas test response"),
            model="toolatlas-deterministic-test", stopReason="endTurn",
        )

    async def elicitation_callback(_ctx: Any, _p: types.ElicitRequestParams) -> types.ElicitResult:
        return types.ElicitResult(action="decline")

    server_env = {"PATH": os.environ.get("PATH", ""), "TOOLATLAS_TEST_ROOT": str(client_root)}
    params = StdioServerParameters(command=node, args=[str(server_js), "stdio"], cwd=client_root, env=server_env)
    memory_server = create_memory_server(memory_path)

    async with Client(
        params, sampling_callback=sampling_callback, elicitation_callback=elicitation_callback,
        list_roots_callback=roots_callback, read_timeout_seconds=90,
    ) as everything, Client(memory_server) as memory:
        listed = await everything.list_tools()
        await memory.call_tool(
            "register_tools",
            {"tools": [
                {"name": t.name, "description": t.description or "", "input_schema": t.input_schema,
                 "provider": EVERYTHING_PROVIDER, "version": EVERYTHING_PACKAGE_VERSION}
                for t in listed.tools
            ]},
        )
        openai_tools = mcp_to_openai_tools(listed.tools)
        training_audit = EverythingAudit(everything)
        await _seed_training_memory(training_audit, memory, )

        per_task: list[dict[str, Any]] = []
        base_calls = assisted_calls = base_pass = assisted_pass = 0
        last_guidance: dict[str, Any] = {}
        for task in TASKS:
            base_audit = EverythingAudit(everything)
            started = time.perf_counter()
            base = await _run_llm_agent(model, temperature, max_steps, SYSTEM_PROMPT, openai_tools, base_audit, task)
            base_elapsed = time.perf_counter() - started

            guidance = _value(await memory.call_tool(
                "get_guidance", {"task": "Echo a message and calculate the sum of two values", "top_k": 3, "read_budget": 8}))
            last_guidance = guidance
            assisted_prompt = SYSTEM_PROMPT + f"\nLearned playbook: {json.dumps(guidance.get('playbook', []))}"
            assisted_audit = EverythingAudit(everything)
            started = time.perf_counter()
            assisted = await _run_llm_agent(model, temperature, max_steps, assisted_prompt, openai_tools, assisted_audit, task)
            assisted_elapsed = time.perf_counter() - started

            base_m = {"agent": "baseline_without_toolatlas", "label": task["label"], "passed": base["passed"],
                      "answer": base["answer"], "expected": task["expected"], "final_text": base["final_text"][:300],
                      "llm_steps": base["steps"], "tool_calls": len(base_audit.calls),
                      "tools_used": [c["tool"] for c in base_audit.calls], "elapsed_ms": round(base_elapsed * 1000, 3)}
            assisted_m = {"agent": "agent_with_toolatlas", "label": task["label"], "passed": assisted["passed"],
                          "answer": assisted["answer"], "expected": task["expected"], "final_text": assisted["final_text"][:300],
                          "llm_steps": assisted["steps"], "tool_calls": len(assisted_audit.calls),
                          "tools_used": [c["tool"] for c in assisted_audit.calls], "elapsed_ms": round(assisted_elapsed * 1000, 3)}
            base_calls += base_m["tool_calls"]
            assisted_calls += assisted_m["tool_calls"]
            base_pass += int(base_m["passed"])
            assisted_pass += int(assisted_m["passed"])
            per_task.append({"baseline": base_m, "toolatlas": assisted_m})

    return {
        "benchmark": "everything_nim_llm_ab", "model": model, "temperature": temperature,
        "verifier": "label_present_and_sum_equals_expected",
        "everything_server": f"{EVERYTHING_PROVIDER}@{EVERYTHING_PACKAGE_VERSION}",
        "client_root": str(client_root), "training_calls_excluded": len(training_audit.calls),
        "per_task": per_task,
        "totals": {"baseline_calls": base_calls, "toolatlas_calls": assisted_calls,
                   "calls_saved": base_calls - assisted_calls,
                   "baseline_passed": f"{base_pass}/{len(TASKS)}", "toolatlas_passed": f"{assisted_pass}/{len(TASKS)}"},
        "retrieved_guidance": last_guidance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="True LLM A/B with NVIDIA NIM + live Everything MCP")
    parser.add_argument("--root", type=Path, default=Path("mcp-sandbox"))
    parser.add_argument("--memory", type=Path, default=Path(".toolatlas/everything-nim-benchmark.db"))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL") or os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    args = parser.parse_args()
    result = asyncio.run(run_everything_nim_ab(Path.cwd(), args.root, args.memory, args.model, args.temperature, args.max_steps))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
