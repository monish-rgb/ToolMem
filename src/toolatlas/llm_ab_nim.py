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

from mcp import Client, StdioServerParameters

from .filesystem_demo import (
    FILESYSTEM_PACKAGE_VERSION,
    FILESYSTEM_PROVIDER,
    filesystem_server_path,
)
from .memory_server import create_memory_server
from .readonly_benchmark import READ_ONLY_TOOLS, _extract_integer, _first_path, _value

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "moonshotai/kimi-k3"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_STEPS = 8

SYSTEM_PROMPT = (
    "You are a filesystem agent. Use only the provided tools. "
    "Answer with the exact requested setting value."
)
# Neutral baseline: no search strategy hint. Arm B gets it via memory.
TASKS = [
    {
        "task": "Find the production EU deployment policy file and report its rollout window setting",
        "field": "rollout_window_minutes",
        "expected": 20,
    },
    {
        "task": "Follow the deploy checklist runbook to find the production rollout window it references",
        "field": "rollout_window_minutes",
        "expected": 20,
    },
    {
        "task": "Find the cache policy file and report its ttl setting",
        "field": "ttl_seconds",
        "expected": 600,
    },
]
TASK = TASKS[0]["task"]
TASK_FIELD = TASKS[0]["field"]
TASK_EXPECTED = TASKS[0]["expected"]


@dataclass
class NIMReadOnlyAudit:
    client: Client
    calls: list[dict[str, Any]] = field(default_factory=list)
    forbidden_attempts: int = 0

    async def call(self, tool: str, arguments: dict[str, Any]) -> Any:
        if tool not in READ_ONLY_TOOLS:
            self.forbidden_attempts += 1
            raise PermissionError(f"write-capable MCP tool blocked: {tool}")
        started = time.perf_counter()
        result = await self.client.call_tool(tool, arguments)
        self.calls.append(
            {
                "tool": tool,
                "is_error": result.is_error,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        return result


def mcp_to_openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    openai_tools: list[dict[str, Any]] = []
    for tool in mcp_tools:
        if tool.name not in READ_ONLY_TOOLS:
            continue
        schema = tool.input_schema or {"type": "object", "properties": {}}
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return openai_tools


def _parse_answer_int(text: str) -> int | None:
    match = re.search(r"(-?\d+)", text or "")
    return int(match.group(1)) if match else None


def _openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("pip install openai>=1.0 to run the NIM A/B harness") from exc
    base_url = os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL)
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("set $env:NVIDIA_API_KEY before running the NIM A/B harness")
    return OpenAI(base_url=base_url, api_key=api_key)


async def _seed_training_memory(audit: NIMReadOnlyAudit, memory: Client, root: Path) -> None:
    training = [
        ("retry", "retry-policy.md", "max_attempts", 5),
        ("timeout", "timeout-policy.md", "request_timeout_seconds", 30),
    ]
    for label, filename, field_name, expected in training:
        search = await audit.call("search_files", {"path": str(root), "pattern": f"**/{filename}"})
        target = _first_path(search)
        read = await audit.call("read_text_file", {"path": target})
        actual = _extract_integer(str(_value(read)), field_name)
        await memory.call_tool(
            "remember_execution",
            {
                "task_id": f"readonly_{label}_policy",
                "summary": f"Find a {label} policy file and report its configured setting",
                "steps": [
                    {"tool": "search_files", "rationale": "locate the requested policy file recursively"},
                    {"tool": "read_text_file", "rationale": "read the matching policy file and extract the requested setting"},
                ],
                "resolved": actual == expected,
                "observation": "the requested setting matched the exact expected value",
                "verifier_type": "exact_field_match",
            },
        )
        if actual != expected:
            raise AssertionError(f"training verifier failed for {filename}")


async def _run_llm_agent(
    model: str,
    temperature: float,
    max_steps: int,
    system_prompt: str,
    openai_tools: list[dict[str, Any]],
    audit: NIMReadOnlyAudit,
    root: Path,
    task: str,
    field: str,
) -> dict[str, Any]:
    client = _openai_client()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{task}. Workspace root: {root}. Report only the integer for {field}.",
        },
    ]
    steps = 0
    transcript: list[dict[str, Any]] = []
    while steps < max_steps:
        steps += 1
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model=model,
            messages=messages,
            tools=openai_tools or None,
            tool_choice="auto" if openai_tools else "none",
            temperature=temperature,
            max_tokens=512,
        )
        choice = response.choices[0].message
        transcript.append({"role": "assistant", "content": choice.content, "tool_calls": bool(choice.tool_calls)})
        if not choice.tool_calls:
            answer = _parse_answer_int(choice.content or "")
            return {"final_text": choice.content or "", "answer": answer, "steps": steps, "transcript": transcript}
        messages.append(
            {
                "role": "assistant",
                "content": choice.content,
                "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in choice.tool_calls
                ],
            }
        )
        for tc in choice.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = await audit.call(name, args)
                payload = str(_value(result))
                if result.is_error:
                    payload = f"MCP tool error: {payload}"
            except PermissionError as exc:
                payload = f"Blocked: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": payload[:8000]})
    return {"final_text": "", "answer": None, "steps": steps, "transcript": transcript}


async def run_nim_ab(
    project_root: Path,
    memory_path: Path,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_steps: int = DEFAULT_MAX_STEPS,
    workspace: str = "complex_workspace",
) -> dict[str, Any]:
    project_root = project_root.resolve()
    root = project_root / "tests" / "fixtures" / workspace
    server_js = filesystem_server_path(project_root)
    if not server_js.is_file():
        raise FileNotFoundError("run npm install before the NIM A/B test")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required")
    parameters = StdioServerParameters(command=node, args=[str(server_js), str(root)])
    memory_server = create_memory_server(memory_path)

    async with Client(parameters) as filesystem, Client(memory_server) as memory:
        listed = await filesystem.list_tools()
        read_specs = [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": tool.input_schema,
                "provider": FILESYSTEM_PROVIDER,
                "version": FILESYSTEM_PACKAGE_VERSION,
            }
            for tool in listed.tools
            if tool.name in READ_ONLY_TOOLS
        ]
        await memory.call_tool("register_tools", {"tools": read_specs})
        openai_tools = mcp_to_openai_tools(listed.tools)

        training_audit = NIMReadOnlyAudit(filesystem)
        await _seed_training_memory(training_audit, memory, root)

        per_task: list[dict[str, Any]] = []
        total_baseline_calls = 0
        total_assisted_calls = 0
        baseline_pass = 0
        assisted_pass = 0
        last_guidance: dict[str, Any] = {}

        def metrics(name: str, task_def: dict[str, Any], run: dict[str, Any], audit: NIMReadOnlyAudit, elapsed: float) -> dict[str, Any]:
            return {
                "agent": name,
                "task": task_def["task"],
                "passed": run["answer"] == task_def["expected"],
                "answer": run["answer"],
                "expected": task_def["expected"],
                "final_text": run["final_text"][:500],
                "llm_steps": run["steps"],
                "filesystem_tool_calls": len(audit.calls),
                "tools_used": [c["tool"] for c in audit.calls],
                "forbidden_write_attempts": audit.forbidden_attempts,
                "elapsed_ms": round(elapsed * 1000, 3),
            }

        for task_def in TASKS:
            # Arm A: filesystem read tools only.
            baseline_audit = NIMReadOnlyAudit(filesystem)
            started = time.perf_counter()
            baseline = await _run_llm_agent(
                model, temperature, max_steps, SYSTEM_PROMPT, openai_tools,
                baseline_audit, root, task_def["task"], task_def["field"],
            )
            baseline_elapsed = time.perf_counter() - started

            # Arm B: get_guidance once before acting, then same read tools.
            guidance_result = await memory.call_tool(
                "get_guidance", {"task": task_def["task"], "top_k": 2, "read_budget": 6}
            )
            guidance = _value(guidance_result)
            last_guidance = guidance
            playbook = guidance.get("playbook", [])
            assisted_prompt = SYSTEM_PROMPT + f"\nLearned playbook for this task: {json.dumps(playbook)}"
            toolatlas_audit = NIMReadOnlyAudit(filesystem)
            started = time.perf_counter()
            assisted = await _run_llm_agent(
                model, temperature, max_steps, assisted_prompt, openai_tools,
                toolatlas_audit, root, task_def["task"], task_def["field"],
            )
            toolatlas_elapsed = time.perf_counter() - started

            baseline_m = metrics("baseline_without_toolatlas", task_def, baseline, baseline_audit, baseline_elapsed)
            assisted_m = metrics("agent_with_toolatlas", task_def, assisted, toolatlas_audit, toolatlas_elapsed)
            total_baseline_calls += baseline_m["filesystem_tool_calls"]
            total_assisted_calls += assisted_m["filesystem_tool_calls"]
            baseline_pass += int(baseline_m["passed"])
            assisted_pass += int(assisted_m["passed"])
            per_task.append({"baseline": baseline_m, "toolatlas": assisted_m})
    return {
        "benchmark": "nim_llm_filesystem_ab",
        "model": model,
        "temperature": temperature,
        "tasks": [t["task"] for t in TASKS],
        "verifier": "model_reported_integer_equals_expected",
        "filesystem_server": f"{FILESYSTEM_PROVIDER}@{FILESYSTEM_PACKAGE_VERSION}",
        "root": str(root),
        "workspace": workspace,
        "training_calls_excluded": len(training_audit.calls),
        "per_task": per_task,
        "totals": {
            "baseline_calls": total_baseline_calls,
            "toolatlas_calls": total_assisted_calls,
            "calls_saved": total_baseline_calls - total_assisted_calls,
            "baseline_passed": f"{baseline_pass}/{len(TASKS)}",
            "toolatlas_passed": f"{assisted_pass}/{len(TASKS)}",
        },
        "comparison": {
            "both_passed": baseline_pass == len(TASKS) and assisted_pass == len(TASKS),
            "filesystem_calls_saved": total_baseline_calls - total_assisted_calls,
        },
        "retrieved_guidance": last_guidance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="True LLM A/B with NVIDIA NIM + live Filesystem MCP")
    parser.add_argument("--memory", type=Path, default=Path(".toolatlas/nim-benchmark.db"))
    parser.add_argument("--model", default=os.environ.get("NVIDIA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--workspace", default="complex_workspace")
    args = parser.parse_args()
    result = asyncio.run(run_nim_ab(Path.cwd(), args.memory, args.model, args.temperature, args.max_steps, args.workspace))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
