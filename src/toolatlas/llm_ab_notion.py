"""Notion MCP Both-Arms A/B Benchmark Harness.

Covers two high-impact Notion workspace tasks:
1. notion_meeting_notes: Locate team workspace, create meeting notes page, and append typed agenda blocks.
2. notion_task_triage: Search task tracker database, query high-priority tasks, and update status.

Evaluates token consumption, tool schema pruning, and turn elimination across both arms.
Supports both Google Gemini REST API and OpenAI-compatible models (Moonshot, DeepSeek, OpenAI).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client

from .gemini_rest import GeminiRestClient, provider_name
from .guidance_render import estimate_tokens as estimate_guidance_tokens
from .guidance_render import render_compact_block
from .llm_ab_nim import _openai_client
from .memory_server import create_memory_server
from .notion_server import (
    NOTION_PROVIDER,
    NOTION_VERSION,
    NotionWorkspaceSimulator,
    create_notion_server,
)
from .readonly_benchmark import _value
from .result_limit import truncate_result
from .tool_filter import estimate_schema_tokens, filter_tools_by_playbook

NOTION_TASKS = {
    "notion_meeting_notes": {
        "summary": "Locate team workspace, create meeting notes page, and append typed agenda blocks",
        "description": (
            "Search for the 'Engineering Team Workspace' page. Create a new sub-page under it titled "
            "'Sprint Retro Notes' using create_page with parent={'page_id': ...}. Then call append_block_children "
            "to add a paragraph block with rich_text containing 'Retro completed successfully'."
        ),
        "playbook": [
            {"tool": "search", "rationale": "search for the parent team workspace page"},
            {"tool": "get_page", "rationale": "verify parent page ID and properties"},
            {"tool": "create_page", "rationale": "create sub-page with valid parent page_id"},
            {"tool": "append_block_children", "rationale": "append typed block objects to the page"},
        ],
        "avoid": [
            {"tool": "create_page", "caution": "parent must be an object with valid page_id from search"},
            {"tool": "append_block_children", "caution": "children must be an array of typed block objects"},
        ],
    },
    "notion_task_triage": {
        "summary": "Search task tracker database, query high-priority tasks, and update status",
        "description": (
            "Search for the 'Project Task Tracker' database. Query tasks using query_database. "
            "Update the status of the 'Fix database connection leak' task to 'In Progress' using update_page_properties."
        ),
        "playbook": [
            {"tool": "search", "rationale": "search for the task tracker database"},
            {"tool": "query_database", "rationale": "query database tasks and filter by priority"},
            {"tool": "update_page_properties", "rationale": "update task status property"},
        ],
        "avoid": [
            {"tool": "get_page", "caution": "do not call get_page on database IDs; use query_database"},
        ],
    },
}

SYSTEM_PROMPT = (
    "You are a Notion workspace agent with MCP tools. "
    "Complete the user's task in Notion using only the provided tools. "
    "Ensure parent objects and typed block payloads follow valid Notion API structure. "
    "Reply with a short summary when done."
)


@dataclass
class CallLog:
    calls: list[str] = field(default_factory=list)
    failed_provider_calls: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_usage: list[dict[str, Any]] = field(default_factory=list)
    request_rows: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def verify_notion_task(task_key: str, sim: NotionWorkspaceSimulator, final_text: str, calls: list[str]) -> tuple[bool, str]:
    if task_key == "notion_meeting_notes":
        created = [p for p in sim.pages.values() if "Retro" in p.get("title", "")]
        has_page = len(created) > 0
        has_blocks = False
        if has_page:
            blocks = sim.get_block_children(created[0]["id"])
            has_blocks = len(blocks) > 0
        passed = has_page and ("create_page" in calls or "append_block_children" in calls)
        return bool(passed), f"page_created={has_page}, blocks_appended={has_blocks}, calls={calls}"

    if task_key == "notion_task_triage":
        task_page = sim.pages.get("task-page-1", {})
        status = task_page.get("properties", {}).get("Status", {}).get("select", {}).get("name", "")
        updated = "In Progress" in status or "In Progress" in final_text
        used_db = "query_database" in calls or "update_page_properties" in calls
        passed = updated and used_db
        return bool(passed), f"status={status}, used_db={used_db}, calls={calls}"

    return False, "unknown task"


async def seed_notion_memory(memory: Client) -> None:
    for key, task_def in NOTION_TASKS.items():
        await memory.call_tool(
            "remember_execution",
            {
                "task_id": f"notion_{key}_train",
                "summary": task_def["summary"],
                "steps": task_def["playbook"],
                "resolved": True,
                "observation": f"Successfully performed {key} in Notion workspace and verified structure",
                "verifier_type": "notion_verifier",
            },
        )


async def run_llm_agent(
    model: str,
    temperature: float,
    max_steps: int,
    system_prompt: str,
    openai_tools: list[dict[str, Any]],
    client: Client,
    task_text: str,
    result_limit: int = 8000,
) -> tuple[CallLog, str]:
    provider = provider_name()
    use_gemini = provider == "gemini"
    llm = GeminiRestClient(model=model) if use_gemini else _openai_client()
    log = CallLog()

    if use_gemini:
        contents: list[dict[str, Any]] = [{"role": "user", "parts": [{"text": task_text}]}]
        steps = 0
        while steps < max_steps:
            steps += 1
            response = await asyncio.to_thread(llm.generate, system_prompt, contents, openai_tools, temperature, 1024)
            log.prompt_tokens += response.get("prompt_tokens", 0)
            log.completion_tokens += response.get("completion_tokens", 0)
            if not response["tool_calls"]:
                return log, response["text"]
            contents.append({"role": "model", "parts": response["raw_parts"]})
            response_parts = []
            for tc in response["tool_calls"]:
                try:
                    args = json.loads(tc["args"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                try:
                    result = await client.call_tool(tc["name"], args)
                    log.calls.append(tc["name"])
                    payload = str(_value(result))
                    if result.is_error or "Notion API error" in payload:
                        log.failed_provider_calls.append(tc["name"])
                        payload = f"MCP tool error: {payload}"
                except Exception as exc:
                    log.calls.append(tc["name"])
                    log.failed_provider_calls.append(tc["name"])
                    payload = f"MCP tool error: {exc}"
                response_parts.append(
                    {"functionResponse": {"name": tc["name"], "response": {"result": truncate_result(payload, result_limit)}}}
                )
            contents.append({"role": "user", "parts": response_parts})
        return log, ""

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_text},
    ]
    steps = 0
    while steps < max_steps:
        steps += 1
        response = await asyncio.to_thread(
            llm.chat.completions.create,
            model=model,
            messages=messages,
            tools=openai_tools or None,
            tool_choice="auto" if openai_tools else "none",
            temperature=temperature,
            max_tokens=1024,
        )
        usage = getattr(response, "usage", None)
        if usage:
            log.prompt_tokens += getattr(usage, "prompt_tokens", 0)
            log.completion_tokens += getattr(usage, "completion_tokens", 0)
        choice = response.choices[0].message
        if not choice.tool_calls:
            return log, choice.content or ""
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
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = await client.call_tool(tc.function.name, args)
                log.calls.append(tc.function.name)
                payload = str(_value(result))
                if result.is_error or "Notion API error" in payload:
                    log.failed_provider_calls.append(tc.function.name)
                    payload = f"MCP tool error: {payload}"
            except Exception as exc:
                log.calls.append(tc.function.name)
                log.failed_provider_calls.append(tc.function.name)
                payload = f"MCP tool error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": truncate_result(payload, result_limit)})
    return log, ""


async def eval_notion_attempt(
    task_key: str,
    arm: str,
    attempt: int,
    model: str,
    temperature: float,
    frozen_db: Path,
    guidance_cap: int = 384,
    result_limit: int = 8000,
) -> dict[str, Any]:
    sim = NotionWorkspaceSimulator()
    notion_server = create_notion_server(sim)
    memory_server = create_memory_server(frozen_db, read_only=True)
    memory_calls = 0
    injected_guidance = None
    guidance_text = ""
    started = time.perf_counter()
    task_def = NOTION_TASKS[task_key]

    async with Client(notion_server) as client, Client(memory_server) as memory:
        listed = await client.list_tools()
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema or {"type": "object", "properties": {}},
                },
            }
            for t in listed.tools
        ]
        prompt = SYSTEM_PROMPT
        agent_tools = openai_tools
        schema_before = estimate_schema_tokens(openai_tools)
        schema_after = schema_before

        if arm == "toolatlas":
            guidance = _value(
                await memory.call_tool(
                    "get_guidance",
                    {"task": task_def["summary"], "top_k": 2, "read_budget": 6, "token_budget": guidance_cap},
                )
            )
            memory_calls += 1
            injected_guidance = guidance.get("playbook", [])
            guidance_text = render_compact_block(guidance, token_budget=guidance_cap)
            if guidance_text:
                prompt += f"\n{guidance_text}"
            agent_tools, _ = filter_tools_by_playbook(openai_tools, guidance)
            schema_after = estimate_schema_tokens(agent_tools)

        log, final_text = await run_llm_agent(
            model, temperature, 15, prompt, agent_tools, client, task_def["description"], result_limit=result_limit
        )

    passed, verifier_msg = verify_notion_task(task_key, sim, final_text, log.calls)
    return {
        "task": task_key,
        "arm": arm,
        "attempt": attempt,
        "passed": passed,
        "verifier_msg": verifier_msg,
        "provider_calls": len(log.calls),
        "failed_provider_calls": len(log.failed_provider_calls),
        "memory_calls": memory_calls,
        "total_mcp_calls": len(log.calls) + memory_calls,
        "tools_used": log.calls,
        "prompt_tokens": log.prompt_tokens,
        "completion_tokens": log.completion_tokens,
        "total_tokens": log.total_tokens,
        "schema_tokens_before": schema_before,
        "schema_tokens_after": schema_after,
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "guidance": injected_guidance,
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    train_db = out_root / "train-memory.db"
    frozen_db = out_root / "frozen-memory.db"

    # Train & Freeze
    memory_server = create_memory_server(train_db)
    sim = NotionWorkspaceSimulator()
    notion_server = create_notion_server(sim)

    async with Client(memory_server) as memory:
        async with Client(notion_server) as probe:
            listed = await probe.list_tools()
        await memory.call_tool(
            "register_tools",
            {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.input_schema,
                        "provider": NOTION_PROVIDER,
                        "version": NOTION_VERSION,
                    }
                    for t in listed.tools
                ]
            },
        )
        await seed_notion_memory(memory)
    from .freeze_memory import freeze_memory

    freeze_memory(train_db, frozen_db)

    results: list[dict[str, Any]] = []
    for task_key in NOTION_TASKS:
        for arm in ("baseline", "toolatlas"):
            for attempt in range(1, args.k + 1):
                print(f"[{task_key} | {arm} | attempt {attempt}/{args.k}] running...", flush=True)
                rec = await eval_notion_attempt(
                    task_key, arm, attempt, args.model, args.temperature, frozen_db, guidance_cap=args.guidance_cap
                )
                results.append(rec)

    # Summary
    def summarize(task: str, arm: str) -> dict[str, Any]:
        rows = [r for r in results if r["task"] == task and r["arm"] == arm]
        passed = sum(1 for r in rows if r["passed"])
        calls = [r["provider_calls"] for r in rows]
        tokens = [r["total_tokens"] for r in rows]
        return {
            "task": task,
            "arm": arm,
            "passed": f"{passed}/{len(rows)}",
            "avg_provider_calls": round(sum(calls) / len(calls), 2) if calls else 0,
            "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
        }

    summary = [summarize(t, a) for t in NOTION_TASKS for a in ("baseline", "toolatlas")]
    report = {
        "benchmark": "notion_workspace_llm_ab",
        "model": args.model,
        "k": args.k,
        "summary": summary,
        "results": results,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Notion Workspace A/B Benchmark")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "gemini-2.5-flash"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--guidance-cap", type=int, default=384)
    args = parser.parse_args()
    report = asyncio.run(main_async(args))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
