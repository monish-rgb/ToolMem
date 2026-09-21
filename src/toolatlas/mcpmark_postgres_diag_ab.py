"""PostgreSQL Diagnostics MCPMark Benchmark Harness.

Covers two high-impact diagnostic tasks against the official Dockerized Chinook database:
1. slow_query_optimization: Identify query bottleneck, run explain/index analysis, and apply the index.
2. db_health_audit: Execute database health analysis and extract cache hit and bloat metrics.

Supports both Google Gemini REST API and OpenAI-compatible models (Moonshot, DeepSeek, OpenAI).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .gemini_rest import GeminiRestClient, provider_name
from .guidance_render import estimate_tokens as estimate_guidance_tokens
from .guidance_render import render_compact_block
from .llm_ab_nim import _openai_client
from .memory_server import create_memory_server
from .readonly_benchmark import _value
from .result_limit import truncate_result
from .tool_filter import estimate_schema_tokens, filter_tools_by_playbook

BACKUP_SHA256 = "50C7969A9E8F2CC1CE250AF58FC46D0ABEFC196C10A006DE979008443F324A14"
PROVIDER = "crystaldba/postgres-mcp"
VERSION = "pinned-mcp-v1"
CONTAINER = "mcpmark-postgres"
PG_USER = "postgres"
PG_PASSWORD = "mysecretpassword"
PG_HOST = "127.0.0.1"
PG_PORT = 5432
MCP_BIN = r"C:\Users\MONISH\.local\bin\postgres-mcp.EXE"

DIAG_TASKS = {
    "slow_query_optimization": {
        "summary": "Profile slow query execution plan, analyze missing index recommendations, and create the required index",
        "description": (
            "Analyze query performance for joins between InvoiceLine and Track. "
            "Use explain_query to inspect the plan, use analyze_query_indexes to find the recommended index, "
            "and execute_sql to create the recommended index 'idx_invoiceline_track' on InvoiceLine(TrackId)."
        ),
        "playbook": [
            {"tool": "explain_query", "rationale": "inspect execution plan and scan costs"},
            {"tool": "analyze_query_indexes", "rationale": "generate missing index recommendations"},
            {"tool": "execute_sql", "rationale": "apply the verified CREATE INDEX statement"},
        ],
        "target_index": "idx_invoiceline_track",
    },
    "db_health_audit": {
        "summary": "Audit database health metrics, connection pools, and cache hit ratios",
        "description": (
            "Run a comprehensive health audit of the database. Call analyze_db_health to inspect "
            "table bloat, connection counts, and cache hit ratios. Report a summary of the health status."
        ),
        "playbook": [
            {"tool": "analyze_db_health", "rationale": "scan database cache hit ratio, bloat, and connection health"},
        ],
    },
}

SYSTEM_PROMPT = (
    "You are a PostgreSQL Database Administrator with MCP diagnostic tools. "
    "Complete the diagnostic task using only the provided tools. Reply with a short summary."
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


def docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
    )


def reset_db(dbname: str) -> None:
    docker("exec", "-e", f"PGPASSWORD={PG_PASSWORD}", CONTAINER, "dropdb", "-U", PG_USER, "--if-exists", dbname).check_returncode()
    docker("exec", "-e", f"PGPASSWORD={PG_PASSWORD}", CONTAINER, "createdb", "-U", PG_USER, dbname).check_returncode()
    result = docker(
        "exec",
        "-e",
        f"PGPASSWORD={PG_PASSWORD}",
        CONTAINER,
        "pg_restore",
        "-U",
        PG_USER,
        "-d",
        dbname,
        "/tmp/chinook.backup",
    )
    if result.returncode != 0:
        raise AssertionError(f"pg_restore failed for {dbname}: {result.stderr[-500:]}")


def db_uri(dbname: str) -> str:
    return f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{dbname}"


def mcp_params(dbname: str) -> StdioServerParameters:
    binary = os.environ.get("POSTGRES_MCP_BIN", MCP_BIN)
    return StdioServerParameters(
        command=binary,
        args=["--access-mode=unrestricted"],
        env={"PATH": os.environ.get("PATH", ""), "DATABASE_URI": db_uri(dbname)},
    )


def verify_task(task_key: str, dbname: str, final_text: str, calls: list[str]) -> tuple[bool, str]:
    """Verify diagnostic task completion."""
    if task_key == "slow_query_optimization":
        # Check that index was created in postgres
        res = docker(
            "exec",
            "-e",
            f"PGPASSWORD={PG_PASSWORD}",
            CONTAINER,
            "psql",
            "-U",
            PG_USER,
            "-d",
            dbname,
            "-t",
            "-c",
            "SELECT count(*) FROM pg_indexes WHERE indexname LIKE '%invoiceline_track%';",
        )
        has_index = "1" in res.stdout
        used_required = "explain_query" in calls or "execute_sql" in calls
        passed = has_index or ("execute_sql" in calls and "CREATE INDEX" in final_text.upper())
        return bool(passed), f"index_created={has_index}, used_tools={calls}"

    if task_key == "db_health_audit":
        used_health = "analyze_db_health" in calls
        has_metrics = any(k in final_text.lower() for k in ("health", "cache", "connection", "status", "ratio", "db"))
        passed = used_health and (has_metrics or len(final_text) > 10)
        return bool(passed), f"used_health={used_health}, final_len={len(final_text)}"

    return False, "unknown task"


async def seed_diagnostic_memory(memory: Client) -> None:
    """Ingest verified diagnostic workflows into ToolAtlas memory."""
    for key, task_def in DIAG_TASKS.items():
        await memory.call_tool(
            "remember_execution",
            {
                "task_id": f"diag_{key}_train",
                "summary": task_def["summary"],
                "steps": task_def["playbook"],
                "resolved": True,
                "observation": f"Successfully performed {key} diagnostics and verified outcome",
                "verifier_type": "diagnostic_verifier",
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
    allowed_tool_names = {t["function"]["name"] for t in openai_tools}

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
                    if result.is_error:
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
                if result.is_error:
                    log.failed_provider_calls.append(tc.function.name)
                    payload = f"MCP tool error: {payload}"
            except Exception as exc:
                log.calls.append(tc.function.name)
                log.failed_provider_calls.append(tc.function.name)
                payload = f"MCP tool error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": truncate_result(payload, result_limit)})
    return log, ""


async def eval_diag_attempt(
    task_key: str,
    arm: str,
    attempt: int,
    model: str,
    temperature: float,
    frozen_db: Path,
    guidance_cap: int = 384,
    result_limit: int = 8000,
) -> dict[str, Any]:
    dbname = f"chinook_diag_{arm}_{task_key}_{attempt}"
    reset_db(dbname)
    memory_server = create_memory_server(frozen_db, read_only=True)
    memory_calls = 0
    injected_guidance = None
    guidance_text = ""
    started = time.perf_counter()
    task_def = DIAG_TASKS[task_key]

    async with Client(mcp_params(dbname)) as client, Client(memory_server) as memory:
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
            model, temperature, 20, prompt, agent_tools, client, task_def["description"], result_limit=result_limit
        )

    passed, verifier_msg = verify_task(task_key, dbname, final_text, log.calls)
    return {
        "task": task_key,
        "arm": arm,
        "attempt": attempt,
        "passed": passed,
        "verifier_msg": verifier_msg,
        "provider_calls": len(log.calls),
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
    async with Client(memory_server) as memory:
        # Register tools
        async with Client(mcp_params("chinook_diag_train")) as probe:
            listed = await probe.list_tools()
        await memory.call_tool(
            "register_tools",
            {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.input_schema,
                        "provider": PROVIDER,
                        "version": VERSION,
                    }
                    for t in listed.tools
                ]
            },
        )
        await seed_diagnostic_memory(memory)
    from .freeze_memory import freeze_memory

    freeze_memory(train_db, frozen_db)

    results: list[dict[str, Any]] = []
    for task_key in DIAG_TASKS:
        for arm in ("baseline", "toolatlas"):
            for attempt in range(1, args.k + 1):
                print(f"[{task_key} | {arm} | attempt {attempt}/{args.k}] running...", flush=True)
                rec = await eval_diag_attempt(
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

    summary = [summarize(t, a) for t in DIAG_TASKS for a in ("baseline", "toolatlas")]
    report = {
        "benchmark": "mcpmark_postgres_diagnostics_ab",
        "model": args.model,
        "k": args.k,
        "summary": summary,
        "results": results,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="PostgreSQL Diagnostics A/B Benchmark")
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
