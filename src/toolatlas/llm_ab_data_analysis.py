"""Synthetic Sales Dataset Analysis MCP A/B Benchmark Harness.

Covers data analysis over Filesystem MCP:
1. Generates a deterministic synthetic sales dataset ('sales_analytics.csv', 100 rows).
2. Computes ground-truth quantitative metrics (total completed revenue, top region, refund count, avg discount rate).
3. Evaluates Baseline (14 unpruned tools, unguided) vs ToolAtlas (playbook-guided, pruned schema).
4. Programmatically verifies the resulting 'report.md' for numerical and categorical accuracy.

Supports both Google Gemini REST API and OpenAI-compatible providers (Moonshot Kimi-k3 via NVIDIA NIM).
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .filesystem_demo import filesystem_server_path
from .gemini_rest import GeminiRestClient, provider_name
from .guidance_render import estimate_tokens as estimate_guidance_tokens
from .guidance_render import render_compact_block
from .llm_ab_nim import _openai_client
from .memory_server import create_memory_server
from .readonly_benchmark import _value
from .result_limit import truncate_result
from .tool_filter import estimate_schema_tokens, filter_tools_by_playbook

SYSTEM_PROMPT = (
    "You are a quantitative data analyst agent with Filesystem MCP tools. "
    "Analyze the dataset provided in your workspace root, compute the requested business metrics, "
    "and call the 'write_file' tool to save the executive report to 'report.md'. "
    "Rely only on the provided tools and ensure you write the file to disk."
)

TASK_SUMMARY = "Analyze sales analytics dataset, compute key business metrics, and write executive report"
TASK_DESCRIPTION = (
    "Analyze the dataset 'sales_analytics.csv' located in the workspace root. "
    "Calculate the following 4 key business metrics:\n"
    "1. Total Completed Revenue: Total revenue from 'Completed' transactions only (exclude Refunded and Pending).\n"
    "2. Top Region: The geographic region with the highest total completed revenue.\n"
    "3. Refunded Transactions: The total count of transactions with status 'Refunded'.\n"
    "4. Average Discount Rate: The mean discount rate across all transactions as a percentage.\n\n"
    "You MUST invoke the 'write_file' tool to write this executive report to 'report.md'."
)

PLAYBOOK = [
    {"tool": "read_file", "rationale": "read sales_analytics.csv to inspect columns and transaction data"},
    {"tool": "write_file", "rationale": "write the computed business metrics and executive summary to report.md"},
]
AVOID_NOTES = [
    {"tool": "list_directory", "caution": "sales_analytics.csv is already known in the root directory"},
    {"tool": "directory_tree", "caution": "do not scan recursive directory tree for a single root file"},
]


@dataclass
class CallLog:
    calls: list[str] = field(default_factory=list)
    failed_provider_calls: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_usage: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _content(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    return "\n".join(str(getattr(block, "text", "")) for block in blocks)


def generate_dataset(root: Path, n_rows: int = 100, seed: int = 42) -> dict[str, Any]:
    """Deterministically generate sales_analytics.csv and calculate ground-truth metrics."""
    rng = random.Random(seed)
    csv_path = root / "sales_analytics.csv"

    segments = ["Enterprise", "SMB", "Consumer"]
    regions = ["North America", "Europe", "Asia-Pacific", "Latin America"]
    categories = ["Cloud Software", "Hardware", "Consulting Services"]
    statuses = ["Completed", "Refunded", "Pending"]
    status_weights = [0.75, 0.15, 0.10]

    rows = []
    completed_rev = 0.0
    region_rev: dict[str, float] = {r: 0.0 for r in regions}
    refunded_count = 0
    discount_sum = 0.0

    for i in range(1, n_rows + 1):
        txn_id = f"TXN-{1000 + i}"
        date = f"2026-0{rng.randint(1, 3)}-{rng.randint(10, 28)}"
        segment = rng.choice(segments)
        region = rng.choice(regions)
        category = rng.choice(categories)
        revenue = round(rng.uniform(500.0, 5000.0), 2)
        discount = round(rng.uniform(0.05, 0.25), 2)
        status = rng.choices(statuses, weights=status_weights)[0]

        discount_sum += discount
        if status == "Completed":
            completed_rev += revenue
            region_rev[region] += revenue
        elif status == "Refunded":
            refunded_count += 1

        rows.append({
            "transaction_id": txn_id,
            "date": date,
            "customer_segment": segment,
            "region": region,
            "product_category": category,
            "revenue": revenue,
            "discount_rate": discount,
            "status": status,
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    top_region = max(region_rev.items(), key=lambda x: x[1])[0]
    avg_discount = round((discount_sum / n_rows) * 100.0, 2)
    completed_rev = round(completed_rev, 2)

    return {
        "total_rows": n_rows,
        "completed_revenue": completed_rev,
        "top_region": top_region,
        "region_breakdown": {k: round(v, 2) for k, v in region_rev.items()},
        "refunded_count": refunded_count,
        "avg_discount_rate": avg_discount,
    }


def verify_analysis_report(root: Path, ground_truth: dict[str, Any], final_text: str = "") -> tuple[bool, str]:
    """Verify that report.md was generated with accurate business metrics."""
    report_file = root / "report.md"
    text = ""
    if report_file.exists():
        text = report_file.read_text(encoding="utf-8", errors="replace")
    elif final_text and len(final_text.strip()) >= 80:
        text = final_text
        report_file.write_text(final_text, encoding="utf-8")
    else:
        return False, "report.md does not exist and no report found in final response"

    if len(text.strip()) < 80:
        return False, f"report content is too short ({len(text)} chars)"

    errors = []

    # 1. Check Completed Revenue (allow +-2% tolerance)
    expected_rev = ground_truth["completed_revenue"]
    rev_numbers = re.findall(r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)", text)
    found_rev = False
    for num_str in rev_numbers:
        clean = num_str.replace(",", "")
        try:
            val = float(clean)
            if abs(val - expected_rev) / expected_rev <= 0.02:
                found_rev = True
                break
        except ValueError:
            continue
    if not found_rev:
        errors.append(f"completed_revenue mismatch: expected ~${expected_rev:,.2f}")

    # 2. Check Top Region
    expected_region = ground_truth["top_region"]
    if expected_region.lower() not in text.lower():
        errors.append(f"top_region mismatch: expected '{expected_region}'")

    # 3. Check Refunded Count
    expected_refunds = ground_truth["refunded_count"]
    if str(expected_refunds) not in text:
        errors.append(f"refunded_count mismatch: expected {expected_refunds}")

    # 4. Check Avg Discount Rate (allow +-1.5% tolerance)
    expected_discount = ground_truth["avg_discount_rate"]
    pct_numbers = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", text)
    found_discount = False
    for p_str in pct_numbers:
        try:
            val = float(p_str)
            if abs(val - expected_discount) <= 1.5:
                found_discount = True
                break
        except ValueError:
            continue
    if not found_discount:
        errors.append(f"avg_discount_rate mismatch: expected ~{expected_discount}%")

    passed = len(errors) == 0
    msg = "All 4 metrics verified successfully" if passed else "; ".join(errors)
    return passed, msg


async def seed_analysis_memory(memory: Client) -> None:
    """Ingest verified data analysis workflow into ToolAtlas memory."""
    for run_id in (1, 2):
        await memory.call_tool(
            "remember_execution",
            {
                "task_id": f"sales_analysis_train_{run_id}",
                "summary": TASK_SUMMARY,
                "steps": PLAYBOOK,
                "resolved": True,
                "observation": "Successfully computed metrics and verified report.md",
                "verifier_type": "programmatic_data_verifier",
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
                    payload = _content(result) or str(_value(result))
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
                payload = _content(result) or str(_value(result))
                if result.is_error:
                    log.failed_provider_calls.append(tc.function.name)
                    payload = f"MCP tool error: {payload}"
            except Exception as exc:
                log.calls.append(tc.function.name)
                log.failed_provider_calls.append(tc.function.name)
                payload = f"MCP tool error: {exc}"
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": truncate_result(payload, result_limit)})
    return log, ""


async def eval_analysis_attempt(
    arm: str,
    attempt: int,
    model: str,
    temperature: float,
    frozen_db: Path,
    out_root: Path,
    guidance_cap: int = 384,
    result_limit: int = 8000,
) -> dict[str, Any]:
    work_dir = out_root / f"analysis-{arm}-k{attempt}"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    ground_truth = generate_dataset(work_dir, n_rows=100, seed=42)

    node = shutil.which("node")
    server_path = filesystem_server_path(Path.cwd()).resolve()
    fs_params = StdioServerParameters(command=node, args=[str(server_path), str(work_dir)])

    memory_server = create_memory_server(frozen_db, read_only=True)
    memory_calls = 0
    injected_guidance = None
    guidance_text = ""
    started = time.perf_counter()

    async with Client(fs_params) as client, Client(memory_server) as memory:
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
                    {"task": TASK_SUMMARY, "top_k": 2, "read_budget": 6, "token_budget": guidance_cap},
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
            model, temperature, 10, prompt, agent_tools, client, TASK_DESCRIPTION, result_limit=result_limit
        )

    passed, verifier_msg = verify_analysis_report(work_dir, ground_truth, final_text=final_text)
    return {
        "task": "sales_dataset_analysis",
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
        "ground_truth": ground_truth,
        "guidance": injected_guidance,
    }


async def main_async(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.output).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    train_db = out_root / "train-memory.db"
    frozen_db = out_root / "frozen-memory.db"

    # Train & Freeze Memory
    memory_server = create_memory_server(train_db)
    node = shutil.which("node")
    server_path = filesystem_server_path(Path.cwd()).resolve()
    dummy_dir = out_root / "probe-dir"
    dummy_dir.mkdir(parents=True, exist_ok=True)
    fs_params = StdioServerParameters(command=node, args=[str(server_path), str(dummy_dir)])

    async with Client(memory_server) as memory:
        async with Client(fs_params) as probe:
            listed = await probe.list_tools()
        await memory.call_tool(
            "register_tools",
            {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "input_schema": t.input_schema,
                        "provider": "io.github.modelcontextprotocol/server-filesystem",
                        "version": "2026.8.31",
                    }
                    for t in listed.tools
                ]
            },
        )
        await seed_analysis_memory(memory)

    from .freeze_memory import freeze_memory

    freeze_memory(train_db, frozen_db)

    results: list[dict[str, Any]] = []
    for arm in ("baseline", "toolatlas"):
        for attempt in range(1, args.k + 1):
            print(f"[sales_dataset_analysis | {arm} | attempt {attempt}/{args.k}] running...", flush=True)
            rec = await eval_analysis_attempt(
                arm, attempt, args.model, args.temperature, frozen_db, out_root, guidance_cap=args.guidance_cap
            )
            results.append(rec)

    def summarize(arm: str) -> dict[str, Any]:
        rows = [r for r in results if r["arm"] == arm]
        passed = sum(1 for r in rows if r["passed"])
        calls = [r["provider_calls"] for r in rows]
        tokens = [r["total_tokens"] for r in rows]
        return {
            "task": "sales_dataset_analysis",
            "arm": arm,
            "passed": f"{passed}/{len(rows)}",
            "avg_provider_calls": round(sum(calls) / len(calls), 2) if calls else 0,
            "avg_tokens": round(sum(tokens) / len(tokens), 1) if tokens else 0,
        }

    summary = [summarize(a) for a in ("baseline", "toolatlas")]
    report = {
        "benchmark": "filesystem_dataset_analysis_llm_ab",
        "model": args.model,
        "k": args.k,
        "summary": summary,
        "results": results,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Filesystem Dataset Analysis A/B Benchmark")
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
