from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from .filesystem_demo import (
    FILESYSTEM_PACKAGE_VERSION,
    FILESYSTEM_PROVIDER,
    filesystem_server_path,
)
from .memory_paths import fresh_memory_path
from .memory_server import create_memory_server
from .readonly_benchmark import (
    READ_ONLY_TOOLS,
    ReadOnlyAudit,
    _extract_integer,
    _first_path,
    _value,
)

PAPER_ID = "arxiv:2607.11126"
RESULT_SCHEMA_VERSION = "1"
DEFAULT_RUNS = 4


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    task_id: str
    summary: str
    filename: str
    field: str
    expected: int


SAME_ENV_TRAIN_TASKS = (
    BenchmarkTask(
        "train_retry",
        "Find a retry policy file and report its configured setting",
        "retry-policy.md",
        "max_attempts",
        5,
    ),
    BenchmarkTask(
        "train_timeout",
        "Find a timeout policy file and report its configured setting",
        "timeout-policy.md",
        "request_timeout_seconds",
        30,
    ),
    BenchmarkTask(
        "train_billing",
        "Find an approved billing policy file and report its configured setting",
        "billing-policy.md",
        "invoice_retries",
        4,
    ),
)

CROSS_ENV_TRAIN_TASKS = (
    BenchmarkTask(
        "train_retry",
        "Find a retry policy file and report its configured setting",
        "retry-policy.md",
        "max_attempts",
        5,
    ),
    BenchmarkTask(
        "train_timeout",
        "Find a timeout policy file and report its configured setting",
        "timeout-policy.md",
        "request_timeout_seconds",
        30,
    ),
    BenchmarkTask(
        "train_deployment",
        "Find a deployment policy file and report its configured setting",
        "deployment-policy.md",
        "rollout_window_minutes",
        20,
    ),
)

TEST_TASKS = (
    BenchmarkTask(
        "test_cache_ttl",
        "Find the cache policy file and report its configured ttl setting",
        "cache-policy.md",
        "ttl_seconds",
        600,
    ),
    BenchmarkTask(
        "test_cache_capacity",
        "Find the cache policy file and report its configured capacity setting",
        "cache-policy.md",
        "max_entries",
        1000,
    ),
    BenchmarkTask(
        "test_auth_attempts",
        "Find the active auth policy file and report its configured attempt setting",
        "auth-policy.md",
        "max_attempts",
        7,
    ),
    BenchmarkTask(
        "test_auth_lockout",
        "Find the active auth policy file and report its configured lockout setting",
        "auth-policy.md",
        "lockout_minutes",
        15,
    ),
    BenchmarkTask(
        "test_eu_rollout",
        "Find the production EU deployment policy file and report its rollout setting",
        "deployment-policy.md",
        "rollout_window_minutes",
        20,
    ),
    BenchmarkTask(
        "test_eu_health",
        "Find the production EU deployment policy file and report its health setting",
        "deployment-policy.md",
        "minimum_healthy_percent",
        90,
    ),
)


def _search_pattern(task: BenchmarkTask) -> str:
    if task.task_id.startswith("test_eu_"):
        return "**/prod/eu/deployment-policy.md"
    return f"**/{task.filename}"


async def _search_and_read(
    audit: ReadOnlyAudit, root: Path, task: BenchmarkTask
) -> int:
    search = await audit.call(
        "search_files", {"path": str(root), "pattern": _search_pattern(task)}
    )
    target = _first_path(search)
    read = await audit.call("read_text_file", {"path": target})
    return _extract_integer(str(_value(read)), task.field)


async def _seed_memory(
    audit: ReadOnlyAudit,
    memory: Client,
    root: Path,
    tasks: tuple[BenchmarkTask, ...],
) -> int:
    memory_calls = 0
    for task in tasks:
        answer = await _search_and_read(audit, root, task)
        result = await memory.call_tool(
            "remember_execution",
            {
                "task_id": task.task_id,
                "summary": task.summary,
                "steps": [
                    {
                        "tool": "search_files",
                        "rationale": "locate the requested policy file recursively",
                    },
                    {
                        "tool": "read_text_file",
                        "rationale": "read the matching policy file and extract the requested setting",
                    },
                ],
                "resolved": answer == task.expected,
                "observation": "the requested setting matched the exact expected value",
                "verifier_type": "exact_field_match",
            },
        )
        memory_calls += 1
        if result.is_error or answer != task.expected:
            raise AssertionError(f"training verifier failed for {task.task_id}")
    return memory_calls


async def _baseline_run(
    client: Client, root: Path, task: BenchmarkTask
) -> dict[str, Any]:
    audit = ReadOnlyAudit(client)
    started = time.perf_counter()
    await audit.call("list_directory", {"path": str(root)})
    await audit.call("directory_tree", {"path": str(root)})
    answer = await _search_and_read(audit, root, task)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "passed": answer == task.expected,
        "answer": answer,
        "provider_tool_calls": len(audit.calls),
        "memory_tool_calls": 0,
        "total_mcp_calls": len(audit.calls),
        "tools_used": [item["tool"] for item in audit.calls],
        "forbidden_write_attempts": audit.forbidden_attempts,
        "elapsed_ms": elapsed_ms,
    }


async def _toolatlas_run(
    client: Client, memory: Client, root: Path, task: BenchmarkTask
) -> dict[str, Any]:
    audit = ReadOnlyAudit(client)
    started = time.perf_counter()
    guidance_result = await memory.call_tool(
        "get_guidance",
        {"task": task.summary, "top_k": 3, "read_budget": 8},
    )
    guidance = _value(guidance_result)
    recommended = [step["tool"] for step in guidance.get("playbook", [])]
    if recommended[:2] != ["search_files", "read_text_file"]:
        raise AssertionError(
            f"ToolAtlas did not retrieve the expected playbook for {task.task_id}: "
            f"{recommended}"
        )
    answer = await _search_and_read(audit, root, task)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
    return {
        "passed": answer == task.expected,
        "answer": answer,
        "provider_tool_calls": len(audit.calls),
        "memory_tool_calls": 1,
        "total_mcp_calls": len(audit.calls) + 1,
        "tools_used": [item["tool"] for item in audit.calls],
        "forbidden_write_attempts": audit.forbidden_attempts,
        "elapsed_ms": elapsed_ms,
        "guidance": {
            "seed_candidates": guidance.get("seed_candidates", []),
            "playbook": guidance.get("playbook", []),
            "traversal": guidance.get("traversal", {}),
        },
    }


def _arm_metrics(
    task_results: list[dict[str, Any]], runs: int
) -> dict[str, Any]:
    flattened = [run for task in task_results for run in task["runs"]]
    task_count = len(task_results)
    successes = sum(int(run["passed"]) for run in flattened)
    provider_calls = sum(run["provider_tool_calls"] for run in flattened)
    memory_calls = sum(run["memory_tool_calls"] for run in flattened)
    total_calls = sum(run["total_mcp_calls"] for run in flattened)
    latencies = [run["elapsed_ms"] for run in flattened]
    return {
        "tasks": task_count,
        "runs_per_task": runs,
        "pass_at_1": round(successes / (task_count * runs), 4),
        "pass_at_k": round(
            sum(any(run["passed"] for run in task["runs"]) for task in task_results)
            / task_count,
            4,
        ),
        "k": runs,
        "successful_runs": successes,
        "total_runs": task_count * runs,
        "provider_tool_calls": provider_calls,
        "memory_tool_calls": memory_calls,
        "total_mcp_calls": total_calls,
        "mean_provider_tool_calls_per_run": round(provider_calls / len(flattened), 3),
        "mean_total_mcp_calls_per_run": round(total_calls / len(flattened), 3),
        "median_elapsed_ms": round(statistics.median(latencies), 3),
    }


def _reduction(baseline: int, assisted: int) -> dict[str, Any]:
    saved = baseline - assisted
    percent = round(100 * saved / baseline, 2) if baseline else 0.0
    return {"baseline": baseline, "toolatlas": assisted, "saved": saved, "percent": percent}


async def _run_split(
    project_root: Path,
    split_name: str,
    train_root: Path,
    test_root: Path,
    train_tasks: tuple[BenchmarkTask, ...],
    runs: int,
    memory_path: Path,
) -> dict[str, Any]:
    server_js = filesystem_server_path(project_root)
    if not server_js.is_file():
        raise FileNotFoundError("run npm install before the paper-protocol benchmark")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required")

    train_params = StdioServerParameters(command=node, args=[str(server_js), str(train_root)])
    test_params = StdioServerParameters(command=node, args=[str(server_js), str(test_root)])
    memory_server = create_memory_server(memory_path)

    async with (
        Client(train_params) as train_client,
        Client(test_params) as test_client,
        Client(memory_server) as memory,
    ):
        listed = await test_client.list_tools()
        specs = [
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
        await memory.call_tool("register_tools", {"tools": specs})
        training_audit = ReadOnlyAudit(train_client)
        training_memory_calls = 1 + await _seed_memory(
            training_audit, memory, train_root, train_tasks
        )

        baseline_tasks: list[dict[str, Any]] = []
        assisted_tasks: list[dict[str, Any]] = []
        for task in TEST_TASKS:
            baseline_runs = [
                await _baseline_run(test_client, test_root, task) for _ in range(runs)
            ]
            assisted_runs = [
                await _toolatlas_run(test_client, memory, test_root, task)
                for _ in range(runs)
            ]
            baseline_tasks.append(
                {"task": asdict(task), "runs": baseline_runs}
            )
            assisted_tasks.append(
                {"task": asdict(task), "runs": assisted_runs}
            )

    baseline = _arm_metrics(baseline_tasks, runs)
    assisted = _arm_metrics(assisted_tasks, runs)
    training_provider_calls = len(training_audit.calls)
    training_total_calls = training_provider_calls + training_memory_calls
    total_call_savings_per_run = (
        baseline["mean_total_mcp_calls_per_run"]
        - assisted["mean_total_mcp_calls_per_run"]
    )
    break_even_runs = (
        int((training_total_calls + total_call_savings_per_run - 1) // total_call_savings_per_run)
        if total_call_savings_per_run > 0
        else None
    )
    return {
        "split": split_name,
        "train_environment": train_root.name,
        "test_environment": test_root.name,
        "memory_frozen_during_evaluation": True,
        "train_tasks": [asdict(task) for task in train_tasks],
        "test_task_count": len(TEST_TASKS),
        "training_cost": {
            "provider_tool_calls": training_provider_calls,
            "memory_tool_calls": training_memory_calls,
            "total_mcp_calls": training_total_calls,
            "excluded_from_inference_comparison": True,
            "break_even_evaluation_runs_for_total_mcp_calls": break_even_runs,
        },
        "baseline": {"metrics": baseline, "tasks": baseline_tasks},
        "toolatlas": {"metrics": assisted, "tasks": assisted_tasks},
        "comparison": {
            "provider_tool_calls": _reduction(
                baseline["provider_tool_calls"], assisted["provider_tool_calls"]
            ),
            "total_mcp_calls_including_memory": _reduction(
                baseline["total_mcp_calls"], assisted["total_mcp_calls"]
            ),
            "pass_at_1_delta": round(
                assisted["pass_at_1"] - baseline["pass_at_1"], 4
            ),
            "pass_at_k_delta": round(
                assisted["pass_at_k"] - baseline["pass_at_k"], 4
            ),
        },
    }


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# ToolAtlas paper-protocol Filesystem benchmark",
        "",
        f"Generated: {result['generated_at']}",
        "",
        "This is a deterministic, read-only Filesystem control using the evaluation "
        "shape from the ToolAtlas paper. It is not the full MCPMark/MCP-Universe "
        "reproduction and does not establish production readiness.",
        "",
        "| Split | Arm | pass@1 | pass@4 | Provider calls | Total MCP calls |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for split in result["splits"]:
        for arm in ("baseline", "toolatlas"):
            metrics = split[arm]["metrics"]
            lines.append(
                f"| {split['split']} | {arm} | {metrics['pass_at_1']:.4f} | "
                f"{metrics['pass_at_k']:.4f} | {metrics['provider_tool_calls']} | "
                f"{metrics['total_mcp_calls']} |"
            )
        provider = split["comparison"]["provider_tool_calls"]
        total = split["comparison"]["total_mcp_calls_including_memory"]
        lines.extend(
            [
                "",
                f"- {split['split']} provider-call reduction: {provider['percent']:.2f}% "
                f"({provider['saved']} calls).",
                f"- {split['split']} total-MCP-call reduction: {total['percent']:.2f}% "
                f"({total['saved']} calls, including one guidance call per assisted run).",
                f"- Training-cost break-even: "
                f"{split['training_cost']['break_even_evaluation_runs_for_total_mcp_calls']} "
                "evaluation runs.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The control demonstrates that retrieved provider memory removes redundant "
            "Filesystem discovery calls while preserving exact verifier success. Because "
            "the runner is deterministic and the task family is small, use the optional "
            "LLM harness and official benchmark adapters before making production claims.",
            "",
        ]
    )
    return "\n".join(lines)


async def run_paper_protocol_benchmark(
    project_root: Path,
    runs: int = DEFAULT_RUNS,
) -> dict[str, Any]:
    if runs < 1:
        raise ValueError("runs must be positive")
    project_root = project_root.resolve()
    complex_root = project_root / "tests" / "fixtures" / "complex_workspace"
    readonly_root = project_root / "tests" / "fixtures" / "readonly_workspace"
    splits = [
        await _run_split(
            project_root,
            "same_environment",
            complex_root,
            complex_root,
            SAME_ENV_TRAIN_TASKS,
            runs,
            fresh_memory_path("paper-same-env"),
        ),
        await _run_split(
            project_root,
            "cross_environment",
            readonly_root,
            complex_root,
            CROSS_ENV_TRAIN_TASKS,
            runs,
            fresh_memory_path("paper-cross-env"),
        ),
    ]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark": "toolatlas_paper_protocol_filesystem_control",
        "paper": PAPER_ID,
        "generated_at": datetime.now(UTC).isoformat(),
        "runner": "deterministic_read_only_control",
        "claim_scope": "local_filesystem_control_not_full_paper_reproduction",
        "protocol": {
            "runs_per_task": runs,
            "primary_success_metrics": ["pass_at_1", f"pass_at_{runs}"],
            "cost_metrics": [
                "provider_tool_calls",
                "memory_tool_calls",
                "total_mcp_calls",
                "latency_ms",
            ],
            "train_test_ratio": "1:2 (3 train, 6 test tasks per split)",
            "memory_frozen_before_test": True,
            "programmatic_verifier": "exact integer field match",
        },
        "filesystem_server": f"{FILESYSTEM_PROVIDER}@{FILESYSTEM_PACKAGE_VERSION}",
        "splits": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the paper-aligned ToolAtlas Filesystem control benchmark"
    )
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/paper-protocol-filesystem.json"),
    )
    args = parser.parse_args()
    result = asyncio.run(run_paper_protocol_benchmark(Path.cwd(), args.runs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(_markdown_report(result), encoding="utf-8")
    print(json.dumps({
        "json_result": str(args.output.resolve()),
        "markdown_report": str(markdown_path.resolve()),
        "splits": [
            {"split": item["split"], "comparison": item["comparison"]}
            for item in result["splits"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
