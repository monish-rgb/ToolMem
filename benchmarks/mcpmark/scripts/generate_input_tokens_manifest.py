"""Generate the locked input-token experiment manifest + schedule (Phase 0).

Derived from ``filesystem-verified-v1.json``: same 10/20 split, same
counterbalanced order bits, plus the token-telemetry contract (primary
endpoint, budgets, exclusion rules, success margin, statistics) and
pre-assigned unique attempt IDs. No model attempt may run under the final
experiment ID before this manifest is locked.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from toolatlas.experiment_manifest import attempt_id, first_arm  # noqa: E402

EXPERIMENT_ID = "filesystem-input-tokens-v1"
PROTOCOL_VERSION = "toolatlas-input-tokens-v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path,
                        default=Path("benchmarks/mcpmark/manifests/filesystem-verified-v1.json"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("benchmarks/mcpmark/manifests/filesystem-input-tokens-v1.json"))
    parser.add_argument("--schedule", type=Path,
                        default=Path("benchmarks/mcpmark/manifests/filesystem-input-tokens-v1.schedule.json"))
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "protocol_version": PROTOCOL_VERSION,
        "derived_from": base.get("experiment_id"),
        "upstream_sha": base.get("upstream_sha"),
        "task_suite": base.get("task_suite", "standard"),
        "service": base.get("service", "filesystem"),
        "split_rule": base.get("split_rule"),
        "split_label": base.get("split_label"),
        "train_tasks": base.get("train_tasks"),
        "eval_tasks": base.get("eval_tasks"),
        "counts": base.get("counts"),
        "attempts": {
            "training_per_task": 4,
            "eval_per_task_per_arm": 4,
            "arms": ["baseline", "toolatlas"],
        },
        "primary_endpoint": {
            "metric": "online_input_tokens",
            "variant": "cached_inclusive",
            "also_report": ["uncached_billable", "total_inference_tokens"],
            "definition": "input_token_saving = baseline_online_input_tokens - toolatlas_online_input_tokens",
            "unit": "provider-reported input tokens per attempt, summed per task-clustered analysis",
        },
        "model_provider_settings": {
            "model": "TBD — lock exact model ID before first evaluation attempt",
            "provider": "TBD — gemini | openai_compatible",
            "adapter": "src/toolatlas/gemini_rest.py GeminiRestClient or OpenAI-compatible client",
            "temperature": 0.0,
            "reasoning_effort": "TBD — lock before first attempt",
            "max_tokens": 1024,
            "max_turns": 30,
            "system_prompt_hash": "TBD — hash of the neutral baseline prompt",
            "tool_schema_hash": "TBD — hash of exposed provider tools",
            "snapshot_hash": "TBD — official task state archive hash",
            "code_rev": "TBD — git rev-parse HEAD at lock time",
            "integration_patch": "benchmarks/mcpmark/patches/0001-toolatlas-agent.patch",
            "locked_note": "Changing model/provider/reasoning/budgets/prompts/code/image/schemas requires a new experiment ID.",
        },
        "guidance_policy": {
            "top_k": 3,
            "read_budget": 8,
            "token_budget": 384,
            "assisted_memory_calls": 1,
            "baseline_memory_calls": 0,
            "empty_retrieval": "inject nothing; empty is valid, never generic advice",
            "second_retrieval": "forbidden in the primary experiment",
        },
        "memory_policy": base.get("memory_policy"),
        "retry_policy": "identical budgets and retry handling in both arms; retries are separate ledger rows",
        "context_compaction_policy": "identical in both arms; no ToolAtlas-only truncation or caching",
        "exclusion_rules": [
            "attempts with missing authoritative input usage are excluded from the primary analysis but retained as diagnostics",
            "infrastructure failures are reported both including and excluding the failure class",
            "failed and timed-out attempts stay in the denominator for success rates",
        ],
        "success_margin": {
            "non_inferiority_pp": 5,
            "note": "ToolAtlas pass rate no more than 5pp below baseline on the 20-task study; tighter 2pp margin needs a larger set",
        },
        "statistics": {
            "unit": "paired (task_id, attempt) blocks; uncertainty clustered by task",
            "metrics": ["mean", "median", "p90", "total"],
            "ci": "task-clustered bootstrap 95% confidence interval for mean input-token saving",
            "success_rule": "lower CI bound > 0 AND success within the non-inferiority margin",
        },
        "claim_label": (
            "On a pre-registered held-out subset of official MCPMark Filesystem tasks, the same model "
            "and agent use fewer online input tokens with frozen ToolAtlas guidance than without it, "
            "while maintaining non-inferior official-verifier success."
        ),
        "claim_scope_limits": base.get("claim_scope_limits"),
        "status": "pre-registered — locked before any model execution; never change split after seeing results",
    }

    eval_schedule = []
    for task in base.get("eval_tasks", []):
        task_id = task["task_id"]
        for attempt in range(1, 5):
            first = first_arm(task_id, attempt, EXPERIMENT_ID)
            for arm in ("baseline", "toolatlas"):
                eval_schedule.append({
                    "experiment_id": EXPERIMENT_ID,
                    "task_id": task_id,
                    "attempt": attempt,
                    "arm": arm,
                    "attempt_id": attempt_id(EXPERIMENT_ID, task_id, arm, attempt),
                    "first": first,
                    "reset_between_arms": True,
                    "fresh_conversation": True,
                })
    # Deterministic order: task, attempt, arm with counterbalanced first.
    eval_schedule.sort(key=lambda e: (e["task_id"], e["attempt"], e["arm"]))

    schedule = {
        "experiment_id": EXPERIMENT_ID,
        "training_schedule": [
            {"task_id": t["task_id"], "attempts": [1, 2, 3, 4]}
            for t in base.get("train_tasks", [])
        ],
        "eval_schedule": eval_schedule,
    }

    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    args.schedule.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.manifest} and {args.schedule}")
    print(f"eval rows: {len(eval_schedule)} (expect 160)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
