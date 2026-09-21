"""Paired runner, aggregator, and artifact auditor (plan Phase 6).

Offline-capable: the aggregator and auditor operate on immutable attempt
records (JSONL) so headline numbers recompute exactly from the request/event
ledgers. Live execution (official setup/agent/verifier) stays behind the
MCPMark integration; this module enforces the invariants that make a run
reportable: explicit arm/attempt IDs, 1/0 memory-call discipline, frozen-memory
immutability, resume provenance checks, and spending caps.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any

from .experiment_manifest import attempt_id
from .token_usage import RequestLedger, check_resume_compatible

TERMINAL_STATUSES = (
    "success", "agent_failure", "verifier_failure", "timeout",
    "setup_failure", "provider_mcp_failure", "cleanup_failure",
    "infrastructure_failure",
)

SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password|bearer)\s*[:=]\s*\S+"),
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"nvapi-[A-Za-z0-9_.\-~+/=]+"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]+"),
    re.compile(r"xox[bap]-[A-Za-z0-9-]+"),
)
HOST_PATH_PATTERNS = (
    re.compile(r"[A-Za-z]:\\(?:[^\\s\"']+\\)*[^\\s\"']*"),
    re.compile(r"(?<!\w)/(?:home|Users|tmp|var|etc|root|mnt|data)/\S*"),
)


def scan_text(text: str) -> dict[str, list[str]]:
    """Find credential and absolute-host-path leaks in shareable artifacts."""
    blob = text if isinstance(text, str) else json.dumps(text, default=str)
    secrets = sorted({m.group(0)[:60] for p in SECRET_PATTERNS for m in p.finditer(blob)})
    paths = sorted({m.group(0)[:120] for p in HOST_PATH_PATTERNS for m in p.finditer(blob)})
    return {"secrets": secrets, "host_paths": paths}


def build_attempt_record(
    *,
    experiment_id: str,
    task_id: str,
    arm: str,
    attempt: int,
    model: str,
    provider: str,
    verifier_passed: bool,
    ledger: RequestLedger | None = None,
    mcp_events: list[dict] | None = None,
    provider_calls: int = 0,
    memory_calls: int = 0,
    guidance_text: str = "",
    guidance_tokens_estimate: int = 0,
    frozen_memory_sha256: str = "",
    terminal_status: str = "",
    elapsed_seconds: float = 0.0,
    model_requests: int | None = None,
) -> dict[str, Any]:
    """Assemble one immutable attempt record with a unique pre-assigned ID."""
    if arm not in ("baseline", "toolatlas"):
        raise ValueError(f"arm must be 'baseline'|'toolatlas', got {arm!r}")
    if attempt < 1:
        raise ValueError(f"attempt must be >= 1, got {attempt!r}")
    rows = ledger.attempt_rows(task_id, arm, attempt) if ledger is not None else []
    totals = ledger.attempt_totals(task_id, arm, attempt) if ledger is not None else {}
    if arm == "baseline" and memory_calls != 0:
        raise ValueError("baseline must make zero memory calls")
    if arm == "toolatlas" and memory_calls != 1:
        raise ValueError("assisted attempts must make exactly one guidance call")
    status = terminal_status or ("success" if verifier_passed else "verifier_failure")
    if status not in TERMINAL_STATUSES:
        raise ValueError(f"unknown terminal status: {status!r}")
    return {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "arm": arm,
        "attempt": attempt,
        "attempt_id": attempt_id(experiment_id, task_id, arm, attempt),
        "verifier_passed": verifier_passed,
        "token_ledger": [r.to_dict() for r in rows],
        "ledger_totals": totals,
        "mcp_events": list(mcp_events or []),
        "provider_calls": provider_calls,
        "memory_calls": memory_calls,
        "total_mcp_calls": provider_calls + memory_calls,
        "model_requests": model_requests if model_requests is not None else len(rows),
        "guidance_text": guidance_text,
        "guidance_tokens_estimate": guidance_tokens_estimate,
        "model": model,
        "provider": provider,
        "frozen_memory_sha256": frozen_memory_sha256,
        "terminal_status": status,
        "elapsed_seconds": elapsed_seconds,
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def _task_clustered_bootstrap_ci(
    savings_by_task: dict[str, list[float]],
    *,
    resamples: int = 2000,
    seed: int = 7,
) -> dict[str, float]:
    """95% CI for mean saving, resampling tasks (clusters) with replacement."""
    tasks = sorted(savings_by_task)
    if not tasks:
        return {"low": 0.0, "high": 0.0, "mean": 0.0}
    flat = [s for t in tasks for s in savings_by_task[t]]
    mean = sum(flat) / len(flat)
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        sample: list[float] = []
        for _ in tasks:
            sample.extend(savings_by_task[rng.choice(tasks)])
        means.append(sum(sample) / len(sample))
    means.sort()
    low = means[int(0.025 * resamples)]
    high = means[min(resamples - 1, int(0.975 * resamples))]
    return {"low": round(low, 2), "high": round(high, 2), "mean": round(mean, 2)}


def _attempt_input(row: dict[str, Any], variant: str = "cached_inclusive") -> float | None:
    totals = row.get("ledger_totals") or {}
    if variant == "uncached_billable":
        value = totals.get("uncached_input_tokens")
        if value is None:
            value = totals.get("input_tokens")
    else:
        value = totals.get("input_tokens")
    if value is None:
        # Fallback for harness rows without a ledger: legacy prompt_tokens.
        value = row.get("prompt_tokens")
    return float(value) if value is not None else None


def aggregate_results(
    attempts: list[dict[str, Any]],
    *,
    variant: str = "cached_inclusive",
    offline_input_tokens: int = 0,
    resamples: int = 2000,
) -> dict[str, Any]:
    """Aggregate paired per-task/attempt savings with task-clustered uncertainty."""
    baseline = [a for a in attempts if a.get("arm") == "baseline"]
    assisted = [a for a in attempts if a.get("arm") == "toolatlas"]
    base_inputs = [v for a in baseline if (v := _attempt_input(a, variant)) is not None]
    aid_inputs = [v for a in assisted if (v := _attempt_input(a, variant)) is not None]

    def _describe(values: list[float]) -> dict[str, float]:
        if not values:
            return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "total": 0.0}
        return {
            "n": len(values),
            "mean": round(sum(values) / len(values), 2),
            "median": round(_percentile(values, 50), 2),
            "p90": round(_percentile(values, 90), 2),
            "total": round(sum(values), 2),
        }

    # Paired savings per (task_id, attempt) block present in both arms.
    base_map = {(a["task_id"], a["attempt"]): a for a in baseline}
    aid_map = {(a["task_id"], a["attempt"]): a for a in assisted}
    savings_by_task: dict[str, list[float]] = {}
    pair_savings: list[float] = []
    for key in sorted(set(base_map) & set(aid_map)):
        b_val = _attempt_input(base_map[key], variant)
        t_val = _attempt_input(aid_map[key], variant)
        if b_val is None or t_val is None:
            continue
        saving = b_val - t_val
        pair_savings.append(saving)
        savings_by_task.setdefault(key[0], []).append(saving)

    ci = _task_clustered_bootstrap_ci(savings_by_task, resamples=resamples)
    mean_saving = ci["mean"]
    pct: float | None = None
    base_mean = _describe(base_inputs).get("mean", 0.0)
    if base_mean:
        pct = round(100.0 * mean_saving / base_mean, 2)

    def _pass_rates(rows: list[dict]) -> dict[str, Any]:
        n = len(rows)
        ok = sum(1 for r in rows if r.get("verifier_passed"))
        tasks = sorted({r["task_id"] for r in rows})
        with_success = sum(
            1 for t in tasks
            if any(r.get("task_id") == t and r.get("verifier_passed") for r in rows))
        pass_at_4 = (with_success / len(tasks)) if tasks else 0.0
        return {"n": n, "passed": ok, "pass_at_1": round(ok / n, 4) if n else 0.0,
                "pass_at_4": round(pass_at_4, 4)}

    base_rates = _pass_rates(baseline)
    aid_rates = _pass_rates(assisted)
    success_diff = round(aid_rates["pass_at_1"] - base_rates["pass_at_1"], 4)

    verified_base = [v for a, v in zip(baseline, base_inputs) if a.get("verifier_passed")]
    verified_aid = [v for a, v in zip(assisted, aid_inputs) if a.get("verifier_passed")]

    mean_positive = ci["low"] > 0
    break_even: int | None = None
    break_even_range: list[int | None] = [None, None]
    if mean_saving > 0 and offline_input_tokens:
        break_even = math.ceil(offline_input_tokens / mean_saving)
        if ci["low"] > 0:
            break_even_range = [
                math.ceil(offline_input_tokens / ci["high"]) if ci["high"] > 0 else None,
                math.ceil(offline_input_tokens / ci["low"]),
            ]

    return {
        "variant": variant,
        "baseline_input_tokens": _describe(base_inputs),
        "toolatlas_input_tokens": _describe(aid_inputs),
        "mean_saving": mean_saving,
        "saving_ci95": {"low": ci["low"], "high": ci["high"]},
        "saving_percent": pct,
        "paired_blocks": len(pair_savings),
        "baseline_success": base_rates,
        "toolatlas_success": aid_rates,
        "success_rate_difference": success_diff,
        "input_per_verified_success": {
            "baseline": round(sum(verified_base) / len(verified_base), 2) if verified_base else None,
            "toolatlas": round(sum(verified_aid) / len(verified_aid), 2) if verified_aid else None,
        },
        "offline_construction_input_tokens": offline_input_tokens,
        "amortized_break_even_tasks": break_even,
        "amortized_break_even_range": break_even_range,
        "primary_gate_mean_saving_positive": mean_positive,
    }


def audit_results(
    attempts: list[dict[str, Any]],
    manifest: dict[str, Any],
    schedule: dict[str, Any],
    *,
    frozen_sha256: str = "",
    resamples: int = 2000,
) -> dict[str, Any]:
    """Recompute aggregates from ledgers and flag contract violations."""
    errors: list[str] = []
    warnings: list[str] = []

    scheduled = {
        (e.get("task_id"), e.get("arm"), e.get("attempt"))
        for e in schedule.get("eval_schedule", [])
    }
    observed = {
        (a.get("task_id"), a.get("arm"), a.get("attempt")) for a in attempts
    }
    if scheduled - observed:
        errors.append(f"missing {len(scheduled - observed)} scheduled attempts")
    if observed - scheduled:
        errors.append(f"{len(observed - scheduled)} unscheduled attempts observed")

    ids = [a.get("attempt_id") for a in attempts]
    if len(set(ids)) != len(ids):
        errors.append("duplicate attempt_ids observed")
    for entry in schedule.get("eval_schedule", []):
        key = (entry.get("task_id"), entry.get("arm"), entry.get("attempt"))
        match = next((a for a in attempts
                      if (a.get("task_id"), a.get("arm"), a.get("attempt")) == key), None)
        if match is not None and match.get("attempt_id") != entry.get("attempt_id"):
            errors.append(f"attempt_id mismatch for {key}")

    for row in attempts:
        if row.get("arm") == "baseline" and row.get("memory_calls", 0) != 0:
            errors.append(f"baseline memory-call violation: {row.get('attempt_id')}")
        if row.get("arm") == "toolatlas" and row.get("memory_calls", 0) != 1:
            errors.append(f"assisted memory-call violation: {row.get('attempt_id')}")
        ledger = row.get("token_ledger") or []
        authoritative = [
            (r.get("usage") or {}).get("has_authoritative_input") for r in ledger
        ]
        if ledger and not all(authoritative):
            warnings.append(f"incomplete usage in {row.get('attempt_id')}: excluded from primary")
        if not ledger and row.get("prompt_tokens") is None:
            warnings.append(f"no token ledger for {row.get('attempt_id')}")
        if "verifier_passed" not in row:
            errors.append(f"missing verifier result: {row.get('attempt_id')}")
        if row.get("terminal_status") not in TERMINAL_STATUSES:
            errors.append(f"bad terminal status: {row.get('attempt_id')}")
        if frozen_sha256 and row.get("frozen_memory_sha256") not in ("", frozen_sha256):
            errors.append(f"frozen-memory hash mismatch: {row.get('attempt_id')}")

    blob = json.dumps(attempts, default=str)
    leak = scan_text(blob)
    if leak["secrets"]:
        errors.append(f"credential patterns in artifacts: {len(leak['secrets'])}")
    if leak["host_paths"]:
        errors.append(f"host paths in artifacts: {len(leak['host_paths'])}")

    recomputed = aggregate_results(attempts, resamples=resamples)
    return {
        "errors": errors,
        "warnings": warnings,
        "scheduled": len(scheduled),
        "observed": len(observed),
        "recomputed": recomputed,
        "leaks": {k: len(v) for k, v in leak.items()},
        "passed": not errors,
    }


def validate_resume(prior: dict[str, Any], current: dict[str, Any]) -> None:
    """Refuse resume when model/provider/settings provenance differs."""
    mismatched = check_resume_compatible(prior, current)
    if mismatched:
        raise SystemExit(
            f"refusing to resume: manifest mismatch on {mismatched}. "
            "Use a fresh output directory when changing configuration.")


def attempt_output_path(out_root: str | Path, record: dict[str, Any]) -> Path:
    """Collision-free per-attempt directory: <task>__<arm>__<attempt_id>."""
    safe_task = re.sub(r"[^A-Za-z0-9_.\-]+", "_", record["task_id"]).strip("_")[:120]
    return Path(out_root) / f"{safe_task}__{record['arm']}__{record['attempt_id']}"


def check_spending_cap(spent: float, cap: float) -> None:
    if cap > 0 and spent >= cap:
        raise SystemExit(f"spending/token cap reached: {spent} >= {cap}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
