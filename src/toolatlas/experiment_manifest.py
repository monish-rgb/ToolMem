"""Versioned experiment-manifest helpers (input-token reduction plan, Phase 0).

The manifest is valid JSON and immutable once live evaluation starts: any
change to a locked field forces a new experiment ID. Attempt IDs are assigned
before execution as ``sha256(experiment|task|arm|attempt)[:16]`` so every
``(experiment_id, task_id, arm, attempt)`` row is unique and pre-registered.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

LOCKED_FIELDS = (
    "experiment_id", "protocol_version", "upstream_sha", "task_suite",
    "service", "split_rule", "train_tasks", "eval_tasks",
    "attempts", "primary_endpoint", "model_provider_settings",
    "guidance_policy", "memory_policy",
)

PRIMARY_VARIANTS = ("cached_inclusive", "uncached_billable")


def attempt_id(experiment_id: str, task_id: str, arm: str, attempt: int) -> str:
    """Deterministic unique ID assigned before execution."""
    seed = f"{experiment_id}|{task_id}|{arm}|{attempt}"
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def first_arm(task_id: str, attempt: int, experiment_id: str) -> str:
    """Counterbalanced order bit: hash decides which arm runs first."""
    bit = hashlib.sha256(f"{experiment_id}|{task_id}|{attempt}".encode()).hexdigest()
    return "toolatlas" if int(bit, 16) % 2 == 0 else "baseline"


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    """Return a list of validation errors (empty means valid)."""
    errors: list[str] = []
    for field in LOCKED_FIELDS:
        if field not in manifest:
            errors.append(f"missing locked field: {field}")
    attempts = manifest.get("attempts") or {}
    if attempts.get("training_per_task") != 4:
        errors.append("attempts.training_per_task must be 4")
    if attempts.get("eval_per_task_per_arm") != 4:
        errors.append("attempts.eval_per_task_per_arm must be 4")
    if sorted(attempts.get("arms", [])) != ["baseline", "toolatlas"]:
        errors.append("attempts.arms must be ['baseline', 'toolatlas']")
    primary = manifest.get("primary_endpoint") or {}
    if primary.get("metric") != "online_input_tokens":
        errors.append("primary_endpoint.metric must be 'online_input_tokens'")
    if primary.get("variant") not in PRIMARY_VARIANTS:
        errors.append(f"primary_endpoint.variant must be one of {PRIMARY_VARIANTS}")
    settings = manifest.get("model_provider_settings") or {}
    if not isinstance(settings, dict):
        errors.append("model_provider_settings must be an object")
    else:
        for key in ("model", "provider", "temperature"):
            if key not in settings:
                errors.append(f"model_provider_settings missing: {key}")
    guidance = manifest.get("guidance_policy") or {}
    for key, expected in (("top_k", 3), ("read_budget", 8), ("token_budget", 384)):
        if guidance.get(key) != expected:
            errors.append(f"guidance_policy.{key} must be {expected}")
    if guidance.get("assisted_memory_calls") != 1 or guidance.get("baseline_memory_calls") != 0:
        errors.append("guidance_policy must lock 1 assisted / 0 baseline memory calls")
    train_ids = {(t.get("task_id")) for t in manifest.get("train_tasks", [])}
    eval_ids = {(t.get("task_id")) for t in manifest.get("eval_tasks", [])}
    if train_ids & eval_ids:
        errors.append(f"train/eval task overlap: {sorted(train_ids & eval_ids)}")
    if len(train_ids) != 10:
        errors.append(f"expected 10 training tasks, got {len(train_ids)}")
    if len(eval_ids) != 20:
        errors.append(f"expected 20 evaluation tasks, got {len(eval_ids)}")
    return errors


def validate_schedule(manifest: dict[str, Any], schedule: dict[str, Any]) -> list[str]:
    """Schedule must cover the full pre-registered attempt set with unique IDs."""
    errors: list[str] = []
    experiment_id = manifest.get("experiment_id", "")
    if schedule.get("experiment_id") != experiment_id:
        errors.append("schedule.experiment_id must match the manifest")
        return errors
    eval_tasks = [t.get("task_id") for t in manifest.get("eval_tasks", [])]
    expected: set[tuple[str, str, int]] = set()
    for task_id in eval_tasks:
        for attempt in range(1, 5):
            for arm in ("baseline", "toolatlas"):
                expected.add((task_id, arm, attempt))
    seen_ids: set[str] = set()
    seen_keys: set[tuple[str, str, int]] = set()
    for entry in schedule.get("eval_schedule", []):
        key = (entry.get("task_id"), entry.get("arm"), entry.get("attempt"))
        if key in seen_keys:
            errors.append(f"duplicate schedule entry: {key}")
        seen_keys.add(key)
        expected_id = attempt_id(experiment_id, entry.get("task_id", ""),
                                 entry.get("arm", ""), entry.get("attempt", 0))
        if entry.get("attempt_id") != expected_id:
            errors.append(f"wrong attempt_id for {key}: {entry.get('attempt_id')}")
        if entry.get("attempt_id") in seen_ids:
            errors.append(f"duplicate attempt_id: {entry.get('attempt_id')}")
        seen_ids.add(entry.get("attempt_id", ""))
    missing = expected - seen_keys
    extra = seen_keys - expected
    if missing:
        errors.append(f"schedule missing {len(missing)} entries, e.g. {sorted(missing)[:3]}")
    if extra:
        errors.append(f"schedule has {len(extra)} unexpected entries: {sorted(extra)[:3]}")
    return errors


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
