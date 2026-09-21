"""Phase 0 evidence-contract tests: manifest, schedule, attempt IDs."""

from __future__ import annotations

import json
from pathlib import Path

from toolatlas.experiment_manifest import (
    attempt_id,
    first_arm,
    validate_manifest,
    validate_schedule,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "benchmarks" / "mcpmark" / "manifests" / "filesystem-input-tokens-v1.json"
SCHEDULE = REPO / "benchmarks" / "mcpmark" / "manifests" / "filesystem-input-tokens-v1.schedule.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_is_valid_and_locked():
    manifest = _load(MANIFEST)
    assert validate_manifest(manifest) == []
    assert manifest["attempts"]["training_per_task"] == 4
    assert manifest["attempts"]["eval_per_task_per_arm"] == 4
    assert manifest["primary_endpoint"]["metric"] == "online_input_tokens"
    assert manifest["primary_endpoint"]["variant"] == "cached_inclusive"
    assert manifest["guidance_policy"]["token_budget"] == 384
    train = {t["task_id"] for t in manifest["train_tasks"]}
    eval_ = {t["task_id"] for t in manifest["eval_tasks"]}
    assert len(train) == 10 and len(eval_) == 20
    assert not (train & eval_)


def test_schedule_covers_160_rows_with_unique_ids():
    manifest = _load(MANIFEST)
    schedule = _load(SCHEDULE)
    assert validate_schedule(manifest, schedule) == []
    assert len(schedule["eval_schedule"]) == 160
    ids = [e["attempt_id"] for e in schedule["eval_schedule"]]
    assert len(set(ids)) == 160


def test_attempt_ids_deterministic_and_paired():
    first = attempt_id("exp", "cat/task", "baseline", 1)
    assert first == attempt_id("exp", "cat/task", "baseline", 1)
    assert first != attempt_id("exp", "cat/task", "toolatlas", 1)
    assert len(first) == 16
    assert first_arm("cat/task", 1, "exp") in ("baseline", "toolatlas")


def test_manifest_rejects_overlap_and_wrong_budgets():
    manifest = _load(MANIFEST)
    bad = json.loads(json.dumps(manifest))
    bad["eval_tasks"] = bad["train_tasks"][:1] + bad["eval_tasks"]
    assert any("overlap" in e for e in validate_manifest(bad))
    bad2 = json.loads(json.dumps(manifest))
    bad2["guidance_policy"]["token_budget"] = 512
    assert any("token_budget" in e for e in validate_manifest(bad2))
