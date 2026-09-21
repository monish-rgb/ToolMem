"""Plan phase tests: budgeted guidance, physical freeze, runner, builder."""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp import Client

from toolatlas.experiment_manifest import attempt_id
from toolatlas.freeze_memory import (
    check_no_sidecars,
    freeze_memory,
    verify_frozen_unchanged,
)
from toolatlas.memory import ToolMemory
from toolatlas.memory_server import READ_ONLY_TOOLS, create_memory_server
from toolatlas.models import Rollout
from toolatlas.offline_builder import (
    OfflineCost,
    build_offline_memory,
    run_exploration_rounds,
    select_backbone,
    select_guidance_budget,
)
from toolatlas.paired_runner import (
    aggregate_results,
    audit_results,
    build_attempt_record,
    scan_text,
)
from toolatlas.retrieval_metrics import retrieval_metrics
from toolatlas.storage import SQLiteStore
from toolatlas.token_usage import RequestLedger, normalize_openai_usage


def _seed_memory(path, n_traces=2):
    memory = ToolMemory(path)
    memory.register_tools([
        __import__("toolatlas.models", fromlist=["ToolSpec"]).ToolSpec(
            name="alpha", description="do alpha"),
        __import__("toolatlas.models", fromlist=["ToolSpec"]).ToolSpec(
            name="beta", description="do beta"),
    ])
    for i in range(n_traces):
        memory.induce(f"t{i}", f"Classify files with alpha step {i}", [
            Rollout(task_id=f"t{i}", summary="s",
                    steps=[__import__("toolatlas.models", fromlist=["ExecutionStep"]).ExecutionStep(
                        "alpha", "establish working scope with alpha"),
                        __import__("toolatlas.models", fromlist=["ExecutionStep"]).ExecutionStep(
                        "beta", "produce output with beta")],
                    resolved=True, observation="ok", verifier_type="v")])
    return memory


def test_guidance_respects_token_budget(tmp_path):
    memory = _seed_memory(tmp_path / "m.db")
    full = memory.guide("Classify files with alpha")
    assert full["playbook"]
    small = memory.guide("Classify files with alpha", token_budget=20)
    assert small["truncated"] is True
    assert small["guidance_tokens_estimate"] <= full["guidance_tokens_estimate"]
    empty = memory.guide("completely unrelated zebra quantum")
    assert empty["seed_candidates"] == [] and empty["playbook"] == []


def test_server_get_guidance_accepts_token_budget(tmp_path):
    db = tmp_path / "m.db"
    _seed_memory(db)
    server = create_memory_server(db)

    async def _call():
        async with Client(server) as memory:
            full = await memory.call_tool("get_guidance", {"task": "Classify files with alpha"})
            capped = await memory.call_tool(
                "get_guidance", {"task": "Classify files with alpha", "token_budget": 20})
            return full.structured_content, capped.structured_content

    full, capped = asyncio.run(_call())
    assert full["playbook"]
    assert capped["truncated"] is True


def test_read_only_memory_serves_but_rejects_mutation(tmp_path):
    src = tmp_path / "train.db"
    _seed_memory(src)
    frozen = tmp_path / "frozen.db"
    record = freeze_memory(src, frozen)
    assert record["integrity_check"] == "ok"

    memory = ToolMemory(frozen, read_only=True)
    guidance = memory.guide("Classify files with alpha")
    assert guidance["seed_candidates"]
    with pytest.raises(RuntimeError):
        memory.induce("tX", "summary", [])
    with pytest.raises(RuntimeError):
        memory.save()

    store = SQLiteStore(frozen, read_only=True)
    with store.connect() as db:
        with pytest.raises(Exception):
            db.execute("CREATE TABLE probe(x TEXT)")


def test_read_only_server_profile_unchanged(tmp_path):
    db = tmp_path / "m.db"
    _seed_memory(db)

    async def _names():
        async with Client(create_memory_server(db, read_only=True)) as memory:
            return {t.name for t in (await memory.list_tools()).tools}

    assert asyncio.run(_names()) == set(READ_ONLY_TOOLS)


def test_frozen_hash_audit(tmp_path):
    src = tmp_path / "train.db"
    _seed_memory(src)
    frozen = tmp_path / "frozen.db"
    record = freeze_memory(src, frozen)
    assert check_no_sidecars(frozen) == []
    ok = verify_frozen_unchanged(frozen, record["frozen_sha256"])
    assert ok["integrity_check"] == "ok"
    with open(frozen, "ab") as fh:
        fh.write(b"tamper")
    with pytest.raises(RuntimeError):
        verify_frozen_unchanged(frozen, record["frozen_sha256"])


def _ledger_row(experiment, task, arm, attempt, prompt, request_id):
    ledger = RequestLedger()
    ledger.log(experiment_id=experiment, task_id=task, arm=arm, attempt=attempt,
               model="m", provider="openai_compatible",
               usage=normalize_openai_usage(
                   {"prompt_tokens": prompt, "completion_tokens": 10,
                    "total_tokens": prompt + 10}, request_id))
    return ledger


def test_paired_runner_enforces_memory_discipline():
    ledger = _ledger_row("e", "t", "baseline", 1, 100, "r1")
    record = build_attempt_record(
        experiment_id="e", task_id="t", arm="baseline", attempt=1,
        model="m", provider="p", verifier_passed=True, ledger=ledger,
        provider_calls=5, memory_calls=0, frozen_memory_sha256="abc")
    assert record["attempt_id"] == attempt_id("e", "t", "baseline", 1)
    assert record["total_mcp_calls"] == 5
    with pytest.raises(ValueError):
        build_attempt_record(experiment_id="e", task_id="t", arm="baseline",
                             attempt=1, model="m", provider="p",
                             verifier_passed=True, provider_calls=5, memory_calls=1)
    with pytest.raises(ValueError):
        build_attempt_record(experiment_id="e", task_id="t", arm="toolatlas",
                             attempt=1, model="m", provider="p",
                             verifier_passed=True, provider_calls=5, memory_calls=0)


def test_aggregate_and_audit_recompute():
    attempts = []
    for i, task in ( enumerate(["t1", "t2"])):
        for attempt in (1, 2):
            for arm, prompt in (("baseline", 200), ("toolatlas", 120)):
                ledger = _ledger_row("e", task, arm, attempt, prompt, f"{task}{arm}{attempt}")
                attempts.append(build_attempt_record(
                    experiment_id="e", task_id=task, arm=arm, attempt=attempt,
                    model="m", provider="p", verifier_passed=True, ledger=ledger,
                    provider_calls=4, memory_calls=1 if arm == "toolatlas" else 0,
                    frozen_memory_sha256="abc"))
    summary = aggregate_results(attempts, resamples=200)
    assert summary["mean_saving"] == 80
    assert summary["saving_ci95"]["low"] <= 80 <= summary["saving_ci95"]["high"]
    assert summary["primary_gate_mean_saving_positive"] is True
    assert summary["toolatlas_success"]["pass_at_1"] == 1.0
    manifest = {"experiment_id": "e"}
    schedule = {"experiment_id": "e", "eval_schedule": [
        {"task_id": a["task_id"], "arm": a["arm"], "attempt": a["attempt"],
         "attempt_id": a["attempt_id"]} for a in attempts]}
    audit = audit_results(attempts, manifest, schedule, frozen_sha256="abc", resamples=200)
    assert audit["passed"] is True
    assert audit["recomputed"]["mean_saving"] == 80
    # Tamper with one memory count -> audit fails.
    attempts[0]["memory_calls"] = 3
    audit2 = audit_results(attempts, manifest, schedule, frozen_sha256="abc", resamples=200)
    assert audit2["passed"] is False


def test_audit_detects_leaks():
    assert scan_text("api_key: sk-abc123XYZ q")["secrets"]
    assert scan_text("output at C:\\work\\x\\out")["host_paths"]
    assert scan_text("benign summary text") == {"secrets": [], "host_paths": []}


def test_offline_builder_backbone_and_cost(tmp_path):
    from toolatlas.models import ExecutionStep
    long_ok = Rollout("t", "s", [ExecutionStep("a", "x"), ExecutionStep("b", "y")],
                      True, "ok", verifier_type="v")
    short_ok = Rollout("t", "s", [ExecutionStep("a", "x")], True, "ok", verifier_type="v")
    assert select_backbone([long_ok, short_ok]) is short_ok

    memory = ToolMemory(tmp_path / "b.db")
    memory.register_tools([
        __import__("toolatlas.models", fromlist=["ToolSpec"]).ToolSpec(
            name="alpha", description="do alpha")])

    def _execute(task_id, seed_steps):
        tool = seed_steps[0]["tool"]
        return [{"tool": tool, "rationale": f"use {tool} carefully"}]

    def _verify(task_id, steps):
        return True, "verified ok"

    cost = OfflineCost()
    report = build_offline_memory(memory, ["alpha"], _execute, _verify, cost=cost)
    assert report["ingested_traces"] == 3  # 3 seeds per tool
    assert cost.training_attempts == 3 * 4
    assert memory.stats()["traces"] >= 3

    explore = run_exploration_rounds(memory, _execute, _verify, cost=cost)
    assert explore["rounds"] == 3
    assert cost.exploration_verified > 0

    tuning = select_guidance_budget({128: 0.9, 256: 0.9, 384: 0.8},
                                    {128: 100.0, 256: 150.0, 384: 160.0},
                                    lambda_weight=0.1)
    assert tuning.selected in (128, 256, 384)
    assert set(tuning.utilities) == {128, 256, 384}


def test_retrieval_metrics_rates():
    guides = [
        {"seed_candidates": [{"summary": "s"}], "playbook": [{"tool": "alpha"}],
         "avoid": [], "conventions": []},
        {"seed_candidates": [], "playbook": [], "avoid": [], "conventions": []},
    ]
    metrics = retrieval_metrics(
        guides, irrelevant_flags=[False, False],
        baseline_turns=[5, 4], assisted_turns=[3, 4],
        baseline_repeats=[2, 1], assisted_repeats=[0, 1],
        baseline_success=[True, False], assisted_success=[True, False])
    assert metrics["empty_guidance_rate"] == 0.5
    assert metrics["guidance_coverage_tools"] == ["alpha"]
    assert metrics["mean_turns_avoided"] == 1.0
    assert metrics["mean_repeated_calls_avoided"] == 1.0
    assert metrics["negative_transfer_rate"] == 0.0
    assert metrics["max_guidance_tokens"] > 0
