from datetime import UTC, datetime, timedelta

from toolatlas import ExecutionStep, Rollout, ToolMemory, ToolSpec


def test_induction_persistence_and_traversal(tmp_path):
    path = tmp_path / "memory.json"
    memory = ToolMemory(path)
    memory.register_tools(
        [
            ToolSpec("normalize_text", "Normalize text"),
            ToolSpec("keyword_count", "Count keywords"),
        ]
    )
    steps = [
        ExecutionStep("normalize_text", "normalize text before analysis"),
        ExecutionStep("keyword_count", "count a target keyword in normalized text"),
    ]
    for task_id, summary in [
        ("q1", "Normalize text and count a keyword"),
        ("q2", "Clean text and calculate keyword frequency"),
    ]:
        memory.induce(task_id, summary, [Rollout(task_id, summary, steps, True)])
    memory.induce(
        "q3",
        "Count an empty keyword",
        [
            Rollout(
                "q3",
                "Count an empty keyword",
                [ExecutionStep("keyword_count", "count a keyword in text")],
                False,
                "an empty keyword is rejected",
            )
        ],
    )

    loaded = ToolMemory(path)
    assert loaded.stats()["strategies"] == 1
    assert loaded.tools["keyword_count"].boundaries
    guidance = loaded.guide("clean some text then count a keyword")
    assert guidance["seed_candidates"]
    assert [step["tool"] for step in guidance["playbook"]][:2] == [
        "normalize_text",
        "keyword_count",
    ]
    assert guidance["strategy"]
    probes = loaded.suggest_probes("keyword_count")
    assert probes["boundary"][0]["direction"] == "outward"
    assert "normalize_text" in probes["affordance"][0]["task"]


def test_irrelevant_task_returns_empty_guidance():
    memory = ToolMemory()
    memory.induce(
        "q1",
        "Normalize text",
        [Rollout("q1", "Normalize text", [ExecutionStep("normalize_text", "normalize text")], True)],
    )
    assert memory.guide("forecast tomorrow weather")["seed_candidates"] == []


def test_batch_provenance_confidence_and_sqlite_reload(tmp_path):
    path = tmp_path / "memory.db"
    memory = ToolMemory(path)
    memory.register_tools(
        [ToolSpec("normalize_text", "Normalize text", {"type": "object"}, version="1")]
    )
    steps = [ExecutionStep("normalize_text", "normalize text")]
    trace = memory.induce(
        "batch",
        "Normalize text",
        [
            Rollout("batch", "Normalize text", steps, True, verifier_type="exact_match"),
            Rollout(
                "batch", "Normalize text", steps, False,
                "unsupported input", verifier_type="exact_match",
            ),
        ],
    )

    assert trace.success_count == 1
    assert trace.failure_count == 1
    assert trace.confidence == 0.5
    assert len(trace.source_executions) == 2
    loaded = ToolMemory(path)
    assert loaded.stats()["executions"] == 2
    assert loaded.traces["batch"].source_executions == trace.source_executions


def test_schema_change_stales_memory_until_reverified(tmp_path):
    memory = ToolMemory(tmp_path / "memory.db")
    memory.register_tools(
        [ToolSpec("normalize_text", "Normalize text", {"required": ["text"]}, version="1")]
    )
    memory.induce(
        "q1",
        "Normalize text",
        [Rollout("q1", "Normalize text", [ExecutionStep("normalize_text", "normalize text")], True)],
    )
    assert memory.guide("normalize text")["playbook"]

    memory.register_tools(
        [ToolSpec("normalize_text", "Normalize Unicode text", {"required": ["text"]}, version="2")]
    )
    assert memory.stats()["stale_traces"] == 1
    assert memory.refresh_status()["refresh_candidates"][0]["qid"] == "q1"
    assert memory.guide("normalize text")["playbook"] == []

    refreshed = memory.reverify_trace("q1", True, "exact_match")
    assert refreshed.status == "active"
    assert memory.stats()["stale_traces"] == 0
    assert memory.guide("normalize text")["playbook"]


def test_governance_and_read_budget():
    memory = ToolMemory()
    memory.register_tools([ToolSpec("normalize_text", "Normalize text")])
    memory.induce(
        "q1",
        "Normalize text",
        [Rollout("q1", "Normalize text", [ExecutionStep("normalize_text", "normalize text")], True)],
    )
    guidance = memory.guide("normalize text", read_budget=1)
    assert guidance["traversal"]["reads_used"] <= 1
    memory.set_trace_status("q1", "quarantined", "manual security review")
    assert memory.guide("normalize text")["playbook"] == []


def test_expired_verification_is_not_served():
    memory = ToolMemory()
    memory.register_tools([ToolSpec("normalize_text", "Normalize text")])
    old = (datetime.now(UTC) - timedelta(days=60)).isoformat()
    memory.induce(
        "old",
        "Normalize old text",
        [
            Rollout(
                "old", "Normalize old text",
                [ExecutionStep("normalize_text", "normalize text")],
                True,
                verified_at=old,
            )
        ],
    )
    assert memory.refresh_status(max_age_days=30)["refresh_candidates"][0]["reason"] == "verification_expired"
    assert memory.guide("normalize old text", max_age_days=30)["playbook"] == []
