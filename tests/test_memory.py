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
