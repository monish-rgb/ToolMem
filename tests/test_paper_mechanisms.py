"""Tests for the paper mechanisms: intent induction, hybrid retrieval,
fuzzy strategy merge, call-saving guidance, explorer, and refresh scheduling.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "benchmarks" / "mcpmark" / "integration")
)

from toolatlas import (  # noqa: E402
    ExecutionStep,
    Rollout,
    ToolMemory,
    ToolSpec,
    affordance_probes,
    boundary_probes,
    explore_with_client,
    exploration_summary,
    is_destructive,
    plan_exploration,
    run_exploration,
)
from toolatlas.explorer import ExplorationReport  # noqa: E402
from toolatlas.memory import (  # noqa: E402
    MAX_PLAYBOOK_STEPS,
    _entries_match,
    _intent_rationale,
    estimate_tokens,
)
from toolatlas.similarity import (  # noqa: E402
    hybrid_similarity,
    normalized_tokens,
    trigram_similarity,
)
from toolatlas_mcpmark.sanitize import format_guidance_block  # noqa: E402


def _rollout(task, summary, tools, resolved=True, observation=""):
    return [Rollout(
        task, summary,
        [ExecutionStep(tool, f"apply {tool} to valid inputs") for tool in tools],
        resolved, observation,
    )]


# -- intent rationales ----------------------------------------------------

def test_structural_rationales_become_positional_intent():
    assert _intent_rationale("alpha", "invoke alpha during the rollout", 0, 3) == (
        "establish working scope and inputs with alpha"
    )
    assert _intent_rationale("beta", "", 1, 3) == (
        "advance the intermediate result toward the task goal with beta"
    )
    assert _intent_rationale("gamma", "run gamma", 2, 3) == (
        "produce and verify the task output with gamma"
    )
    assert _intent_rationale("solo", "call solo", 0, 1) == (
        "apply solo to the task input and verify the result"
    )


def test_real_rationales_are_kept():
    assert _intent_rationale("alpha", "normalize text before analysis", 0, 2) == (
        "normalize text before analysis"
    )


def test_induction_upgrades_structural_backbone():
    from toolatlas import Rollout as _Rollout

    memory = ToolMemory()
    memory.induce("q", "Do a thing", [_Rollout(
        "q", "Do a thing",
        [ExecutionStep("a", "invoke a during the rollout"),
         ExecutionStep("b", "run b")],
        True,
    )])
    steps = memory.traces["q"].agent_neutral_trace
    assert steps[0].rationale == "establish working scope and inputs with a"
    assert steps[1].rationale == "produce and verify the task output with b"


# -- tips from success and failure -----------------------------------------

def test_tips_cover_planning_and_fixes():
    memory = ToolMemory()
    memory.induce("q", "Do a thing",
        _rollout("q", "Do a thing", ["a", "b"], True)
        + _rollout("q", "Do a thing", ["a"], False, "bad mode flag"))
    tips = memory.traces["q"].task_level_tips
    assert any("working scope first" in tip for tip in tips)
    assert any("bad mode flag" in tip for tip in tips)
    assert any("instead of repeating the failed call" in tip for tip in tips)


# -- fuzzy entry merge ------------------------------------------------------

def test_entries_match_normalized_and_fuzzy():
    assert _entries_match("Reliably supports X", "reliably supports x ")
    assert _entries_match(
        "Reliably supports workflows that locate files quickly",
        "Reliably supports workflows that locate files rapidly",
    )
    assert not _entries_match(
        "Reliably supports workflows that locate files",
        "Avoid or validate when quota exceeded",
    )
    assert not _entries_match("abc", "abd")


def test_near_duplicate_affordances_merge_evidence():
    memory = ToolMemory()
    memory.register_tools([ToolSpec("alpha", "do alpha")])
    memory.induce("q1", "Do alpha", _rollout("q1", "Do alpha", ["alpha"]))
    first = len(memory.tools["alpha"].affordances)
    memory.induce("q2", "Do alpha again", _rollout("q2", "Do alpha again", ["alpha"]))
    node = memory.tools["alpha"]
    assert len(node.affordances) == first
    assert set(node.affordances[0].source_queries) == {"q1", "q2"}
    assert node.affordances[0].success_count == 2


# -- normalized strategy merge -----------------------------------------------

def test_near_sequence_traces_form_a_strategy():
    memory = ToolMemory()
    memory.induce("q1", "First list then read files", _rollout("q1", "x", ["list", "read"]))
    assert memory.stats()["strategies"] == 0
    memory.induce("q2", "Second list filter then read files",
                  _rollout("q2", "x", ["list", "filter", "read"]))
    assert memory.stats()["strategies"] == 1
    strategy = next(iter(memory.stats() and memory.strategies.values()))
    assert set(strategy.source_queries) == {"q1", "q2"}


def test_unrelated_sequences_form_no_strategy():
    memory = ToolMemory()
    memory.induce("q1", "List then read files", _rollout("q1", "x", ["list", "read"]))
    memory.induce("q2", "Write then delete records", _rollout("q2", "x", ["write", "delete"]))
    assert memory.stats()["strategies"] == 0


# -- hybrid retrieval ----------------------------------------------------------

def test_trigram_and_hybrid_scores():
    assert trigram_similarity("forecast tomorrow weather", "Normalize text") == 0.0
    assert hybrid_similarity("forecast tomorrow weather", "Normalize text") == 0.0
    assert hybrid_similarity("tally words in a sentence", "Count keywords in text") > 0.15
    assert "count" in normalized_tokens("tally totals")


def test_paraphrase_query_rescued_by_fallback():
    memory = ToolMemory()
    memory.induce("q1", "Count keywords", _rollout("q1", "Count keywords", ["counter"]))
    guidance = memory.guide("tally keyword totals")
    assert guidance["seed_candidates"], "zero lexical overlap, shared stems"
    assert guidance["coverage"]["retrieval"] == "trigram-fallback"
    assert memory.guide("forecast tomorrow weather")["seed_candidates"] == []


# -- call-saving guidance -------------------------------------------------------

def _rich_memory():
    memory = ToolMemory()
    memory.register_tools([
        ToolSpec("alpha", "do alpha"),
        ToolSpec("beta", "do beta"),
    ])
    memory.induce("q1", "Do alpha then beta",
        _rollout("q1", "Do alpha then beta", ["alpha", "beta"], True)
        + _rollout("q1", "Do alpha then beta", ["alpha"], False, "beta rejected empty input"))
    memory.induce("q2", "Do alpha then beta again", _rollout("q2", "x", ["alpha", "beta"]))
    return memory


def test_guidance_has_avoid_conventions_and_estimate():
    guidance = _rich_memory().guide("do alpha then beta")
    assert guidance["avoid"], "verified boundary caution must reach the prompt"
    assert any("beta" in item["caution"] for item in guidance["avoid"])
    assert guidance["conventions"], "strategy first, then task tips"
    assert guidance["conventions"][0]["source"].startswith("strategy_")
    assert guidance["guidance_tokens_estimate"] > 0
    assert guidance["truncated"] is False
    assert guidance["coverage"]["tools_covered"]


def test_token_budget_truncates_smallest_value_first():
    memory = _rich_memory()
    full = memory.guide("do alpha then beta")
    small = memory.guide("do alpha then beta", token_budget=10)
    assert small["truncated"] is True
    assert small["guidance_tokens_estimate"] <= full["guidance_tokens_estimate"]
    with pytest.raises(ValueError):
        memory.guide("do alpha then beta", token_budget=0)


def test_playbook_is_capped():
    memory = ToolMemory()
    tools = [f"tool{i}" for i in range(12)]
    memory.register_tools([ToolSpec(name, f"do {name}") for name in tools])
    memory.induce("long", "Run a long chain", _rollout("long", "x", tools))
    guidance = memory.guide("run a long chain")
    assert len(guidance["playbook"]) <= MAX_PLAYBOOK_STEPS


# -- refresh scheduling ----------------------------------------------------------

def test_reverification_due_orders_overdue_first():
    memory = ToolMemory()
    old = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    memory.induce("old", "Old task", [
        Rollout("old", "Old task", [ExecutionStep("a", "use a")], True, verified_at=old)
    ])
    memory.induce("new", "New task", _rollout("new", "New task", ["b"]))
    due = memory.reverification_due()
    assert [item["qid"] for item in due["traces_due"]] == ["old"]
    assert due["traces_due"][0]["reason"] == "verification_expired"
    assert due["traces_due"][0]["days_overdue"] >= 50
    with pytest.raises(ValueError):
        memory.reverification_due(max_age_days=0)


# -- explorer --------------------------------------------------------------------

class FakeMCP:
    def __init__(self, behaviors):
        self.behaviors = behaviors
        self.calls = []

    async def list_tools(self):
        return SimpleNamespace(tools=[
            SimpleNamespace(name=name, description=spec[0], inputSchema=spec[1])
            for name, spec in self.behaviors.items()
        ])

    async def call_tool(self, tool, args):
        self.calls.append((tool, args))
        behavior = self.behaviors[tool][2]
        return behavior(tool, args)


def _ok(text="done"):
    return SimpleNamespace(is_error=False, structured_content={"result": text}, content=text)


def _err(text="rejected"):
    return SimpleNamespace(is_error=True, structured_content=None, content=text)


TEXT_SCHEMA = {"type": "object", "required": ["text"],
               "properties": {"text": {"type": "string"}}}


def _shout(tool, args):
    text = args.get("text")
    if not isinstance(text, str) or not text or len(text) > 100:
        return _err("bad text")
    return _ok("loud")


def _boom(tool, args):
    raise RuntimeError("always explodes")


def test_explorer_probes_and_ingests_verified_outcomes():
    memory = ToolMemory()
    client = FakeMCP({"shout": ("make loud", TEXT_SCHEMA, _shout)})
    report = asyncio.run(explore_with_client(memory, client, ["shout"]))
    assert report.tools_explored == ["shout"]
    assert report.affordances_verified == 1
    assert report.boundaries_verified == 4
    assert report.unexpected == []
    assert len(report.traces_ingested) == 2
    assert memory.tools["shout"].boundaries
    summary = exploration_summary(report)
    assert summary["probes_executed"] == 5


def test_explorer_records_unexpected_instead_of_ingesting():
    memory = ToolMemory()
    client = FakeMCP({"boom": ("explode", TEXT_SCHEMA, _boom)})
    report = asyncio.run(explore_with_client(memory, client, ["boom"]))
    # The affordance probe contradicts its expectation (recorded, not ingested
    # as a success); the universal rejection itself is valid boundary evidence.
    assert len(report.unexpected) == 1
    assert report.unexpected[0]["probe"] == "minimal-valid-call"
    assert report.affordances_verified == 0
    assert report.boundaries_verified == 4
    assert report.traces_ingested == ["explore-boom-boundary"]
    assert memory.traces["explore-boom-boundary"].failure_count == 4


def test_explorer_refuses_destructive_tools_by_default():
    assert is_destructive("delete_file")
    assert not is_destructive("list_files")
    memory = ToolMemory()
    client = FakeMCP({"delete_file": ("remove", TEXT_SCHEMA, _shout)})
    report = asyncio.run(explore_with_client(memory, client, ["delete_file"]))
    assert report.tools_explored == []
    assert report.unexpected[0]["reason"] == "destructive-refused"
    assert client.calls == []


def test_plan_lists_probes_without_executing():
    specs = [ToolSpec("shout", "make loud", TEXT_SCHEMA)]
    probes = plan_exploration(specs, ["shout"])
    assert {probe.direction for probe in probes} == {"inward", "outward"}
    assert affordance_probes(specs[0])[0].expect_success is True
    assert all(not probe.expect_success for probe in boundary_probes(specs[0]))


def test_run_exploration_needs_no_client():
    memory = ToolMemory()

    async def _prober(tool, args):
        return True, "fine"

    report = asyncio.run(run_exploration(
        memory, [ToolSpec("alpha", "do alpha", TEXT_SCHEMA)],
        _prober, ["alpha"], task_prefix="t",
    ))
    assert isinstance(report, ExplorationReport)
    assert report.affordances_verified == 1


# -- server exposure ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_reverification_due_served_read_only(tmp_path):
    from mcp import Client

    from toolatlas.memory_server import READ_ONLY_TOOLS, create_memory_server

    assert "reverification_due" in READ_ONLY_TOOLS
    db = tmp_path / "mem.db"
    async with Client(create_memory_server(db)) as memory:
        await memory.call_tool("remember_execution", {
            "task_id": "t", "summary": "Do",
            "steps": [{"tool": "a", "rationale": "use a"}],
            "resolved": True,
        })
    async with Client(create_memory_server(db, read_only=True)) as memory:
        result = await memory.call_tool("reverification_due", {"max_age_days": 30})
        assert result.structured_content["count"] == 0


# -- integration template ------------------------------------------------------------

def test_template_orders_sequence_avoid_verify():
    block = format_guidance_block({
        "seed_candidates": [{"summary": "Prior", "confidence": 0.5}],
        "playbook": [{"tool": "alpha", "rationale": "do it"}],
        "avoid": [{"tool": "beta", "caution": "beta rejects empties"}],
        "conventions": [{"text": "Verify the result", "source": "strategy_1"}],
    })
    positions = [
        block.index("1. alpha"),
        block.index("Avoid:"),
        block.index("Verify:"),
    ]
    assert positions == sorted(positions)
    assert block.startswith("[ToolAtlas")
    # Carriage rule: no prior-task summaries, confidence labels, or wrapper.
    assert "Prior" not in block
    assert "confidence" not in block
    assert "related prior task" not in block
    assert "End of ToolAtlas guidance" not in block
