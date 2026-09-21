"""LLM proposer / reflection / embedding tests (paper pipeline, fakes only).

No network, no credentials, no paid calls: every LLM interaction goes
through ``FakeLLMCall`` with canned responses. Live wiring
(``llm_call_from_env``, ``provider_embedder_from_env``) is tested only for
its refusal behavior without credentials.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from mcp import Client

from toolatlas.embeddings import (
    HashEmbedder,
    cosine_vec,
    vectors_compatible,
)
from toolatlas.embeddings import StoredEmbedding
from toolatlas.llm_client import FakeLLMCall
from toolatlas.llm_proposer import (
    ProposalError,
    propose_probes,
    propose_seed_tasks,
    scrub_proposal_text,
)
from toolatlas.memory import ToolMemory
from toolatlas.memory_server import READ_ONLY_TOOLS, create_memory_server
from toolatlas.models import ExecutionStep, Rollout, ToolSpec
from toolatlas.offline_builder import OfflineCost, build_llm_memory
from toolatlas.reflection import (
    aggregate_capabilities,
    reflect_trace,
)

SEED_JSON = json.dumps({"tasks": [
    {"summary": "Classify files by size into category folders",
     "steps": [{"tool": "alpha", "goal": "list the working directory"},
               {"tool": "beta", "goal": "move each file by size"}]},
    {"summary": "Group documents by size for review",
     "steps": [{"tool": "alpha", "goal": "inspect directory contents"},
               {"tool": "beta", "goal": "relocate files to groups"}]},
    {"summary": "Sort workspace files into size buckets",
     "steps": [{"tool": "alpha", "goal": "establish the file set"},
               {"tool": "beta", "goal": "place files into buckets"}]},
]})

REFLECT_JSON = json.dumps({
    "rationales": ["establish working scope and inputs with alpha",
                   "produce and verify the task output with beta"],
    "tips": ["Verify the final tool result against the task requirement."],
})


def _memory(tmp_path):
    memory = ToolMemory(tmp_path / "m.db")
    memory.register_tools([ToolSpec(name="alpha", description="do alpha"),
                           ToolSpec(name="beta", description="do beta")])
    return memory


# -- embeddings --------------------------------------------------------

def test_hash_embedder_deterministic_and_normalized():
    embedder = HashEmbedder()
    first = embedder.embed(["Classify files by size"])[0]
    second = embedder.embed(["Classify files by size"])[0]
    assert first == second
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9
    near = embedder.embed(["Classify files by size and move them"])[0]
    far = embedder.embed(["Bake sourdough bread slowly"])[0]
    assert cosine_vec(first, near) > cosine_vec(first, far)
    assert cosine_vec([], first) == 0.0
    assert cosine_vec(first, first[:10]) == 0.0


def test_vectors_compatible_requires_same_model_version():
    vec = [1.0, 0.0]
    assert vectors_compatible(StoredEmbedding("q", "m", "v", vec),
                              HashEmbedder(model="m", version="v")) is True
    assert vectors_compatible(StoredEmbedding("q", "m", "v", vec),
                              HashEmbedder(model="m", version="other")) is False
    assert vectors_compatible(None, HashEmbedder()) is False
    assert vectors_compatible(StoredEmbedding("q", "m", "v", []),
                              HashEmbedder(model="m", version="v")) is False


def test_embedding_sidecar_roundtrip_and_invalidation(tmp_path):
    memory = _memory(tmp_path)
    memory.induce("t1", "Classify files by size", [Rollout(
        task_id="t1", summary="s",
        steps=[ExecutionStep("alpha", "use alpha")],
        resolved=True, observation="ok", verifier_type="v")])
    memory.store_embedding("t1", "m", "v", [0.5, 0.5])
    stored = memory.get_embedding("t1", "m", "v")
    assert stored is not None and stored.vector == [0.5, 0.5]
    assert memory.get_embedding("t1", "m", "other") is None
    assert memory.get_embedding("t1", "other", "v") is None
    assert memory.get_embedding("unknown", "m", "v") is None
    with pytest.raises(ValueError):
        memory.store_embedding("unknown", "m", "v", [1.0])


def test_read_only_store_rejects_embedding_writes(tmp_path):
    memory = ToolMemory(tmp_path / "w.db")
    memory.register_tools([ToolSpec(name="alpha", description="do alpha")])
    memory.induce("t1", "Classify files by size", [Rollout(
        task_id="t1", summary="s", steps=[ExecutionStep("alpha", "use alpha")],
        resolved=True, observation="ok", verifier_type="v")])
    from toolatlas.freeze_memory import freeze_memory
    freeze_memory(tmp_path / "w.db", tmp_path / "f.db")
    frozen = ToolMemory(tmp_path / "f.db", read_only=True)
    assert frozen.get_embedding("t1", "m", "v") is None
    with pytest.raises(RuntimeError):
        frozen.store_embedding("t1", "m", "v", [1.0])


# -- proposer ----------------------------------------------------------

def test_propose_seed_tasks_grounded():
    llm = FakeLLMCall(default=SEED_JSON)
    seeds = propose_seed_tasks("alpha", ["alpha", "beta"], llm)
    assert len(seeds) == 3
    assert all("alpha" in {s["tool"] for s in seed.steps} for seed in seeds)
    assert all(seed.summary for seed in seeds)
    assert llm.calls and "alpha" in llm.calls[0]["user"]


def test_propose_seed_tasks_rejects_unknown_tool():
    llm = FakeLLMCall(default=json.dumps({"tasks": [
        {"summary": "Do things", "steps": [{"tool": "gamma", "goal": "mystery"}]},
        {"summary": "Do things", "steps": [{"tool": "alpha", "goal": "x"}]},
        {"summary": "Do things", "steps": [{"tool": "alpha", "goal": "x"}]},
    ]}))
    with pytest.raises(ProposalError):
        propose_seed_tasks("alpha", ["alpha", "beta"], llm)


def test_propose_seed_tasks_rejects_non_json_and_wrong_count():
    with pytest.raises(ProposalError):
        propose_seed_tasks("alpha", ["alpha"], FakeLLMCall(default="not json"))
    with pytest.raises(ProposalError):
        propose_seed_tasks("alpha", ["alpha"],
                           FakeLLMCall(default=json.dumps({"tasks": []})))
    with pytest.raises(ProposalError):
        propose_seed_tasks("gamma", ["alpha"], FakeLLMCall(default=SEED_JSON))


def test_propose_probes_sanitized():
    payload = json.dumps({"affordance": "Verify alpha with key sk-abc123XYZ",
                          "boundary": "Test alpha at C:\\data\\x with 300 items"})
    probes = propose_probes("alpha", [], [], FakeLLMCall(default=payload))
    blob = json.dumps([probes.affordance, probes.boundary])
    assert "sk-abc123XYZ" not in blob and "C:\\data" not in blob
    assert "<secret>" in blob and "<path>" in blob and "<value>" in blob


def test_scrub_proposal_text():
    assert scrub_proposal_text("") == ""
    assert "<secret>" in scrub_proposal_text("api_key: sk-xyzABC123")


# -- reflection --------------------------------------------------------

def test_reflect_trace_grounded():
    llm = FakeLLMCall(default=REFLECT_JSON)
    reflected = reflect_trace(
        "Classify files",
        [{"tool": "alpha", "rationale": ""}, {"tool": "beta", "rationale": ""}],
        [{"resolved": True, "observation": "ok", "verifier_type": "v"}],
        ["alpha", "beta"], llm)
    assert len(reflected.step_rationales) == 2
    assert reflected.tips


def test_reflect_trace_rejects_cross_tool_reference():
    bad = json.dumps({"rationales": ["use gamma instead", "produce output"],
                      "tips": []})
    with pytest.raises(ProposalError):
        reflect_trace("Classify files",
                      [{"tool": "alpha", "rationale": ""},
                       {"tool": "beta", "rationale": ""}],
                      [{"resolved": True, "observation": "ok",
                        "verifier_type": "v"}],
                      ["alpha", "beta", "gamma"],
                      FakeLLMCall(default=bad))
    short = json.dumps({"rationales": ["only one"], "tips": []})
    with pytest.raises(ProposalError):
        reflect_trace("Classify files", [{"tool": "alpha", "rationale": ""},
                                         {"tool": "beta", "rationale": ""}],
                      [{"resolved": True, "observation": "ok",
                        "verifier_type": "v"}],
                      ["alpha", "beta"], FakeLLMCall(default=short))


def test_aggregate_capabilities_rejects_inventions():
    candidates = {"affordances": ["Reliably lists directory contents for review"],
                  "boundaries": ["Avoid empty path inputs that are rejected"]}
    inventing = json.dumps({
        "affordances": ["Reliably lists directory contents for review"],
        "boundaries": ["Teleports files to the moon instantly"]})
    result = aggregate_capabilities(candidates, FakeLLMCall(default=inventing))
    assert result.affordances
    assert all("moon" not in boundary for boundary in result.boundaries)


# -- memory seams ------------------------------------------------------

def _rollout(task_id, tools):
    return Rollout(task_id=task_id, summary="s",
                   steps=[ExecutionStep(tool, f"use {tool}") for tool in tools],
                   resolved=True, observation="ok", verifier_type="v")


def test_induce_reflected_overrides_with_provenance(tmp_path):
    memory = _memory(tmp_path)
    trace = memory.induce(
        "t1", "Classify files by size", [_rollout("t1", ["alpha", "beta"])],
        task_level_tips=["Check sizes before moving files."],
        step_rationales=["scope the directory with alpha",
                         "file each item with beta"],
        induction="llm-reflected")
    assert trace.induction == "llm-reflected"
    assert trace.agent_neutral_trace[0].rationale == "scope the directory with alpha"
    assert trace.task_level_tips == ["Check sizes before moving files."]


def test_induce_reflected_rejects_mismatch_and_strangers(tmp_path):
    memory = _memory(tmp_path)
    with pytest.raises(ValueError):
        memory.induce("t1", "s", [_rollout("t1", ["alpha"])],
                      step_rationales=["one", "two"], induction="llm-reflected")
    with pytest.raises(ValueError):
        memory.induce("t1", "s", [_rollout("t1", ["alpha"])],
                      step_rationales=["invoke gamma instead"],
                      induction="llm-reflected",
                      known_tools=["alpha", "beta", "gamma"])
    with pytest.raises(ValueError):
        memory.induce("t1", "s", [_rollout("t1", ["alpha"])],
                      induction="mystery")


def test_induction_provenance_survives_reload(tmp_path):
    memory = _memory(tmp_path)
    memory.induce("t1", "Classify files", [_rollout("t1", ["alpha"])],
                  induction="llm-reflected")
    reloaded = ToolMemory(tmp_path / "m.db")
    assert reloaded.traces["t1"].induction == "llm-reflected"


def test_guide_embedding_mode_labels_audit(tmp_path):
    memory = _memory(tmp_path)
    memory.induce("t1", "Classify files by size into folders",
                  [_rollout("t1", ["alpha", "beta"])])
    memory.induce("t2", "Bake sourdough bread slowly",
                  [_rollout("t2", ["alpha"])])
    embedder = HashEmbedder()
    guided = memory.guide("Classify files by size", embed_mode="embedding",
                          embedder=embedder)
    assert guided["seed_candidates"]
    assert guided["traversal"]["retrieval"] == "embedding-hash-v1"
    assert guided["coverage"]["retrieval"] == "embedding-hash-v1"
    lexical = memory.guide("Classify files by size")
    assert lexical["traversal"]["retrieval"] == "lexical"
    with pytest.raises(ValueError):
        memory.guide("Classify files", embed_mode="embedding")
    with pytest.raises(ValueError):
        memory.guide("Classify files", embed_mode="semantic")


def test_guide_embedding_paraphrase_recall(tmp_path):
    memory = _memory(tmp_path)
    memory.induce("t1", "Tally tiny files",
                  [_rollout("t1", ["alpha"])])
    paraphrase = "Compute little documents"  # zero lexical/trigram overlap
    assert memory.guide(paraphrase)["seed_candidates"] == []
    first = memory.guide(paraphrase, embed_mode="embedding",
                         embedder=HashEmbedder())
    assert [c["qid"] for c in first["seed_candidates"]] == ["t1"]
    again = memory.guide(paraphrase, embed_mode="embedding",
                         embedder=HashEmbedder())
    assert [c["qid"] for c in again["seed_candidates"]] == ["t1"]
    assert memory.guide("quantum zebra orbit", embed_mode="embedding",
                        embedder=HashEmbedder())["seed_candidates"] == []


# -- server tool -------------------------------------------------------

def test_induce_reflected_tool_and_read_only_exclusion(tmp_path):
    async def _names(db, read_only):
        async with Client(create_memory_server(db, read_only=read_only)) as cli:
            return {t.name for t in (await cli.list_tools()).tools}

    db = tmp_path / "m.db"
    names = asyncio.run(_names(db, False))
    assert "induce_reflected" in names
    assert asyncio.run(_names(db, True)) == set(READ_ONLY_TOOLS)

    async def _induce():
        async with Client(create_memory_server(db)) as cli:
            await cli.call_tool("register_tools", {"tools": [
                {"name": "alpha", "description": "do alpha",
                 "input_schema": {"type": "object"}}]})
            result = await cli.call_tool("induce_reflected", {
                "task_id": "t1", "summary": "Classify files",
                "rollouts": [{"steps": [{"tool": "alpha", "rationale": "use alpha"}],
                              "resolved": True, "observation": "ok",
                              "verifier_type": "v"}],
                "task_level_tips": ["Verify before finishing."],
                "step_rationales": ["scope with alpha"]})
            return result.structured_content

    record = asyncio.run(_induce())
    assert record["induction"] == "llm-reflected"

    async def _refused():
        async with Client(create_memory_server(db, read_only=True)) as cli:
            return await cli.call_tool("induce_reflected", {
                "task_id": "t2", "summary": "Classify files",
                "rollouts": [{"steps": [{"tool": "alpha", "rationale": "x"}],
                              "resolved": True}]})
    assert asyncio.run(_refused()).is_error is True


def test_server_get_guidance_embed_mode(tmp_path):
    memory = _memory(tmp_path)
    memory.induce("t1", "Classify files by size", [_rollout("t1", ["alpha"])])

    async def _call():
        async with Client(create_memory_server(tmp_path / "m.db")) as cli:
            result = await cli.call_tool("get_guidance", {
                "task": "Classify files by size", "embed_mode": "embedding"})
            return result.structured_content

    guided = asyncio.run(_call())
    assert guided["seed_candidates"]
    assert guided["traversal"]["retrieval"] == "embedding-hash-v1"


# -- builder -----------------------------------------------------------

def _executor(task_id, planned):
    return [{"tool": step["tool"], "rationale": step.get("goal") or "do it"}
            for step in planned]


def _verifier(task_id, steps):
    return True, "verified ok"


def test_build_llm_memory_end_to_end(tmp_path):
    beta_seeds = json.dumps({"tasks": [
        {"summary": "Tally beta records into groups",
         "steps": [{"tool": "beta", "goal": "tally each beta record"},
                   {"tool": "beta", "goal": "verify the beta tally"}]},
        {"summary": "Weigh beta records for review",
         "steps": [{"tool": "beta", "goal": "weigh each beta record"},
                   {"tool": "beta", "goal": "verify the beta weights"}]},
        {"summary": "Grade beta records into tiers",
         "steps": [{"tool": "beta", "goal": "grade each beta record"},
                   {"tool": "beta", "goal": "verify the beta grades"}]},
    ]})
    beta_reflect = json.dumps({
        "rationales": ["establish working scope and inputs with beta",
                       "produce and verify the task output with beta"],
        "tips": ["Verify the final tool result against the task requirement."],
    })
    responses = {
        "TARGET TOOL: alpha": SEED_JSON,
        "TARGET TOOL: beta": beta_seeds,
        "TASK: Classify": REFLECT_JSON,
        "TASK: Group": REFLECT_JSON,
        "TASK: Sort": REFLECT_JSON,
        "TASK: Tally": beta_reflect,
        "TASK: Weigh": beta_reflect,
        "TASK: Grade": beta_reflect,
    }
    llm = FakeLLMCall(responses=responses, default=REFLECT_JSON)
    memory = _memory(tmp_path)
    cost = OfflineCost()
    report = build_llm_memory(memory, ["alpha", "beta"], _executor, _verifier,
                              llm, HashEmbedder(), cost=cost)
    assert report["ingested_traces"] == 6
    assert report["reflected_traces"] == 6
    assert report["proposal_fallbacks"] == 0
    assert cost.llm_calls > 0
    assert all(t.induction == "llm-reflected" for t in memory.traces.values())
    assert memory.get_embedding("seed_alpha_1", "hash", "hash-v1") is not None
    guided = memory.guide("Classify files by size", embed_mode="embedding",
                          embedder=HashEmbedder())
    assert guided["seed_candidates"]


def test_build_llm_memory_falls_back_and_counts(tmp_path):
    llm = FakeLLMCall(default="not json at all")
    memory = _memory(tmp_path)
    cost = OfflineCost()
    report = build_llm_memory(memory, ["alpha"], _executor, _verifier, llm,
                              cost=cost)
    assert report["ingested_traces"] == 3
    assert report["reflected_traces"] == 0
    assert report["proposal_fallbacks"] > 0
    assert all(t.induction == "deterministic" for t in memory.traces.values())


def test_build_llm_memory_strict_raise(tmp_path):
    llm = FakeLLMCall(default="not json at all")
    with pytest.raises(Exception):
        build_llm_memory(_memory(tmp_path), ["alpha"], _executor, _verifier,
                         llm, on_proposal_failure="raise")


def test_live_factories_refuse_without_credentials(monkeypatch):
    for key in ("GEMINI_API_KEY", "LLM_API_KEY", "LLM_MODEL", "GEMINI_MODEL",
                "LLM_PROVIDER", "LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    from toolatlas.llm_client import llm_call_from_env
    from toolatlas.embeddings import provider_embedder_from_env
    with pytest.raises(RuntimeError):
        llm_call_from_env()
    with pytest.raises(RuntimeError):
        provider_embedder_from_env()
