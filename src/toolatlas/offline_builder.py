"""Paper-style offline memory builder (plan Phase 3, feature-flagged).

The deterministic ``ToolMemory.induce`` path remains the default control.
This builder adds the paper-aligned protocol as an explicit opt-in:
seed-task bootstrapping (3 tasks per tool, 4 rollouts each), programmatic
verification, shortest-verified backbone, failed-rollouts-as-boundaries,
agent-neutral positional rationales, three exploration rounds with novelty
tracking, and full offline construction-cost accounting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .memory import ToolMemory
from .models import ExecutionStep, Rollout

Verifier = Callable[[str, list[dict[str, str]]], tuple[bool, str]]
Executor = Callable[[str, list[dict[str, str]]], list[dict[str, str]]]

SEEDS_PER_TOOL = 3
ROLLOUTS_PER_SEED = 4
EXPLORATION_ROUNDS = 3


@dataclass(slots=True)
class OfflineCost:
    """Offline construction cost (kept separate from online inference)."""

    training_attempts: int = 0
    training_failures: int = 0
    exploration_proposals: int = 0
    exploration_verified: int = 0
    offline_input_tokens: int = 0
    offline_provider_calls: int = 0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    proposal_fallbacks: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "training_attempts": self.training_attempts,
            "training_failures": self.training_failures,
            "exploration_proposals": self.exploration_proposals,
            "exploration_verified": self.exploration_verified,
            "offline_input_tokens": self.offline_input_tokens,
            "offline_provider_calls": self.offline_provider_calls,
            "llm_calls": self.llm_calls,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "proposal_fallbacks": self.proposal_fallbacks,
        }


def propose_seed_tasks(tool: str, count: int = SEEDS_PER_TOOL) -> list[dict[str, str]]:
    """Deterministic seed-task proposals per target tool (no model needed)."""
    return [
        {"task_id": f"seed_{tool}_{i + 1}",
         "summary": f"Verify a representative workflow using {tool} (seed {i + 1})"}
        for i in range(count)
    ]


def select_backbone(rollouts: list[Rollout]) -> Rollout:
    """Shortest verified successful sequence is the trace backbone."""
    successful = [r for r in rollouts if r.resolved]
    candidates = successful or rollouts
    return min(candidates, key=lambda r: len(r.steps))


def build_offline_memory(
    memory: ToolMemory,
    tools: list[str],
    execute: Executor,
    verify: Verifier,
    *,
    cost: OfflineCost | None = None,
    seeds_per_tool: int = SEEDS_PER_TOOL,
    rollouts_per_seed: int = ROLLOUTS_PER_SEED,
) -> dict[str, Any]:
    """Bootstrap memory from verified rollouts; failures become boundaries."""
    owned = cost or OfflineCost()
    ingested = 0
    for tool in tools:
        for seed in propose_seed_tasks(tool, seeds_per_tool):
            rollouts: list[Rollout] = []
            for run in range(rollouts_per_seed):
                steps = execute(seed["task_id"], [{"tool": tool, "rationale": ""}])
                resolved, observation = verify(seed["task_id"], steps)
                owned.training_attempts += 1
                owned.training_failures += int(not resolved)
                rollouts.append(Rollout(
                    task_id=f"{seed['task_id']}_r{run + 1}",
                    summary=seed["summary"],
                    steps=[__import__("toolatlas.models", fromlist=["ExecutionStep"]).ExecutionStep(
                        s.get("tool", tool), s.get("rationale", "")) for s in steps],
                    resolved=resolved,
                    observation=observation,
                    verifier_type="offline-builder-verifier",
                ))
            # One trace per seed task: backbone + boundary evidence preserved.
            backbone_task = seed["task_id"]
            memory.induce(backbone_task, seed["summary"], rollouts)
            ingested += 1
    return {"ingested_traces": ingested, "cost": owned.to_dict()}


def run_exploration_rounds(
    memory: ToolMemory,
    execute: Executor,
    verify: Verifier,
    *,
    cost: OfflineCost | None = None,
    rounds: int = EXPLORATION_ROUNDS,
) -> dict[str, Any]:
    """Three exploration rounds over under-covered capabilities.

    Only verified outcomes are ingested: confirmations become affordances,
    rejections become boundaries. Proposal novelty is tracked so later
    rounds do not repeat already-covered behavior.
    """
    owned = cost or OfflineCost()
    seen: set[str] = set()
    verified = 0
    for round_no in range(1, rounds + 1):
        proposals: list[dict[str, str]] = []
        for tool_name in sorted(memory.tools):
            try:
                probes = memory.suggest_probes(tool_name)
            except ValueError:
                continue
            for direction in ("affordance", "boundary"):
                for probe in probes.get(direction, []):
                    key = f"{tool_name}:{direction}:{probe.get('task', '')}"
                    if key not in seen:
                        seen.add(key)
                        proposals.append({"tool": tool_name, "direction": direction,
                                          "task": probe.get("task", "")})
        owned.exploration_proposals += len(proposals)
        for index, proposal in enumerate(proposals):
            task_id = f"explore_r{round_no}_{proposal['tool']}_{index}"
            steps = execute(task_id, [{"tool": proposal["tool"], "rationale": ""}])
            resolved, observation = verify(task_id, steps)
            memory.induce(task_id, proposal["task"], [Rollout(
                task_id=task_id, summary=proposal["task"],
                steps=[__import__("toolatlas.models", fromlist=["ExecutionStep"]).ExecutionStep(
                    s.get("tool", proposal["tool"]), s.get("rationale", "")) for s in steps],
                resolved=resolved, observation=observation,
                verifier_type="offline-exploration-verifier")])
            owned.exploration_verified += 1
            verified += 1
    return {"rounds": rounds, "verified": verified, "cost": owned.to_dict()}


def build_llm_memory(
    memory: ToolMemory,
    tools: list[str],
    execute: Executor,
    verify: Verifier,
    llm: Any,
    embedder: Any | None = None,
    *,
    cost: OfflineCost | None = None,
    seeds_per_tool: int = SEEDS_PER_TOOL,
    rollouts_per_seed: int = ROLLOUTS_PER_SEED,
    on_proposal_failure: str = "fallback",
) -> dict[str, Any]:
    """Paper-style construction: LLM-proposed seeds + reflected induction.

    ``llm`` is an injected ``LLMCall`` (fake in tests, live via
    ``llm_call_from_env``). Proposal failures either fall back to the
    deterministic template (counted in ``cost.proposal_fallbacks``) or raise,
    per ``on_proposal_failure``. Every ingested trace is labeled
    ``induction="llm-reflected"`` for audit.
    """
    from .llm_proposer import ProposalError
    from .llm_proposer import propose_seed_tasks as llm_propose_seed_tasks
    from .reflection import reflect_trace
    if on_proposal_failure not in ("fallback", "raise"):
        raise ValueError("on_proposal_failure must be 'fallback' or 'raise'")
    owned = cost or OfflineCost()
    ingested = 0
    reflected = 0
    for tool in tools:
        allowed = sorted(set(tools))
        try:
            seeds = llm_propose_seed_tasks(tool, allowed, llm, count=seeds_per_tool, cost=owned)
            seed_dicts = [{"task_id": s.task_id, "summary": s.summary, "steps": s.steps}
                          for s in seeds]
        except ProposalError:
            if on_proposal_failure == "raise":
                raise
            owned.proposal_fallbacks += 1
            seed_dicts = propose_seed_tasks(tool, seeds_per_tool)
            seed_dicts = [{"task_id": s["task_id"], "summary": s["summary"],
                           "steps": [{"tool": tool, "goal": ""}]} for s in seed_dicts]
        for seed in seed_dicts:
            rollouts: list[Rollout] = []
            for run in range(rollouts_per_seed):
                planned = [{"tool": s.get("tool", tool),
                            "rationale": s.get("goal", "")} for s in seed["steps"]]
                steps = execute(seed["task_id"], planned)
                resolved, observation = verify(seed["task_id"], steps)
                owned.training_attempts += 1
                owned.training_failures += int(not resolved)
                rollouts.append(Rollout(
                    task_id=f"{seed['task_id']}_r{run + 1}",
                    summary=seed["summary"],
                    steps=[ExecutionStep(
                        s.get("tool", tool), s.get("rationale", "")) for s in steps],
                    resolved=resolved,
                    observation=observation,
                    verifier_type="llm-builder-verifier",
                ))
            backbone = select_backbone(rollouts)
            outcomes = [{"resolved": r.resolved, "observation": r.observation,
                         "verifier_type": r.verifier_type} for r in rollouts]
            try:
                reflected_trace = reflect_trace(
                    seed["summary"],
                    [{"tool": s.tool, "rationale": s.rationale}
                     for s in backbone.steps],
                    outcomes, allowed, llm, cost=owned)
                tips: list[str] | None = reflected_trace.tips
                rationales: list[str] | None = reflected_trace.step_rationales
                provenance = "llm-reflected"
                reflected += 1
            except ProposalError:
                if on_proposal_failure == "raise":
                    raise
                owned.proposal_fallbacks += 1
                tips, rationales, provenance = None, None, "deterministic"
            trace = memory.induce(
                seed["task_id"], seed["summary"], rollouts,
                task_level_tips=tips, step_rationales=rationales,
                induction=provenance, known_tools=allowed)
            if embedder is not None:
                try:
                    vector = embedder.embed([seed["summary"]])[0]
                    memory.store_embedding(trace.qid, embedder.model,
                                           embedder.version, vector)
                except (ValueError, RuntimeError):
                    pass  # in-memory or frozen DBs skip the vector cache
            ingested += 1
    return {"ingested_traces": ingested, "reflected_traces": reflected,
            "proposal_fallbacks": owned.proposal_fallbacks,
            "cost": owned.to_dict()}


@dataclass(slots=True)
class BudgetTuning:
    """Training-only guidance-budget selection with a pre-declared utility."""

    budgets: list[int] = field(default_factory=lambda: [128, 256, 384, 512])
    selected: int = 384
    utilities: dict[int, float] = field(default_factory=dict)


def select_guidance_budget(
    success_rate: dict[int, float],
    mean_input_tokens: dict[int, float],
    *,
    lambda_weight: float,
) -> BudgetTuning:
    """Utility: ``success_rate - lambda * normalized_online_input_tokens``.

    ``lambda_weight`` must be locked in the manifest before selection, and
    selection must use training tasks only — never evaluation tasks.
    """
    if lambda_weight < 0:
        raise ValueError("lambda_weight must be non-negative")
    budgets = sorted(set(success_rate) & set(mean_input_tokens))
    if not budgets:
        raise ValueError("no shared budgets to compare")
    ceiling = max(mean_input_tokens[b] for b in budgets) or 1.0
    utilities = {
        b: success_rate[b] - lambda_weight * (mean_input_tokens[b] / ceiling)
        for b in budgets
    }
    selected = max(budgets, key=lambda b: (utilities[b], -b))
    return BudgetTuning(budgets=budgets, selected=selected, utilities=utilities)
