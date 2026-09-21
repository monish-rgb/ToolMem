"""Retrieval and traversal metrics (plan Phase 4).

The lexical/trigram retriever stays the deterministic fallback. These
helpers score what retrieval did — separately from what the model did —
so budget and threshold tuning can use training tasks only.
"""

from __future__ import annotations

from typing import Any

from .memory import estimate_tokens


def guidance_size_tokens(guidance: dict[str, Any]) -> int:
    """Actual injected-block size class estimate for cost accounting."""
    import json as _json
    return estimate_tokens(_json.dumps({
        "playbook": guidance.get("playbook", []),
        "avoid": guidance.get("avoid", []),
        "conventions": guidance.get("conventions", []),
    }, sort_keys=True))


def retrieval_metrics(
    guidances: list[dict[str, Any]],
    *,
    irrelevant_flags: list[bool] | None = None,
    baseline_turns: list[int] | None = None,
    assisted_turns: list[int] | None = None,
    baseline_repeats: list[int] | None = None,
    assisted_repeats: list[int] | None = None,
    baseline_success: list[bool] | None = None,
    assisted_success: list[bool] | None = None,
) -> dict[str, Any]:
    """Summarize retrieval behavior over a set of assisted attempts."""
    n = len(guidances)
    empty = sum(1 for g in guidances
                if not (g.get("seed_candidates") or g.get("playbook")))
    sizes = [guidance_size_tokens(g) for g in guidances]
    covered_tools = sorted({step.get("tool") for g in guidances
                            for step in (g.get("playbook") or [])
                            if step.get("tool")})
    result: dict[str, Any] = {
        "n": n,
        "empty_guidance_rate": round(empty / n, 4) if n else 0.0,
        "guidance_coverage_tools": covered_tools,
        "mean_guidance_tokens": round(sum(sizes) / len(sizes), 2) if sizes else 0.0,
        "max_guidance_tokens": max(sizes) if sizes else 0,
    }
    if irrelevant_flags is not None:
        if len(irrelevant_flags) != n:
            raise ValueError("irrelevant_flags must align with guidances")
        result["irrelevant_guidance_rate"] = round(sum(1 for f in irrelevant_flags if f) / n, 4) if n else 0.0
    if baseline_turns is not None and assisted_turns is not None:
        pairs = [(b - a) for b, a in zip(baseline_turns, assisted_turns)]
        result["mean_turns_avoided"] = round(sum(pairs) / len(pairs), 2) if pairs else 0.0
    if baseline_repeats is not None and assisted_repeats is not None:
        pairs = [(b - a) for b, a in zip(baseline_repeats, assisted_repeats)]
        result["mean_repeated_calls_avoided"] = round(sum(pairs) / len(pairs), 2) if pairs else 0.0
    if baseline_success is not None and assisted_success is not None:
        negative = sum(1 for b, a in zip(baseline_success, assisted_success) if b and not a)
        result["negative_transfer_rate"] = round(negative / len(baseline_success), 4) if baseline_success else 0.0
    return result
