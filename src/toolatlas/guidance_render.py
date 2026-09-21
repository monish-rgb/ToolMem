"""Compact guidance rendering (Plan 1).

The injected block pays carriage on *every* subsequent model request, so it
may contain only turn-earning content: the preferred tool sequence, up to
two failure notes, and a verification step. Prior-task summaries,
confidence labels, and wrapper prose stay in the traversal audit — never in
the model prompt. The token cap is applied *after* rendering: tail steps go
first, then avoid notes; the first step and the verification line are the
floor and are always retained.
"""

from __future__ import annotations

from typing import Any

VERIFY_FALLBACK = "Verify the final state against the task before finishing."
HEADER = "[ToolAtlas verified plan]"
MAX_LINE_CHARS = 200


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _clean(text: object, limit: int = MAX_LINE_CHARS) -> str:
    return " ".join(str(text or "").split())[:limit].strip()


def guidance_is_empty(guidance: object) -> bool:
    if not isinstance(guidance, dict):
        return True
    return not (guidance.get("seed_candidates") or guidance.get("playbook"))


def pick_verify_line(guidance: dict[str, Any]) -> str:
    """First verification-flavored convention, else first convention, else fallback."""
    conventions = guidance.get("conventions") or []
    texts = [_clean(item.get("text", "") if isinstance(item, dict) else item)
             for item in conventions]
    texts = [text for text in texts if text]
    for text in texts:
        if "verif" in text.lower():
            return text
    if texts:
        return texts[0]
    return VERIFY_FALLBACK


def render_compact_block(
    guidance: dict[str, Any],
    max_steps: int = 6,
    token_budget: int = 384,
    suppress_discovery: bool = True,
) -> str:
    """Render the compact injected block; empty retrieval renders nothing."""
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if guidance_is_empty(guidance):
        return ""
    steps = [
        (_clean(step.get("tool", "?")), _clean(step.get("rationale", "")))
        for step in (guidance.get("playbook") or [])[:max_steps]
        if isinstance(step, dict)
    ]
    steps = [(tool, rationale) for tool, rationale in steps if tool and tool != "?"]
    if not steps:
        return ""
    avoid = [
        (_clean(item.get("tool", "?")), _clean(item.get("caution", "")))
        for item in (guidance.get("avoid") or [])[:2]
        if isinstance(item, dict)
    ]
    avoid = [(tool, caution) for tool, caution in avoid if caution]
    verify = pick_verify_line(guidance)

    def _render(kept_steps: list, kept_avoid: list) -> str:
        lines = [f"{HEADER} ({len(kept_steps)} steps)"]
        if suppress_discovery and kept_steps:
            lines.append(f"Directive: Execute step 1 directly ({kept_steps[0][0]}). Do not invoke unneeded discovery tools.")
        lines.extend(
            f"{index + 1}. {tool}: {rationale}" if rationale else f"{index + 1}. {tool}"
            for index, (tool, rationale) in enumerate(kept_steps)
        )
        lines.extend(f"Avoid: {tool}: {caution}" for tool, caution in kept_avoid)
        lines.append(f"Verify: {verify}")
        return "\n".join(lines)

    kept_steps, kept_avoid = list(steps), list(avoid)
    while estimate_tokens(_render(kept_steps, kept_avoid)) > token_budget:
        if len(kept_steps) > 1:
            kept_steps.pop()
        elif kept_avoid:
            kept_avoid.pop()
        else:
            break
    return _render(kept_steps, kept_avoid)
