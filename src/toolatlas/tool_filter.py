"""Tool-schema pruning based on guidance playbook (Phase 1).

Full tool schemas are ~300 tokens each and are re-sent on every LLM request.
With 14 Filesystem tools, that's ~4200 tokens per request of dead weight when
the guidance playbook only references 3-4 tools. This module prunes the tool
list to just the playbook-referenced tools, saving thousands of tokens per
turn.

Both arms must receive the same pruning decision to keep the comparison fair:
baseline always gets all tools; assisted gets pruned tools when guidance has a
playbook. The pruning decision and list of removed tools are recorded in the
attempt record for audit.
"""

from __future__ import annotations

from typing import Any


def playbook_tool_names(guidance: dict[str, Any] | None) -> set[str]:
    """Extract unique tool names from a guidance playbook."""
    if not guidance:
        return set()
    playbook = guidance.get("playbook") or []
    return {
        step.get("tool", "")
        for step in playbook
        if isinstance(step, dict) and step.get("tool")
    }


def filter_tools_by_playbook(
    all_tools: list[dict[str, Any]],
    guidance: dict[str, Any] | None,
    *,
    always_include: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter tool schemas to playbook-referenced tools only.

    Returns ``(filtered_tools, audit)`` where ``audit`` records what was
    pruned and why, for the attempt record.

    When the guidance has no playbook (empty retrieval, baseline arm, or
    retrieval miss), all tools are returned unchanged — the agent needs
    full discovery access.

    ``always_include`` names tools that must stay regardless (e.g. safety
    critical tools that the agent may need for error recovery).
    """
    keep_names = playbook_tool_names(guidance)
    if not keep_names:
        return list(all_tools), {
            "pruned": False,
            "reason": "no_playbook",
            "kept": len(all_tools),
            "removed": 0,
            "removed_tools": [],
        }

    always = always_include or set()
    keep_names = keep_names | always

    # Also include tools referenced in avoid notes — the agent needs their
    # schemas to understand what *not* to call (some models inspect schemas
    # even when told to avoid a tool).
    avoid_tools = set()
    for item in (guidance or {}).get("avoid") or []:
        if isinstance(item, dict) and item.get("tool"):
            avoid_tools.add(item["tool"])
    keep_names = keep_names | avoid_tools

    kept: list[dict[str, Any]] = []
    removed_names: list[str] = []
    for tool in all_tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        if name in keep_names:
            kept.append(tool)
        else:
            removed_names.append(name)

    return kept, {
        "pruned": True,
        "reason": "playbook_filter",
        "kept": len(kept),
        "removed": len(removed_names),
        "removed_tools": sorted(removed_names),
        "playbook_tools": sorted(playbook_tool_names(guidance)),
        "always_include": sorted(always),
        "avoid_tools": sorted(avoid_tools),
    }


def estimate_schema_tokens(tools: list[dict[str, Any]]) -> int:
    """Rough token estimate for a tool-schema list (chars // 4)."""
    import json
    text = json.dumps(tools, sort_keys=True, separators=(",", ":"))
    return max(1, len(text) // 4)
