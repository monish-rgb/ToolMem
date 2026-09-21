"""Sanitization and guidance formatting for the ToolAtlas MCPMark integration.

Everything here is pure (no MCPMark or ToolAtlas imports) so it can be unit
tested in isolation. Rules mirror the training protocol: agent-neutral,
environment-invariant memory — no secrets, paths, task literals, account IDs,
answer content, or agent syntax reaches the memory writer.
"""

from __future__ import annotations

import hashlib
import json
import re

SECRET_PATTERNS = (
    re.compile(r"(?i)(api[_-]?key|token|secret|password|bearer)\s*[:=]\s*\S+"),
    re.compile(r"sk-[A-Za-z0-9]{8,}"),
    re.compile(r"nvapi-[A-Za-z0-9_.\-~+/=]+"),
    re.compile(r"xox[bap]-[A-Za-z0-9-]+"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]+"),
)
PATH_PATTERN = re.compile(r"(?:[A-Za-z]:)?[/\\][\w./\\\-~]+")
QUOTED_PATTERN = re.compile(r"(?s)(['\"]).*?\1")
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?\b")
WHITESPACE_PATTERN = re.compile(r"\s+")

GUIDANCE_HEADER = "[ToolAtlas verified plan]"
VERIFY_FALLBACK = "Verify the final state against the task before finishing."
MAX_LINE_CHARS = 200


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 4)


def _clean_line(text: object, limit: int = MAX_LINE_CHARS) -> str:
    return WHITESPACE_PATTERN.sub(" ", str(text or "")).strip()[:limit].strip()


def sanitize_text(text: str, limit: int = 1000) -> str:
    """Strip secrets, paths, quoted literals, and numbers from free text."""
    if not isinstance(text, str):
        text = str(text)
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("<secret>", text)
    text = PATH_PATTERN.sub("<path>", text)
    text = QUOTED_PATTERN.sub("<value>", text)
    text = NUMBER_PATTERN.sub("<value>", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()[:limit]


def sanitize_args(args: object, limit: int = 2000) -> object:
    """Recursively sanitize tool arguments, preserving structure."""
    if isinstance(args, dict):
        return {str(key)[:128]: sanitize_args(value, limit) for key, value in args.items()}
    if isinstance(args, (list, tuple)):
        return [sanitize_args(value, limit) for value in args]
    if isinstance(args, str):
        return sanitize_text(args, limit=min(limit, 500))
    if isinstance(args, bool) or args is None:
        return args
    if isinstance(args, (int, float)):
        return "<value>"
    return args


def summarize_result(result: object, limit: int = 500) -> str:
    """Summarize a tool result without storing answer content."""
    try:
        text = json.dumps(result, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(result)
    if isinstance(result, dict) and result.get("isError"):
        return "tool reported an error"
    return sanitize_text(text, limit=limit)


def fingerprint_tool(spec: dict) -> str:
    """Hash a provider tool schema for change detection."""
    canonical = json.dumps(
        {
            "name": spec.get("name", ""),
            "description": spec.get("description", ""),
            "input_schema": spec.get("input_schema", {}),
            "version": spec.get("version", "unknown"),
            "provider": spec.get("provider", "unknown"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def retrieval_query(instruction: str) -> str:
    """Use the exact official task instruction as the retrieval query."""
    return instruction.strip()


def guidance_is_empty(guidance: object) -> bool:
    """Empty lexical retrieval is valid: no candidates and no playbook."""
    if not isinstance(guidance, dict):
        return True
    candidates = guidance.get("seed_candidates") or []
    playbook = guidance.get("playbook") or []
    return not candidates and not playbook


def format_guidance_block(guidance: dict, max_steps: int = 6,
                          token_budget: int = 384) -> str:
    """Render one compact guidance block; empty retrieval renders nothing.

    Carriage rule (Plan 1): only the preferred tool sequence, up to two
    failure notes, and a verification step enter the model prompt.
    Prior-task summaries, confidence labels, and wrapper prose stay in the
    traversal audit. The token cap applies *after* rendering — tail steps go
    first, then avoid notes; the first step and the verification line are
    always retained.
    """
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if guidance_is_empty(guidance):
        return ""
    steps = [
        (_clean_line(re.sub(r"[^A-Za-z0-9_.\-]", "", str(step.get("tool", "?")))[:64]),
         _clean_line(step.get("rationale", "")))
        for step in (guidance.get("playbook") or [])[:max_steps]
        if isinstance(step, dict)
    ]
    steps = [(tool, rationale) for tool, rationale in steps if tool and tool != "?"]
    if not steps:
        return ""
    avoid = [
        (_clean_line(re.sub(r"[^A-Za-z0-9_.\-]", "", str(item.get("tool", "?")))[:64]),
         _clean_line(item.get("caution", "")))
        for item in (guidance.get("avoid") or [])[:2]
        if isinstance(item, dict)
    ]
    avoid = [(tool, caution) for tool, caution in avoid if caution]
    verify = VERIFY_FALLBACK
    for item in (guidance.get("conventions") or [])[:3]:
        text = _clean_line(item.get("text", "") if isinstance(item, dict) else item)
        if text:
            verify = text
            if "verif" in text.lower():
                break

    def _render(kept_steps: list, kept_avoid: list) -> str:
        lines = ["%s (%d steps)" % (GUIDANCE_HEADER, len(kept_steps))]
        for index, (tool, rationale) in enumerate(kept_steps):
            lines.append("%d. %s: %s" % (index + 1, tool, sanitize_text(rationale))
                         if rationale else "%d. %s" % (index + 1, tool))
        for tool, caution in kept_avoid:
            lines.append("Avoid: %s: %s" % (tool, sanitize_text(caution)))
        lines.append("Verify: %s" % sanitize_text(verify))
        return "\n".join(lines)

    kept_steps, kept_avoid = list(steps), list(avoid)
    while _estimate_tokens(_render(kept_steps, kept_avoid)) > token_budget:
        if len(kept_steps) > 1:
            kept_steps.pop()
        elif kept_avoid:
            kept_avoid.pop()
        else:
            break
    return _render(kept_steps, kept_avoid)
