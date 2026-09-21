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

GUIDANCE_TEMPLATE = (
    "[ToolAtlas memory guidance — learned from verified prior rollouts]\n"
    "{body}\n"
    "[End of ToolAtlas guidance — use only if it fits the current task]"
)


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


def format_guidance_block(guidance: dict, max_steps: int = 6) -> str:
    """Render one fixed guidance block; empty retrieval renders nothing.

    Order is call-saving density first: planning conventions, then failure
    avoid-notes, then the tool sequence, then provenance candidates.
    """
    if guidance_is_empty(guidance):
        return ""
    lines = []
    conventions = (guidance.get("conventions") or [])[:3]
    if conventions:
        lines.append("Conventions from verified prior work:")
        for item in conventions:
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            lines.append("- %s" % sanitize_text(str(text)))
    avoid = (guidance.get("avoid") or [])[:2]
    if avoid:
        lines.append("Avoid (verified failure modes — do not retry these):")
        for item in avoid:
            if isinstance(item, dict):
                tool = re.sub(r"[^A-Za-z0-9_.\-]", "", str(item.get("tool", "?")))[:64]
                caution = sanitize_text(str(item.get("caution", "")))
                lines.append("- %s: %s" % (tool, caution))
            else:
                lines.append("- %s" % sanitize_text(str(item)))
    steps = (guidance.get("playbook") or [])[:max_steps]
    if steps:
        lines.append("Suggested tool sequence:")
        for step in steps:
            tool = re.sub(r"[^A-Za-z0-9_.\-]", "", str(step.get("tool", "?")))[:64]
            rationale = sanitize_text(str(step.get("rationale", "")))
            lines.append("- %s: %s" % (tool, rationale))
    for cand in (guidance.get("seed_candidates") or [])[:4]:
        summary = sanitize_text(str(cand.get("summary", "")))
        lines.append(
            "- related prior task: %s (confidence %s)"
            % (summary, cand.get("confidence", "?"))
        )
    body = "\n".join(lines).strip()
    if not body:
        return ""
    return GUIDANCE_TEMPLATE.format(body=body)
