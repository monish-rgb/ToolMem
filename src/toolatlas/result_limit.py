"""Shared deterministic tool-result size limit (Plan 2, separate ablation).

Long tool outputs accumulate in conversation history and are resent on every
later request, so repeated history can dwarf any one-time prompt edit. This
helper applies one identical policy in both arms: a generous character
limit, verbatim errors (recovery depends on them), and a clear truncation
marker. Short results — including file-metadata outputs — pass through
untouched. Savings from this limit must be reported as their own ablation,
never folded into the ToolAtlas guidance effect.
"""

from __future__ import annotations

ERROR_PREFIX = "MCP tool error:"


def truncate_result(text: object, limit: int = 8000) -> str:
    """Cap a provider-tool result payload deterministically."""
    if limit < 1:
        raise ValueError("limit must be positive")
    payload = "" if text is None else str(text)
    if len(payload) <= limit:
        return payload
    if payload.startswith(ERROR_PREFIX):
        return payload  # error detail is recovery-critical; never cut it
    kept = payload[:limit]
    return (
        f"{kept}\n[truncated: showing {len(kept)} of {len(payload)} chars; "
        "re-run a narrower call for the rest]"
    )
