"""History & Tool Result Compression.

Provides tool-aware compression for MCP tool outputs appended to conversation history.
Applied identically to both Arm A and Arm B in A/B evaluation harnesses.
"""

from __future__ import annotations

import json
from typing import Any

from .result_limit import ERROR_PREFIX, truncate_result


def compress_tool_result(tool_name: str, payload: str, max_chars: int = 2000) -> str:
    """Compress tool results for conversation history.
    
    Preserves verbatim errors. Appends summary metadata if truncated.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if not payload or payload.startswith(ERROR_PREFIX):
        return payload

    if len(payload) <= max_chars:
        return payload

    # Tool-specific smart compression
    name_lower = tool_name.lower()
    
    if "list" in name_lower or "dir" in name_lower:
        # Directory listing
        lines = payload.splitlines()
        if len(lines) > 20:
            kept = lines[:15]
            return "\n".join(kept) + f"\n... [{len(lines) - 15} items omitted; total {len(lines)} items]"
            
    elif "read" in name_lower or "content" in name_lower:
        # Text/File reading
        lines = payload.splitlines()
        if len(lines) > 40:
            head = lines[:30]
            tail = lines[-5:]
            return "\n".join(head) + f"\n... [{len(lines) - 35} lines omitted; total {len(lines)} lines]\n" + "\n".join(tail)

    # Standard fallback
    return truncate_result(payload, limit=max_chars)
