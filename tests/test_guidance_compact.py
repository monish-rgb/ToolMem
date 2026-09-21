"""Plan 1/2 tests: compact guidance rendering and result truncation."""

from __future__ import annotations

import pytest

from toolatlas.guidance_render import (
    HEADER,
    VERIFY_FALLBACK,
    estimate_tokens,
    guidance_is_empty,
    pick_verify_line,
    render_compact_block,
)
from toolatlas.result_limit import truncate_result

GUIDANCE = {
    "seed_candidates": [{"summary": "Prior work", "confidence": 0.9}],
    "playbook": [
        {"tool": "list_directory", "rationale": "inspect the workspace"},
        {"tool": "create_directory", "rationale": "create category folders"},
        {"tool": "get_file_info", "rationale": "read file metadata"},
        {"tool": "move_file", "rationale": "move each file"},
    ],
    "avoid": [
        {"tool": "keyword_count", "caution": "empty keywords are rejected"},
        {"tool": "move_file", "caution": "missing parents fail the move"},
        {"tool": "extra", "caution": "third note must be cut"},
    ],
    "conventions": [{"text": "Verify the final tool result first.", "source": "s1"}],
}


def test_compact_block_sections_and_order():
    block = render_compact_block(GUIDANCE)
    assert block.startswith(HEADER)
    seq = block.index("1. list_directory")
    avoid = block.index("Avoid: keyword_count")
    verify = block.index("Verify:")
    assert seq < avoid < verify
    # Only sequence, avoid notes, verification: nothing else leaks in.
    assert "Prior work" not in block
    assert "confidence" not in block
    assert "related prior task" not in block
    assert "End of ToolAtlas guidance" not in block
    # At most two failure notes.
    assert block.count("Avoid:") == 2
    assert "third note must be cut" not in block


def test_empty_guidance_renders_nothing():
    assert render_compact_block({}) == ""
    assert render_compact_block({"seed_candidates": [], "playbook": []}) == ""
    assert guidance_is_empty({}) is True


def test_verify_line_prefers_verification_convention():
    assert pick_verify_line(GUIDANCE) == "Verify the final tool result first."
    assert pick_verify_line({"conventions": [{"text": "Scope first"}]}) == "Scope first"
    assert pick_verify_line({}) == VERIFY_FALLBACK


def test_cap_applied_after_rendering_trims_tail_first():
    tiny = render_compact_block(GUIDANCE, token_budget=64)
    assert estimate_tokens(tiny) <= 64 or "1. list_directory" in tiny
    assert "1. list_directory" in tiny  # head step is the floor
    assert tiny.rstrip().endswith(
        "Verify: Verify the final tool result first.")  # verify line kept
    full = render_compact_block(GUIDANCE, token_budget=384)
    assert "4. move_file" in full  # generous budget keeps the tail
    assert estimate_tokens(full) <= 384
    with pytest.raises(ValueError):
        render_compact_block(GUIDANCE, token_budget=0)


def test_truncate_result_passthrough_and_marker():
    assert truncate_result("short") == "short"
    assert truncate_result("", 100) == ""
    assert truncate_result("x" * 2000, 2000) == "x" * 2000
    long_text = "y" * 5000
    cut = truncate_result(long_text, 2000)
    assert len(cut) < len(long_text)
    assert "truncated" in cut and "5000" in cut and cut.startswith("y" * 2000)
    with pytest.raises(ValueError):
        truncate_result("x", 0)


def test_truncate_result_preserves_errors():
    error = "MCP tool error: " + "z" * 5000
    assert truncate_result(error, 2000) == error
    metadata = "size: 120\nmodified: Mon Jul 09 2025"  # short file metadata
    assert truncate_result(metadata, 2000) == metadata
