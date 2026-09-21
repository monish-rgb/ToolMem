"""Tests for history_compress.py."""

from __future__ import annotations

import pytest
from toolatlas.history_compress import compress_tool_result


def test_compress_error_passthrough():
    err = "MCP tool error: file not found"
    assert compress_tool_result("read_file", err, max_chars=10) == err


def test_compress_short_payload_passthrough():
    short = "Hello world"
    assert compress_tool_result("read_file", short, max_chars=100) == short


def test_compress_dir_listing():
    items = [f"file_{i}.txt" for i in range(50)]
    payload = "\n".join(items)
    compressed = compress_tool_result("list_directory", payload, max_chars=100)
    assert "omitted" in compressed
    assert "file_0.txt" in compressed
    assert "file_49.txt" not in compressed


def test_compress_file_content():
    lines = [f"Line {i}: data content" for i in range(100)]
    payload = "\n".join(lines)
    compressed = compress_tool_result("read_file", payload, max_chars=200)
    assert "omitted" in compressed
    assert "Line 0" in compressed
    assert "Line 99" in compressed  # Tail preserved


def test_compress_fallback():
    long_text = "x" * 3000
    compressed = compress_tool_result("unknown_tool", long_text, max_chars=500)
    assert "truncated" in compressed
    assert len(compressed) < 3000
