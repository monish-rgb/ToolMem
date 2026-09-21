"""Tests for the read-only memory profile and the freeze helper."""

from __future__ import annotations

import json
import sqlite3

import pytest
from mcp import Client

from toolatlas.freeze_memory import freeze_memory
from toolatlas.memory_server import READ_ONLY_TOOLS, create_memory_server


async def _seed_async(path):
    server = create_memory_server(path)
    async with Client(server) as memory:
        await memory.call_tool("register_tools", {"tools": [
            {"name": "alpha", "description": "do alpha",
             "input_schema": {"type": "object"}},
        ]})
        await memory.call_tool("remember_execution", {
            "task_id": "t1", "summary": "Do alpha",
            "steps": [{"tool": "alpha", "rationale": "use alpha"}],
            "resolved": True,
        })
    return server


def _seed(path):
    import asyncio
    return asyncio.run(_seed_async(path))


async def _tool_names(server):
    async with Client(server) as memory:
        listed = await memory.list_tools()
        return {t.name for t in listed.tools}


def test_read_only_profile_exposes_only_retrieval_and_inspection(tmp_path):
    import asyncio

    db = tmp_path / "mem.db"
    _seed(db)
    names = asyncio.run(_tool_names(create_memory_server(db, read_only=True)))
    assert names == set(READ_ONLY_TOOLS)
    assert "register_tools" not in names
    assert "remember_execution" not in names
    assert "reverify_trace" not in names
    assert "set_trace_status" not in names


@pytest.mark.asyncio
async def test_read_only_reads_still_work(tmp_path):
    db = tmp_path / "mem.db"
    await _seed_async(db)
    async with Client(create_memory_server(db, read_only=True)) as memory:
        stats = await memory.call_tool("memory_stats", {})
        assert stats.structured_content["traces"] == 1
        guidance = await memory.call_tool("get_guidance", {"task": "Do alpha"})
        assert guidance.structured_content["seed_candidates"]


@pytest.mark.asyncio
async def test_read_only_write_call_fails(tmp_path):
    db = tmp_path / "mem.db"
    await _seed_async(db)
    async with Client(create_memory_server(db, read_only=True)) as memory:
        # Unregistered write tools cannot succeed: the server answers with an
        # error result (same contract as the boundary probe in demo.py).
        result = await memory.call_tool("remember_execution", {
            "task_id": "t2", "summary": "Do alpha",
            "steps": [{"tool": "alpha", "rationale": "use alpha"}],
            "resolved": True,
        })
        assert getattr(result, "is_error", False) is True
        # And nothing was written.
        stats = await memory.call_tool("memory_stats", {})
        assert stats.structured_content["traces"] == 1


def test_freeze_protocol(tmp_path):
    src = tmp_path / "training.db"
    dst = tmp_path / "frozen.db"
    _seed(src)
    record = freeze_memory(src, dst)
    assert record["integrity_check"] == "ok"
    assert record["stats"]["traces"] == 1
    assert record["stats"]["tools"] == 1
    assert len(record["frozen_sha256"]) == 64
    # Frozen copy is stable and self-contained.
    again = freeze_memory(src, tmp_path / "frozen2.db")
    assert again["frozen_sha256"] == record["frozen_sha256"]
    ro = sqlite3.connect(f"file:{dst.resolve()}?mode=ro", uri=True)
    try:
        assert ro.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
    finally:
        ro.close()


def test_freeze_rejects_missing_source_and_self_copy(tmp_path):
    with pytest.raises(FileNotFoundError):
        freeze_memory(tmp_path / "nope.db", tmp_path / "out.db")
    src = tmp_path / "a.db"
    _seed(src)
    with pytest.raises(ValueError):
        freeze_memory(src, src)


def test_frozen_stats_are_json_serializable(tmp_path):
    src = tmp_path / "s.db"
    _seed(src)
    record = freeze_memory(src, tmp_path / "f.db")
    json.dumps(record)
