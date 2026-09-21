"""Tests for tool-schema pruning (Phase 1)."""

from __future__ import annotations

import pytest

from toolatlas.tool_filter import (
    estimate_schema_tokens,
    filter_tools_by_playbook,
    playbook_tool_names,
)


def _tool(name: str, desc: str = "") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc or f"Description of {name}",
            "parameters": {"type": "object", "properties": {}},
        },
    }


SAMPLE_TOOLS = [
    _tool("list_directory"),
    _tool("directory_tree"),
    _tool("search_files"),
    _tool("read_text_file"),
    _tool("get_file_info"),
    _tool("create_directory"),
    _tool("move_file"),
    _tool("write_file"),
]


class TestPlaybookToolNames:
    def test_empty_guidance(self):
        assert playbook_tool_names(None) == set()
        assert playbook_tool_names({}) == set()
        assert playbook_tool_names({"playbook": []}) == set()

    def test_extracts_tool_names(self):
        guidance = {
            "playbook": [
                {"tool": "search_files", "rationale": "find the file"},
                {"tool": "read_text_file", "rationale": "read it"},
            ]
        }
        assert playbook_tool_names(guidance) == {"search_files", "read_text_file"}

    def test_skips_invalid_steps(self):
        guidance = {
            "playbook": [
                {"tool": "search_files"},
                "not_a_dict",
                {"no_tool_key": True},
                {"tool": ""},
            ]
        }
        assert playbook_tool_names(guidance) == {"search_files"}


class TestFilterToolsByPlaybook:
    def test_no_playbook_returns_all(self):
        tools, audit = filter_tools_by_playbook(SAMPLE_TOOLS, None)
        assert len(tools) == len(SAMPLE_TOOLS)
        assert audit["pruned"] is False
        assert audit["reason"] == "no_playbook"
        assert audit["removed"] == 0

    def test_empty_playbook_returns_all(self):
        tools, audit = filter_tools_by_playbook(
            SAMPLE_TOOLS, {"playbook": []}
        )
        assert len(tools) == len(SAMPLE_TOOLS)
        assert audit["pruned"] is False

    def test_playbook_prunes_tools(self):
        guidance = {
            "playbook": [
                {"tool": "search_files", "rationale": "find"},
                {"tool": "read_text_file", "rationale": "read"},
            ]
        }
        tools, audit = filter_tools_by_playbook(SAMPLE_TOOLS, guidance)
        names = {t["function"]["name"] for t in tools}
        assert names == {"search_files", "read_text_file"}
        assert audit["pruned"] is True
        assert audit["removed"] == 6
        assert "list_directory" in audit["removed_tools"]
        assert "directory_tree" in audit["removed_tools"]

    def test_always_include_preserved(self):
        guidance = {
            "playbook": [{"tool": "search_files", "rationale": "find"}]
        }
        tools, audit = filter_tools_by_playbook(
            SAMPLE_TOOLS, guidance, always_include={"get_file_info"}
        )
        names = {t["function"]["name"] for t in tools}
        assert "search_files" in names
        assert "get_file_info" in names
        assert "always_include" in audit
        assert "get_file_info" in audit["always_include"]

    def test_avoid_tools_preserved(self):
        guidance = {
            "playbook": [{"tool": "search_files", "rationale": "find"}],
            "avoid": [{"tool": "write_file", "caution": "read-only"}],
        }
        tools, audit = filter_tools_by_playbook(SAMPLE_TOOLS, guidance)
        names = {t["function"]["name"] for t in tools}
        assert "write_file" in names
        assert "write_file" in audit["avoid_tools"]

    def test_audit_records_playbook_tools(self):
        guidance = {
            "playbook": [
                {"tool": "search_files", "rationale": "find"},
                {"tool": "read_text_file", "rationale": "read"},
            ]
        }
        _, audit = filter_tools_by_playbook(SAMPLE_TOOLS, guidance)
        assert sorted(audit["playbook_tools"]) == [
            "read_text_file",
            "search_files",
        ]

    def test_unknown_playbook_tool_not_in_list(self):
        """Playbook may reference tools not in the provided list (e.g.
        memory tools). filter should not break."""
        guidance = {
            "playbook": [
                {"tool": "search_files", "rationale": "find"},
                {"tool": "nonexistent_tool", "rationale": "magic"},
            ]
        }
        tools, audit = filter_tools_by_playbook(SAMPLE_TOOLS, guidance)
        names = {t["function"]["name"] for t in tools}
        assert "search_files" in names
        # nonexistent_tool was in playbook but not in all_tools, so it's
        # simply not in the output (no crash).
        assert "nonexistent_tool" not in names
        assert audit["pruned"] is True

    def test_all_tools_in_playbook_keeps_all_relevant(self):
        guidance = {
            "playbook": [{"tool": t["function"]["name"], "rationale": ""}
                         for t in SAMPLE_TOOLS]
        }
        tools, audit = filter_tools_by_playbook(SAMPLE_TOOLS, guidance)
        assert len(tools) == len(SAMPLE_TOOLS)
        assert audit["removed"] == 0
        assert audit["pruned"] is True  # pruning was attempted, just nothing to prune


class TestEstimateSchemaTokens:
    def test_non_empty(self):
        tokens = estimate_schema_tokens(SAMPLE_TOOLS)
        assert tokens > 0
        # 8 tools with minimal schemas should be a few hundred tokens
        assert tokens > 50

    def test_empty(self):
        tokens = estimate_schema_tokens([])
        assert tokens >= 1

    def test_more_tools_more_tokens(self):
        small = estimate_schema_tokens(SAMPLE_TOOLS[:2])
        large = estimate_schema_tokens(SAMPLE_TOOLS)
        assert large > small
