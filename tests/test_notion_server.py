"""Unit tests for Notion MCP server and workspace simulator."""

from __future__ import annotations

import json
import pytest
from toolatlas.notion_server import NotionWorkspaceSimulator, create_notion_server


def test_notion_simulator_search():
    sim = NotionWorkspaceSimulator()
    results = sim.search("Engineering")
    assert len(results) >= 1
    assert results[0]["title"] == "Engineering Team Workspace"
    assert results[0]["id"] == "root-team-workspace-id"


def test_notion_simulator_create_page():
    sim = NotionWorkspaceSimulator()
    parent = {"page_id": "root-team-workspace-id"}
    props = {"title": {"title": [{"text": {"content": "Weekly Sync Meeting"}}]}}
    children = [
        {"type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": "Action items discussed."}}]}}
    ]
    page = sim.create_page(parent, props, children)
    assert page["id"].startswith("page-")
    assert page["title"] == "Weekly Sync Meeting"

    # Blocks verified
    blocks = sim.get_block_children(page["id"])
    assert len(blocks) == 1
    assert blocks[0]["type"] == "paragraph"


def test_notion_simulator_validation_error():
    sim = NotionWorkspaceSimulator()
    # Invalid parent
    with pytest.raises(ValueError):
        sim.create_page({}, {})

    # Invalid block payload
    with pytest.raises(ValueError):
        sim.append_block_children("root-team-workspace-id", [{"type": "paragraph"}])  # missing paragraph payload


def test_notion_simulator_database_query():
    sim = NotionWorkspaceSimulator()
    tasks = sim.query_database("db-task-tracker-id")
    assert len(tasks) == 1
    assert "Fix database connection leak" in str(tasks[0]["properties"])
