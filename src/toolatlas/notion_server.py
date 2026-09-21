"""Notion MCP Server and Workspace Simulator.

Provides standard Notion MCP tools:
- search: search pages, databases, or content
- get_page: retrieve page metadata and properties
- create_page: create a child page under a parent page or database
- update_page_properties: update page title, status, or properties
- get_block_children: list child blocks of a page/block
- append_block_children: add typed block children (headings, paragraphs, todos)
- query_database: query structured database entries with filter/sort

Includes a high-fidelity in-memory workspace simulator for fast, hermetic,
credential-free offline execution and unit testing.
"""

from __future__ import annotations

import json
import uuid
from typing import Any
from mcp.server import MCPServer

NOTION_PROVIDER = "io.github.modelcontextprotocol/server-notion"
NOTION_VERSION = "2025.1.0"


class NotionWorkspaceSimulator:
    """In-memory Notion workspace with schema validation."""

    def __init__(self) -> None:
        self.pages: dict[str, dict[str, Any]] = {}
        self.databases: dict[str, dict[str, Any]] = {}
        self.blocks: dict[str, list[dict[str, Any]]] = {}
        self._seed_default_workspace()

    def _seed_default_workspace(self) -> None:
        # Team workspace root
        root_id = "root-team-workspace-id"
        self.pages[root_id] = {
            "id": root_id,
            "title": "Engineering Team Workspace",
            "parent": {"type": "workspace", "workspace": True},
            "archived": False,
            "properties": {"title": {"title": [{"text": {"content": "Engineering Team Workspace"}}]}},
        }
        self.blocks[root_id] = [
            {"id": "b1", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": "Welcome to Engineering."}}]}}
        ]

        # Task tracker database
        db_id = "db-task-tracker-id"
        self.databases[db_id] = {
            "id": db_id,
            "title": "Project Task Tracker",
            "parent": {"page_id": root_id},
            "properties": {"Name": {"type": "title"}, "Status": {"type": "select"}, "Priority": {"type": "select"}},
        }
        # Seed tasks
        t1_id = "task-page-1"
        self.pages[t1_id] = {
            "id": t1_id,
            "parent": {"database_id": db_id},
            "properties": {
                "Name": {"title": [{"text": {"content": "Fix database connection leak"}}]},
                "Status": {"select": {"name": "Not Started"}},
                "Priority": {"select": {"name": "High"}},
            },
        }

    def search(self, query: str = "", filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        results = []
        q = query.lower()
        for p in self.pages.values():
            title = ""
            for prop in p.get("properties", {}).values():
                if "title" in prop and prop["title"]:
                    title = prop["title"][0].get("text", {}).get("content", "")
                    break
            if not q or q in title.lower():
                results.append({"object": "page", "id": p["id"], "title": title, "parent": p.get("parent")})
        for d in self.databases.values():
            if not q or q in d.get("title", "").lower():
                results.append({"object": "database", "id": d["id"], "title": d.get("title", ""), "parent": d.get("parent")})
        return results

    def get_page(self, page_id: str) -> dict[str, Any]:
        if page_id in self.databases:
            raise ValueError(f"ID {page_id} is a database, not a page. Call query_database instead.")
        if page_id not in self.pages:
            raise KeyError(f"Page not found: {page_id}")
        return self.pages[page_id]

    def create_page(self, parent: dict[str, Any], properties: dict[str, Any], children: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        if not isinstance(parent, dict) or not (parent.get("page_id") or parent.get("database_id")):
            raise ValueError("Validation error: parent must specify 'page_id' or 'database_id'")
        new_id = f"page-{uuid.uuid4().hex[:8]}"
        title_text = "Untitled"
        for prop in properties.values():
            if isinstance(prop, dict) and "title" in prop and prop["title"]:
                title_text = prop["title"][0].get("text", {}).get("content", "Untitled")
                break
        page = {
            "id": new_id,
            "parent": parent,
            "properties": properties,
            "title": title_text,
            "archived": False,
        }
        self.pages[new_id] = page
        self.blocks[new_id] = []
        if children:
            self.append_block_children(new_id, children)
        return page

    def update_page_properties(self, page_id: str, properties: dict[str, Any]) -> dict[str, Any]:
        if page_id not in self.pages:
            raise KeyError(f"Page not found: {page_id}")
        page = self.pages[page_id]
        page.setdefault("properties", {}).update(properties)
        return page

    def get_block_children(self, block_id: str) -> list[dict[str, Any]]:
        return self.blocks.get(block_id, [])

    def append_block_children(self, block_id: str, children: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(children, list) or not children:
            raise ValueError("Validation error: children must be a non-empty list of typed block objects")
        validated = []
        for b in children:
            b_type = b.get("type")
            if not b_type or b_type not in b:
                raise ValueError(f"Malformed block: block must have 'type' and typed payload dictionary for '{b_type}'")
            b_copy = dict(b)
            b_copy.setdefault("id", f"block-{uuid.uuid4().hex[:6]}")
            validated.append(b_copy)
        self.blocks.setdefault(block_id, []).extend(validated)
        return validated

    def query_database(self, database_id: str, filter: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if database_id not in self.databases:
            raise KeyError(f"Database not found: {database_id}")
        entries = [p for p in self.pages.values() if p.get("parent", {}).get("database_id") == database_id]
        return entries


def create_notion_server(simulator: NotionWorkspaceSimulator | None = None) -> MCPServer:
    """Factory to create Notion MCPServer."""
    sim = simulator or NotionWorkspaceSimulator()
    server = MCPServer("notion-mcp-server")

    @server.tool()
    def search(query: str = "") -> str:
        """Search pages, databases, or content by text title."""
        results = sim.search(query=query)
        return json.dumps(results)

    @server.tool()
    def get_page(page_id: str) -> str:
        """Retrieve metadata, parent, and properties of a Notion page."""
        try:
            return json.dumps(sim.get_page(page_id))
        except Exception as exc:
            return f"Notion API error: {exc}"

    @server.tool()
    def create_page(parent: str, properties: str, children: str = "[]") -> str:
        """Create a new page. parent and properties must be valid JSON strings."""
        try:
            parent_dict = json.loads(parent) if isinstance(parent, str) else parent
            props_dict = json.loads(properties) if isinstance(properties, str) else properties
            children_list = json.loads(children) if isinstance(children, str) and children else []
            page = sim.create_page(parent_dict, props_dict, children_list)
            return json.dumps(page)
        except Exception as exc:
            return f"Notion API error: {exc}"

    @server.tool()
    def update_page_properties(page_id: str, properties: str) -> str:
        """Update properties of a page (e.g. status, tags). properties must be a JSON string."""
        try:
            props_dict = json.loads(properties) if isinstance(properties, str) else properties
            updated = sim.update_page_properties(page_id, props_dict)
            return json.dumps(updated)
        except Exception as exc:
            return f"Notion API error: {exc}"

    @server.tool()
    def get_block_children(block_id: str) -> str:
        """Get child blocks of a page or block."""
        return json.dumps(sim.get_block_children(block_id))

    @server.tool()
    def append_block_children(block_id: str, children: str) -> str:
        """Append typed block children to a page or block. children must be a JSON string array."""
        try:
            children_list = json.loads(children) if isinstance(children, str) else children
            appended = sim.append_block_children(block_id, children_list)
            return json.dumps(appended)
        except Exception as exc:
            return f"Notion API error: {exc}"

    @server.tool()
    def query_database(database_id: str, filter: str = "{}") -> str:
        """Query entries in a structured Notion database."""
        try:
            filter_dict = json.loads(filter) if isinstance(filter, str) and filter else {}
            results = sim.query_database(database_id, filter_dict)
            return json.dumps(results)
        except Exception as exc:
            return f"Notion API error: {exc}"

    return server
