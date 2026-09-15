from __future__ import annotations

import re

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

mcp = MCPServer("toolatlas-text-tools")


@mcp.tool()
def normalize_text(text: str) -> str:
    """Lowercase text and collapse punctuation/whitespace into single spaces."""
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


@mcp.tool()
def word_count(text: str) -> int:
    """Count whitespace-delimited words in text."""
    return len(text.split())


@mcp.tool()
def keyword_count(text: str, keyword: str) -> int:
    """Count exact, case-insensitive keyword occurrences in normalized text."""
    if not keyword.strip():
        raise ToolError("keyword must not be empty")
    normalized_text = normalize_text(text)
    normalized_keyword = normalize_text(keyword)
    return normalized_text.split().count(normalized_keyword)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
