from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def fresh_memory_path(prefix: str, directory: Path = Path(".toolatlas")) -> Path:
    """Return a unique memory database path without creating or deleting anything."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return directory / f"{prefix}-{timestamp}-{uuid4().hex[:8]}.db"
