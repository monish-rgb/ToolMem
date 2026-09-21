"""Freeze a ToolAtlas memory database for evaluation.

Implements the plan freeze protocol without opening a writer on the frozen
copy: checkpoint and truncate WAL on the training database, copy it with the
SQLite backup API, validate with ``PRAGMA integrity_check``, hash both files,
and export logical statistics through a read-only connection.

Usage:
    python -m toolatlas.freeze_memory --source training.db --output frozen.db
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_stats(path: Path) -> dict[str, int | str]:
    uri = f"file:{path.resolve()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=10)
    try:
        connection.execute("PRAGMA query_only=ON")
        version = connection.execute(
            "SELECT value FROM metadata WHERE key='schema_version'"
        ).fetchone()
        stats: dict[str, int | str] = {
            "schema_version": version[0] if version else "unknown",
        }
        for table, key in (
            ("tools", "COUNT(*)"),
            ("traces", "COUNT(*)"),
            ("strategies", "COUNT(*)"),
            ("executions", "COUNT(*)"),
        ):
            stats[table] = connection.execute(f"SELECT {key} FROM {table}").fetchone()[0]
        return stats
    finally:
        connection.close()


def freeze_memory(source: str | Path, output: str | Path) -> dict:
    """Freeze *source* into *output* and return the freeze record."""
    src = Path(source)
    dst = Path(output)
    if not src.exists():
        raise FileNotFoundError(f"training database not found: {src}")
    if src.resolve() == dst.resolve():
        raise ValueError("source and output must be different files")
    dst.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict = {}
    connection = sqlite3.connect(src, timeout=30)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        checkpoint["wal_checkpoint"] = connection.execute(
            "PRAGMA wal_checkpoint(TRUNCATE)"
        ).fetchone()
        target = sqlite3.connect(dst, timeout=30)
        try:
            with target:
                connection.backup(target)
        finally:
            target.close()
    finally:
        connection.close()

    frozen = sqlite3.connect(f"file:{dst.resolve()}?mode=ro", uri=True, timeout=10)
    try:
        frozen.execute("PRAGMA query_only=ON")
        integrity = frozen.execute("PRAGMA integrity_check").fetchall()
    finally:
        frozen.close()
    if integrity != [("ok",)]:
        raise RuntimeError(f"frozen database failed integrity_check: {integrity!r}")

    record = {
        "source": str(src),
        "output": str(dst),
        "source_sha256": _sha256(src),
        "frozen_sha256": _sha256(dst),
        "integrity_check": "ok",
        "stats": _logical_stats(dst),
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze a ToolAtlas memory database.")
    parser.add_argument("--source", required=True, help="Training database path")
    parser.add_argument("--output", required=True, help="Frozen database output path")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(freeze_memory(args.source, args.output), indent=2))
    except (FileNotFoundError, ValueError, RuntimeError, sqlite3.Error) as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
