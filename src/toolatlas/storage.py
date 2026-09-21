from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "2"


class SQLiteStore:
    """Small WAL-backed persistence layer for provider-side memory.

    ``read_only=True`` opens the database through SQLite URI ``mode=ro`` with
    ``PRAGMA query_only=ON``: no directory creation, no schema
    initialization, no WAL-mode changes. Any write attempt raises.
    """

    def __init__(self, path: str | Path, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError(f"frozen memory database not found: {self.path}")
            with self.connect() as db:
                db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        if self.read_only:
            connection = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True, timeout=10)
            connection.execute("PRAGMA query_only=ON")
            return connection
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tools (
                    name TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS traces (
                    qid TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS strategies (
                    sid TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS executions (
                    execution_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    resolved INTEGER NOT NULL,
                    verified_at TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_executions_task ON executions(task_id);
                CREATE INDEX IF NOT EXISTS idx_traces_status ON traces(status);
                CREATE TABLE IF NOT EXISTS embeddings (
                    qid TEXT NOT NULL,
                    model TEXT NOT NULL,
                    version TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    vector TEXT NOT NULL,
                    PRIMARY KEY (qid, model, version)
                );
                """
            )
            db.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('schema_version', ?)",
                (SCHEMA_VERSION,),
            )

    def load(self) -> dict[str, dict[str, dict[str, Any]]]:
        result: dict[str, dict[str, dict[str, Any]]] = {
            "tools": {}, "traces": {}, "strategies": {}, "executions": {}
        }
        with self.connect() as db:
            for table, key in (
                ("tools", "name"), ("traces", "qid"),
                ("strategies", "sid"), ("executions", "execution_id")
            ):
                for identifier, data in db.execute(f"SELECT {key}, data FROM {table}"):
                    result[table][identifier] = json.loads(data)
        return result

    def save(self, payload: dict[str, dict[str, dict[str, Any]]]) -> None:
        if self.read_only:
            raise RuntimeError("frozen memory is read-only: mutation rejected")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for table in ("tools", "traces", "strategies", "executions"):
                db.execute(f"DELETE FROM {table}")
            db.executemany(
                "INSERT INTO tools(name, status, fingerprint, data) VALUES (?, ?, ?, ?)",
                [
                    (key, value["status"], value["schema_fingerprint"], json.dumps(value))
                    for key, value in payload["tools"].items()
                ],
            )
            db.executemany(
                "INSERT INTO traces(qid, status, confidence, data) VALUES (?, ?, ?, ?)",
                [
                    (key, value["status"], value["confidence"], json.dumps(value))
                    for key, value in payload["traces"].items()
                ],
            )
            db.executemany(
                "INSERT INTO strategies(sid, status, confidence, data) VALUES (?, ?, ?, ?)",
                [
                    (key, value["status"], value["confidence"], json.dumps(value))
                    for key, value in payload["strategies"].items()
                ],
            )
            db.executemany(
                "INSERT INTO executions(execution_id, task_id, resolved, verified_at, data) VALUES (?, ?, ?, ?, ?)",
                [
                    (
                        key, value["task_id"], int(value["resolved"]),
                        value["verified_at"], json.dumps(value),
                    )
                    for key, value in payload["executions"].items()
                ],
            )

    def save_embedding(self, qid: str, model: str, version: str,
                       vector: list[float]) -> None:
        """Cache one provider vector. Frozen (read-only) stores reject writes."""
        if self.read_only:
            raise RuntimeError("frozen memory is read-only: mutation rejected")
        if not qid.strip() or not vector:
            raise ValueError("qid and a non-empty vector are required")
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO embeddings(qid, model, version, dim, vector)"
                " VALUES (?, ?, ?, ?, ?)",
                (qid, model, version, len(vector), json.dumps(vector)))

    def get_embedding(self, qid: str, model: str,
                      version: str) -> list[float] | None:
        """Return the cached vector, or None when absent, stale, or pre-table.

        Version/model mismatches return None so retrieval falls back to
        lexical instead of comparing incomparable spaces.
        """
        try:
            with self.connect() as db:
                row = db.execute(
                    "SELECT dim, vector FROM embeddings"
                    " WHERE qid = ? AND model = ? AND version = ?",
                    (qid, model, version)).fetchone()
        except sqlite3.OperationalError:
            return None  # database predates the embeddings table
        if not row:
            return None
        try:
            vector = json.loads(row[1])
        except (TypeError, ValueError):
            return None
        if not isinstance(vector, list) or len(vector) != row[0]:
            return None
        return [float(value) for value in vector]
