from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "2"


class SQLiteStore:
    """Small WAL-backed persistence layer for provider-side memory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
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
