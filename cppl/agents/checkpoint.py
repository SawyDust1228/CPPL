"""SQLite persistence for resumable CPPL design/module execution metadata."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class RunCheckpointStore:
    """Persist only JSON state; executable ModuleDef objects stay in process."""

    def __init__(self, path: Path, enabled: bool = True) -> None:
        self.enabled = enabled
        self.path = path
        self._lock = threading.Lock()
        if not enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS module_runs (
                    run_key TEXT NOT NULL,
                    module_name TEXT NOT NULL,
                    semantic_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (run_key, module_name)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30, check_same_thread=False)

    def load(self, run_key: str, module_name: str, semantic_key: str) -> dict | None:
        if not self.enabled:
            return None
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT semantic_key, payload FROM module_runs "
                "WHERE run_key=? AND module_name=?",
                (run_key, module_name),
            ).fetchone()
        if row is None or row[0] != semantic_key:
            return None
        try:
            value = json.loads(row[1])
        except (TypeError, ValueError):
            return None
        return value if isinstance(value, dict) else None

    def save(
        self,
        run_key: str,
        module_name: str,
        semantic_key: str,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO module_runs
                    (run_key, module_name, semantic_key, status, payload, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_key, module_name) DO UPDATE SET
                    semantic_key=excluded.semantic_key,
                    status=excluded.status,
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (run_key, module_name, semantic_key, status, encoded, time.time()),
            )


__all__ = ["RunCheckpointStore"]
