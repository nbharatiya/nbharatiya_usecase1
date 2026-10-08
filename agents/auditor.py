"""SQLite run and stage audit trail."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator

from core.config import AUDIT_DB


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    AUDIT_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(AUDIT_DB, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            business_intent TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            error_message TEXT
        );
        CREATE TABLE IF NOT EXISTS stage_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            stage TEXT NOT NULL,
            status TEXT NOT NULL,
            rows_out INTEGER,
            quality_score REAL,
            error_message TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES runs(run_id)
        );
        """
    )
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def start_run(run_id: str, business_intent: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            "INSERT OR REPLACE INTO runs(run_id,business_intent,status,started_at,finished_at,error_message) VALUES(?,?,?,?,NULL,NULL)",
            (run_id, business_intent, "running", now),
        )


def finish_run(run_id: str, status: str, error_message: str | None = None) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE runs SET status=?,finished_at=?,error_message=? WHERE run_id=?",
            (status, datetime.now(timezone.utc).isoformat(), error_message, run_id),
        )


def log_stage_event(
    run_id: str,
    stage: str,
    status: str,
    rows_out: int | None = None,
    quality_score: float | None = None,
    error_message: str | None = None,
    started_at: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO stage_events(run_id,stage,status,rows_out,quality_score,error_message,started_at,finished_at) VALUES(?,?,?,?,?,?,?,?)",
            (run_id, stage, status, rows_out, quality_score, error_message, started_at or now, now),
        )


def get_stage_events(run_id: str) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM stage_events WHERE run_id=? ORDER BY event_id", (run_id,)
        ).fetchall()
    return [dict(row) for row in rows]


def get_recent_runs(n: int = 5) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (max(1, int(n)),)
        ).fetchall()
    return [dict(row) for row in rows]
