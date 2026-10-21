"""SQLite incident log. Writes are isolated so a database error cannot stop the loop."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_VERSION = 1
REVIEW_STATUSES = {"acknowledged", "false_positive", "confirmed"}
STATUSES = REVIEW_STATUSES | {"open", "expired"}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_ts REAL NOT NULL,
    ended_ts REAL,
    track_id INTEGER,
    seat TEXT,
    max_score REAL NOT NULL DEFAULT 0,
    mean_score REAL NOT NULL DEFAULT 0,
    score_count INTEGER NOT NULL DEFAULT 0,
    evidence_type TEXT,
    nir_mode TEXT,
    synthetic INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    reviewer_note TEXT,
    created_at TEXT NOT NULL,
    keyframe_sha256 TEXT,
    clip_sha256 TEXT,
    clip_container TEXT,
    evidence_status TEXT,
    keyframe_path TEXT,
    clip_path TEXT,
    suppress_until_ts REAL
);
CREATE TABLE IF NOT EXISTS incident_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER NOT NULL,
    ts REAL NOT NULL,
    state TEXT,
    score REAL,
    breakdown_json TEXT
);
CREATE TABLE IF NOT EXISTS config_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    incident_id INTEGER,
    ts REAL,
    config_json TEXT
);
"""


class IncidentStore:
    """Incident rows, events, and config snapshots in ``data/cineguard.db``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self.degraded = False
        self.error: str | None = None
        self._ready = False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as conn:
                self._migrate(conn)
            self._ready = True
        except Exception as exc:
            self._fail("incident database failed to open", exc)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        version = int(row["version"]) if row else 0
        if version < 1:
            conn.executescript(_SCHEMA)
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
        elif version < SCHEMA_VERSION:
            conn.execute("UPDATE schema_version SET version = ?", (SCHEMA_VERSION,))
        conn.commit()

    def _fail(self, message: str, exc: Exception) -> None:
        self.degraded = True
        self.error = f"{message}: {type(exc).__name__}: {exc}"
        logger.exception("%s (%s)", message, self.path)

    def is_suppressed(self, seat: str, now: float) -> bool:
        if not self._ready or not seat or seat == "Unmapped":
            return False
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT 1 FROM incidents
                    WHERE seat = ? AND status = 'false_positive' AND suppress_until_ts IS NOT NULL AND suppress_until_ts > ?
                    LIMIT 1
                    """,
                    (seat, now),
                ).fetchone()
            return row is not None
        except Exception as exc:
            self._fail("suppression lookup failed", exc)
            return False

    def open_or_update(
        self,
        *,
        now: float,
        track_id: int,
        seat: str,
        score: float,
        state: str,
        breakdown: dict[str, Any],
        evidence_type: str,
        nir_mode: str,
        synthetic: bool,
        config_snapshot: dict[str, Any],
        merge_window_s: float,
        prefer_id: int | None = None,
    ) -> int | None:
        """Create or refresh an incident. Returns None when the write fails."""
        if not self._ready:
            return None
        try:
            with self._lock, self._connect() as conn:
                incident_id = prefer_id if self._exists(conn, prefer_id) else None
                if incident_id is not None:
                    status = conn.execute("SELECT status FROM incidents WHERE id = ?", (incident_id,)).fetchone()
                    if status is not None and status["status"] == "false_positive":
                        incident_id = None
                if incident_id is None and seat and seat != "Unmapped":
                    incident_id = self._match_seat(conn, seat, now, merge_window_s)
                if incident_id is None:
                    created = datetime.now(timezone.utc).isoformat()
                    cursor = conn.execute(
                        """
                        INSERT INTO incidents (
                            started_ts, track_id, seat, max_score, mean_score, score_count,
                            evidence_type, nir_mode, synthetic, status, created_at, evidence_status
                        ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, 'open', ?, 'pending')
                        """,
                        (
                            now, track_id, seat, score, score, evidence_type, nir_mode,
                            1 if synthetic else 0, created,
                        ),
                    )
                    incident_id = int(cursor.lastrowid)
                    conn.execute(
                        "INSERT INTO config_snapshots (incident_id, ts, config_json) VALUES (?, ?, ?)",
                        (incident_id, now, json.dumps(config_snapshot, sort_keys=True)),
                    )
                else:
                    row = conn.execute(
                        "SELECT max_score, mean_score, score_count, synthetic FROM incidents WHERE id = ?",
                        (incident_id,),
                    ).fetchone()
                    count = int(row["score_count"]) + 1
                    mean = float(row["mean_score"]) + (score - float(row["mean_score"])) / count
                    conn.execute(
                        """
                        UPDATE incidents
                        SET ended_ts = NULL, track_id = ?, max_score = ?, mean_score = ?, score_count = ?,
                            synthetic = ?, status = CASE WHEN status = 'open' THEN 'open' ELSE status END
                        WHERE id = ?
                        """,
                        (
                            track_id,
                            max(float(row["max_score"]), score),
                            mean,
                            count,
                            1 if synthetic or row["synthetic"] else 0,
                            incident_id,
                        ),
                    )
                conn.execute(
                    "INSERT INTO incident_events (incident_id, ts, state, score, breakdown_json) VALUES (?, ?, ?, ?, ?)",
                    (incident_id, now, state, score, json.dumps(breakdown, sort_keys=True)),
                )
                conn.commit()
                return incident_id
        except Exception as exc:
            self._fail("incident write failed", exc)
            return None

    def close(self, incident_id: int, now: float, state: str = "IDLE") -> None:
        if not self._ready:
            return
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "UPDATE incidents SET ended_ts = ? WHERE id = ? AND ended_ts IS NULL",
                    (now, incident_id),
                )
                conn.execute(
                    "INSERT INTO incident_events (incident_id, ts, state, score, breakdown_json) VALUES (?, ?, ?, NULL, NULL)",
                    (incident_id, now, state),
                )
                conn.commit()
        except Exception as exc:
            self._fail(f"incident close failed for {incident_id}", exc)

    def set_evidence(self, incident_id: int, **fields: Any) -> None:
        if not self._ready:
            return
        allowed = {
            "keyframe_sha256", "clip_sha256", "clip_container", "evidence_status",
            "keyframe_path", "clip_path",
        }
        updates = {key: value for key, value in fields.items() if key in allowed and value is not None}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    f"UPDATE incidents SET {assignments} WHERE id = ?",
                    (*updates.values(), incident_id),
                )
                conn.commit()
        except Exception as exc:
            self._fail(f"evidence metadata write failed for {incident_id}", exc)

    def review(self, incident_id: int, status: str, note: str, now: float, suppress_s: float) -> dict[str, Any] | None:
        if status not in REVIEW_STATUSES:
            raise ValueError("status must be acknowledged, false_positive, or confirmed")
        if not self._ready:
            return None
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute("SELECT id FROM incidents WHERE id = ?", (incident_id,)).fetchone()
                if row is None:
                    return None
                suppress_until = now + suppress_s if status == "false_positive" else None
                conn.execute(
                    "UPDATE incidents SET status = ?, reviewer_note = ?, suppress_until_ts = ? WHERE id = ?",
                    (status, note, suppress_until, incident_id),
                )
                conn.commit()
            return self.get(incident_id)
        except ValueError:
            raise
        except Exception as exc:
            self._fail(f"incident review failed for {incident_id}", exc)
            return None

    def get(self, incident_id: int) -> dict[str, Any] | None:
        if not self._ready:
            return None
        try:
            with self._lock, self._connect() as conn:
                row = conn.execute("SELECT * FROM incidents WHERE id = ?", (incident_id,)).fetchone()
                if row is None:
                    return None
                events = conn.execute(
                    "SELECT id, ts, state, score, breakdown_json FROM incident_events WHERE incident_id = ? ORDER BY id",
                    (incident_id,),
                ).fetchall()
            payload = _public_row(row)
            payload["events"] = [_event(item) for item in events]
            return payload
        except Exception as exc:
            self._fail(f"incident read failed for {incident_id}", exc)
            return None

    def list_incidents(
        self,
        *,
        status: str | None = None,
        seat: str | None = None,
        since: float | None = None,
        until: float | None = None,
        synthetic: bool | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, Any]:
        if status is not None and status not in STATUSES:
            raise ValueError("unknown incident status")
        if limit < 1 or limit > 200:
            raise ValueError("limit must be between 1 and 200")
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if not self._ready:
            return {"items": [], "total": 0, "limit": limit, "offset": offset}
        clauses = ["1 = 1"]
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if seat:
            clauses.append("seat = ?")
            params.append(seat)
        if since is not None:
            clauses.append("started_ts >= ?")
            params.append(since)
        if until is not None:
            clauses.append("started_ts <= ?")
            params.append(until)
        if synthetic is not None:
            clauses.append("synthetic = ?")
            params.append(1 if synthetic else 0)
        where = " AND ".join(clauses)
        try:
            with self._lock, self._connect() as conn:
                total = conn.execute(f"SELECT COUNT(*) AS n FROM incidents WHERE {where}", params).fetchone()["n"]
                rows = conn.execute(
                    f"SELECT * FROM incidents WHERE {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                    (*params, limit, offset),
                ).fetchall()
            return {
                "items": [_public_row(row) for row in rows],
                "total": int(total),
                "limit": limit,
                "offset": offset,
            }
        except Exception as exc:
            self._fail("incident list failed", exc)
            return {"items": [], "total": 0, "limit": limit, "offset": offset}

    def export_csv(self) -> str:
        page = self.list_incidents(limit=200, offset=0)
        rows = list(page["items"])
        offset = 200
        while offset < page["total"]:
            more = self.list_incidents(limit=200, offset=offset)
            rows.extend(more["items"])
            offset += 200
        buffer = io.StringIO()
        fieldnames = [
            "id", "started_ts", "ended_ts", "track_id", "seat", "max_score", "mean_score",
            "evidence_type", "nir_mode", "synthetic", "status", "reviewer_note",
            "keyframe_sha256", "clip_sha256", "clip_container", "evidence_status",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return buffer.getvalue()

    def _exists(self, conn: sqlite3.Connection, incident_id: int | None) -> bool:
        if incident_id is None:
            return False
        return conn.execute("SELECT 1 FROM incidents WHERE id = ?", (incident_id,)).fetchone() is not None

    def _match_seat(self, conn: sqlite3.Connection, seat: str, now: float, window: float) -> int | None:
        row = conn.execute(
                    """
                    SELECT id FROM incidents
                    WHERE seat = ? AND status != 'false_positive'
                      AND (ended_ts IS NULL OR (? - ended_ts) <= ?)
                    ORDER BY id DESC LIMIT 1
                    """,
                    (seat, now, window),
                ).fetchone()
        return int(row["id"]) if row else None


def _public_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    data["synthetic"] = bool(data.get("synthetic"))
    incident_id = data["id"]
    data["keyframe_url"] = f"/api/v1/evidence/{incident_id}/keyframe.jpg"
    data["clip_url"] = f"/api/v1/evidence/{incident_id}/clip"
    data["download_url"] = f"/api/v1/evidence/{incident_id}/clip?download=1"
    return data


def _event(row: sqlite3.Row) -> dict[str, Any]:
    breakdown = None
    if row["breakdown_json"]:
        breakdown = json.loads(row["breakdown_json"])
    return {
        "id": row["id"],
        "ts": row["ts"],
        "state": row["state"],
        "score": row["score"],
        "breakdown": breakdown,
    }
