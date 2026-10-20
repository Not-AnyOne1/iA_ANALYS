"""Write-once SQLite tables for the strategy shadow evaluation.

They live in the existing database file (``STORAGE_DB_PATH``, on the Railway
volume), beside the live tables, which are never touched. Created with
``CREATE TABLE IF NOT EXISTS`` and versioned in ``strategy_shadow_meta``;
a later schema change is an additive migration keyed on that version.

``strategy_shadow_signals``    one row per signal, written once, at decision
                               time. ``signal_key`` ("<chat id>:<message id>")
                               is UNIQUE: a signal seen again after a restart
                               or a Telegram reconnect is not evaluated twice.
``strategy_shadow_decisions``  one row per (signal, strategy), written in the
                               same transaction as the signal.
``strategy_shadow_outcomes``   the original signal's hypothetical outcome,
                               one row per (signal, basis, revision). Revision
                               1 is the first resolution; a correction is a
                               new revision with its reason, so the original
                               stays readable.

Triggers refuse every UPDATE and DELETE on the three tables: a prediction
cannot be rewritten once outcomes are known.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from strategy_shadow.versions import ACCOUNTING_VERSION

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_BUSY_SECONDS = 10.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS strategy_shadow_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_shadow_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_key TEXT NOT NULL UNIQUE,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    signal_at TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    evaluated_at TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT,
    order_type TEXT,
    entries_json TEXT NOT NULL,
    entry_mid REAL,
    stop_loss REAL,
    take_profits_json TEXT NOT NULL,
    price_at_decision REAL,
    snapshot_fingerprint TEXT NOT NULL,
    data_status TEXT NOT NULL,
    data_quality_json TEXT NOT NULL,
    regime_json TEXT NOT NULL,
    live_verdict TEXT,
    live_setup_score INTEGER,
    live_reason TEXT,
    baseline_matches_live INTEGER,
    duplicate_of INTEGER,
    cohort TEXT NOT NULL,
    code_version TEXT NOT NULL,
    rules_hash TEXT NOT NULL,
    trade_json TEXT,
    untracked_reason TEXT,
    errors_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_strategy_shadow_signals_decided ON strategy_shadow_signals (decided_at);

CREATE TABLE IF NOT EXISTS strategy_shadow_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL REFERENCES strategy_shadow_signals (id),
    strategy TEXT NOT NULL,
    strategy_version TEXT NOT NULL,
    verdict TEXT,
    reason TEXT,
    base_score INTEGER,
    bonus INTEGER,
    penalty INTEGER,
    setup_score INTEGER,
    blocked INTEGER,
    score_label TEXT NOT NULL,
    error TEXT,
    UNIQUE (signal_id, strategy)
);

CREATE TABLE IF NOT EXISTS strategy_shadow_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id INTEGER NOT NULL REFERENCES strategy_shadow_signals (id),
    basis TEXT NOT NULL,
    revision INTEGER NOT NULL,
    status TEXT NOT NULL,
    r REAL,
    entry_fill REAL,
    exit_price REAL,
    filled_at TEXT,
    exit_at TEXT,
    cost_spread REAL,
    cost_slippage REAL,
    cost_source TEXT,
    accounting_version TEXT NOT NULL,
    note TEXT,
    resolved_at TEXT NOT NULL,
    correction_reason TEXT,
    UNIQUE (signal_id, basis, revision)
);
"""

_TRIGGERS = "".join(
    f"""
CREATE TRIGGER IF NOT EXISTS {table}_no_{op} BEFORE {op.upper()} ON {table}
BEGIN SELECT RAISE(ABORT, '{table} is write-once'); END;"""
    for table in ("strategy_shadow_signals", "strategy_shadow_decisions", "strategy_shadow_outcomes")
    for op in ("update", "delete"))

SIGNAL_COLUMNS = (
    "signal_key", "chat_id", "message_id", "signal_at", "decided_at", "evaluated_at", "symbol",
    "direction", "order_type", "entries_json", "entry_mid", "stop_loss", "take_profits_json",
    "price_at_decision", "snapshot_fingerprint", "data_status", "data_quality_json", "regime_json",
    "live_verdict", "live_setup_score", "live_reason", "baseline_matches_live", "duplicate_of",
    "cohort", "code_version", "rules_hash", "trade_json", "untracked_reason", "errors_json")
DECISION_COLUMNS = ("strategy", "strategy_version", "verdict", "reason", "base_score", "bonus",
                    "penalty", "setup_score", "blocked", "score_label", "error")


class StrategyShadowStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    @property
    def path(self) -> Path:
        return self._db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=_BUSY_SECONDS)
        conn.execute(f"PRAGMA busy_timeout={int(_BUSY_SECONDS * 1000)}")
        conn.row_factory = sqlite3.Row
        return conn

    async def _run(self, fn, *args):
        return await asyncio.to_thread(fn, *args)

    # ------------------------------------------------------------ schema

    async def initialize(self) -> int:
        """Create or migrate the tables; returns the schema version."""
        return await self._run(self._initialize_sync)

    def _initialize_sync(self) -> int:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA + _TRIGGERS)
            row = conn.execute("SELECT value FROM strategy_shadow_meta WHERE key = 'schema_version'").fetchone()
            current = int(row["value"]) if row else 0
            if current > SCHEMA_VERSION:
                raise RuntimeError(f"strategy shadow schema v{current} is newer than this code "
                                   f"(v{SCHEMA_VERSION}); refusing to write")
            # Additive migrations from `current` up to SCHEMA_VERSION go here.
            conn.execute("INSERT OR REPLACE INTO strategy_shadow_meta (key, value) VALUES "
                         "('schema_version', ?)", (str(SCHEMA_VERSION),))
            conn.commit()
            return SCHEMA_VERSION
        finally:
            conn.close()

    # ------------------------------------------------------------- write

    async def has(self, signal_key: str) -> bool:
        return await self._run(self._has_sync, signal_key)

    def _has_sync(self, signal_key: str) -> bool:
        conn = self._connect()
        try:
            return conn.execute("SELECT 1 FROM strategy_shadow_signals WHERE signal_key = ?",
                                (signal_key,)).fetchone() is not None
        finally:
            conn.close()

    async def record(self, signal: Dict[str, Any], decisions: List[Dict[str, Any]]) -> Optional[int]:
        """Signal and decisions in one transaction. Returns the new id, or
        None if this signal key was already recorded (the first is kept)."""
        return await self._run(self._record_sync, signal, decisions)

    def _record_sync(self, signal, decisions) -> Optional[int]:
        conn = self._connect()
        try:
            with conn:
                cur = conn.execute(
                    f"INSERT OR IGNORE INTO strategy_shadow_signals ({', '.join(SIGNAL_COLUMNS)}) "
                    f"VALUES ({', '.join('?' for _ in SIGNAL_COLUMNS)})",
                    [signal[c] for c in SIGNAL_COLUMNS])
                if cur.rowcount != 1:
                    return None
                signal_id = cur.lastrowid
                conn.executemany(
                    f"INSERT INTO strategy_shadow_decisions (signal_id, {', '.join(DECISION_COLUMNS)}) "
                    f"VALUES (?, {', '.join('?' for _ in DECISION_COLUMNS)})",
                    [[signal_id, *[d[c] for c in DECISION_COLUMNS]] for d in decisions])
            return signal_id
        finally:
            conn.close()

    async def record_outcome(self, signal_id: int, results: Dict[str, Any], resolved_at: datetime,
                             *, correction_reason: Optional[str] = None) -> int:
        """Final results, one row per basis. First resolution is revision 1;
        with ``correction_reason`` a new revision is appended instead."""
        return await self._run(self._record_outcome_sync, signal_id, results, resolved_at,
                               correction_reason)

    def _record_outcome_sync(self, signal_id, results, resolved_at, correction_reason) -> int:
        conn = self._connect()
        written = 0
        try:
            with conn:
                for basis, r in results.items():
                    if not r.final:
                        continue
                    last = conn.execute("SELECT MAX(revision) AS m FROM strategy_shadow_outcomes "
                                        "WHERE signal_id = ? AND basis = ?", (signal_id, basis)).fetchone()["m"]
                    if last is not None and correction_reason is None:
                        continue                       # already resolved; never overwritten
                    revision = 1 if last is None else last + 1
                    iso = lambda d: d.isoformat() if d is not None else None  # noqa: E731
                    conn.execute(
                        "INSERT INTO strategy_shadow_outcomes (signal_id, basis, revision, status, r, "
                        "entry_fill, exit_price, filled_at, exit_at, cost_spread, cost_slippage, "
                        "cost_source, accounting_version, note, resolved_at, correction_reason) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (signal_id, basis, revision, r.status, r.r, r.entry_fill, r.exit_price,
                         iso(r.filled_at), iso(r.exit_at), r.costs.spread, r.costs.slippage,
                         r.costs.source, ACCOUNTING_VERSION, r.note, resolved_at.isoformat(),
                         correction_reason))
                    written += 1
            return written
        finally:
            conn.close()

    # -------------------------------------------------------------- read

    async def recent_signals(self, since: datetime) -> List[Dict[str, Any]]:
        return await self._run(self._recent_sync, since)

    def _recent_sync(self, since: datetime) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            return [dict(r) for r in conn.execute(
                "SELECT id, signal_at, direction, stop_loss, take_profits_json, duplicate_of "
                "FROM strategy_shadow_signals WHERE signal_at >= ? ORDER BY id", (since.isoformat(),))]
        finally:
            conn.close()

    async def pending(self) -> List[Dict[str, Any]]:
        """Tracked signals without a final outcome on both bases."""
        return await self._run(self._pending_sync)

    def _pending_sync(self) -> List[Dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT s.id, s.trade_json FROM strategy_shadow_signals s WHERE s.trade_json IS NOT NULL "
                "AND (SELECT COUNT(DISTINCT basis) FROM strategy_shadow_outcomes o "
                "WHERE o.signal_id = s.id) < 2 ORDER BY s.id").fetchall()
            return [{"signal_id": r["id"], **json.loads(r["trade_json"])} for r in rows]
        finally:
            conn.close()

    async def load(self) -> Dict[str, List[Dict[str, Any]]]:
        """Every row, for the report (latest outcome revision only)."""
        return await self._run(self.load_sync)

    def load_sync(self) -> Dict[str, List[Dict[str, Any]]]:
        conn = self._connect()
        try:
            signals = [dict(r) for r in conn.execute("SELECT * FROM strategy_shadow_signals ORDER BY id")]
            decisions = [dict(r) for r in conn.execute("SELECT * FROM strategy_shadow_decisions")]
            outcomes = [dict(r) for r in conn.execute(
                "SELECT o.* FROM strategy_shadow_outcomes o JOIN (SELECT signal_id, basis, "
                "MAX(revision) AS rev FROM strategy_shadow_outcomes GROUP BY signal_id, basis) m "
                "ON o.signal_id = m.signal_id AND o.basis = m.basis AND o.revision = m.rev")]
            return {"signals": signals, "decisions": decisions, "outcomes": outcomes}
        finally:
            conn.close()


def load_readonly(db_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """The report's rows from a database opened read-only (safe on a copy of
    the live file, and on the live file itself)."""
    conn = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True, timeout=_BUSY_SECONDS)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "strategy_shadow_signals" not in tables:
            return {"signals": [], "decisions": [], "outcomes": []}
        signals = [dict(r) for r in conn.execute("SELECT * FROM strategy_shadow_signals ORDER BY id")]
        decisions = [dict(r) for r in conn.execute("SELECT * FROM strategy_shadow_decisions")]
        outcomes = [dict(r) for r in conn.execute(
            "SELECT o.* FROM strategy_shadow_outcomes o JOIN (SELECT signal_id, basis, "
            "MAX(revision) AS rev FROM strategy_shadow_outcomes GROUP BY signal_id, basis) m "
            "ON o.signal_id = m.signal_id AND o.basis = m.basis AND o.revision = m.rev")]
        return {"signals": signals, "decisions": decisions, "outcomes": outcomes}
    finally:
        conn.close()
